"""CPU-only GPT-style language model trainer.

This script streams a large text file, tokenizes on the fly with
SentencePiece (or a compatible tokenizer exposing `encode`), and trains a
small decoder-only Transformer. It is sized to fit on a 12-core CPU with
24GB RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset


# ------------------------- Tokenizer helpers -------------------------


class SentencePieceTokenizer:
    """Wrapper around a SentencePiece model for simple encode/decode."""

    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(model_path):
            raise ValueError(f"Failed to load tokenizer at {model_path}")

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)


# ------------------------- Dataset -------------------------


class TextDataset(IterableDataset):
    """Streams a text file and yields token blocks.

    The dataset reads the file in fixed-size byte chunks to avoid loading the
    whole corpus. It tokenizes each chunk and yields non-overlapping blocks of
    size `block_size`, paired with the next-token targets.
    """

    def __init__(
        self,
        path: str,
        tokenizer: SentencePieceTokenizer,
        block_size: int,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.chunk_bytes = chunk_bytes

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        tail: List[int] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            while True:
                chunk = fh.read(self.chunk_bytes)
                if not chunk:
                    break

                token_ids = tail + self.tokenizer.encode(chunk)
                num_full = (len(token_ids) - 1) // self.block_size
                usable = num_full * self.block_size + 1
                for i in range(0, usable - 1, self.block_size):
                    x = token_ids[i : i + self.block_size]
                    y = token_ids[i + 1 : i + 1 + self.block_size]
                    yield torch.tensor(x, dtype=torch.long), torch.tensor(
                        y, dtype=torch.long
                    )

                # keep the tail to prepend to the next chunk
                tail = token_ids[usable - 1 :]


# ------------------------- Model -------------------------


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 3
    n_head: int = 8
    d_model: int = 256
    d_ff: int = 1024
    block_size: int = 256
    dropout: float = 0.1
    weight_tie: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "GPTConfig":
        return GPTConfig(**data)


@dataclass
class TrainConfig:
    data: str
    tokenizer: str
    vocab_size: int
    val_data: Optional[str] = None
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    weight_tie: bool = True
    batch_size: int = 12
    grad_accum: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 1000
    eval_batches: int = 20
    checkpoint_dir: str = "ckpts"
    resume: Optional[str] = None
    num_workers: int = 0
    seed: int = 42
    save_interval: int = 1000
    chunk_bytes: int = 8 * 1024 * 1024

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        allowed = {f.name for f in fields(TrainConfig)}
        unknown = [k for k in raw.keys() if k not in allowed]
        if unknown:
            print(f"[warn] Ignoring unknown config keys: {unknown}")
        filtered = {k: v for k, v in raw.items() if k in allowed}
        try:
            return TrainConfig(**filtered)
        except TypeError as exc:
            raise ValueError(
                f"Invalid config at {path}. Check required fields and types."
            ) from exc


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff)
        self.proj = nn.Linear(config.d_ff, config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTSmall(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tie:
            self.head.weight = self.token_emb.weight

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)[None, :, :]
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


# ------------------------- Training loop -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: GPTSmall,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    config: GPTConfig,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def make_scheduler(max_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def evaluate(
    model: GPTSmall,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
            if count >= max_batches:
                break
    model.train()
    return total_loss / max(1, count)


def cycle(loader: DataLoader) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small GPT model on CPU")
    p.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file (defaults to config.json)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional override for resume checkpoint path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    if args.resume:
        cfg.resume = args.resume

    if not Path(cfg.data).is_file():
        raise FileNotFoundError(
            f"Training data not found at {cfg.data}. Update the path in {args.config}."
        )
    if cfg.val_data and not Path(cfg.val_data).is_file():
        raise FileNotFoundError(
            f"Validation data not found at {cfg.val_data}. Update the path in {args.config}."
        )
    if not Path(cfg.tokenizer).is_file():
        raise FileNotFoundError(
            f"Tokenizer model not found at {cfg.tokenizer}. Update the path in {args.config}."
        )

    set_seed(cfg.seed)

    device = torch.device("cpu")

    tokenizer = SentencePieceTokenizer(cfg.tokenizer)
    if tokenizer.vocab_size != cfg.vocab_size:
        print(
            f"[warn] tokenizer vocab_size={tokenizer.vocab_size} "
            f"differs from provided config vocab_size={cfg.vocab_size}. Using tokenizer value."
        )
    vocab_size = tokenizer.vocab_size

    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        weight_tie=cfg.weight_tie,
    )

    model = GPTSmall(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_scheduler(cfg.max_steps, cfg.warmup_steps),
    )

    start_step = 0
    if cfg.resume:
        checkpoint = load_checkpoint(cfg.resume)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_step = checkpoint.get("step", 0)
        print(f"[info] Resumed from {cfg.resume} at step {start_step}")

    train_ds = TextDataset(
        cfg.data,
        tokenizer=tokenizer,
        block_size=cfg.block_size,
        chunk_bytes=cfg.chunk_bytes,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    train_iter = iter(cycle(train_loader))

    val_loader = None
    if cfg.val_data:
        val_ds = TextDataset(
            cfg.val_data,
            tokenizer=tokenizer,
            block_size=cfg.block_size,
            chunk_bytes=cfg.chunk_bytes,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # CPU-only

    model.train()
    loop_start_time = time.time()
    for step in range(start_step, cfg.max_steps):
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(train_iter)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 10 == 0 or step == start_step:
            lr = scheduler.get_last_lr()[0]
            steps_done = step - start_step + 1
            elapsed = time.time() - loop_start_time
            avg_time_per_step = elapsed / max(1, steps_done)
            remaining_steps = cfg.max_steps - (step + 1)
            eta = remaining_steps * avg_time_per_step
            print(
                f"step {step+1}/{cfg.max_steps} | loss {loss_accum:.4f} | lr {lr:.6f} "
                f"| avg_step {avg_time_per_step:.2f}s | eta {format_seconds(eta)}"
            )

        if val_loader and (step + 1) % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches)
            print(f"[eval] step {step+1}: val_loss={val_loss:.4f}")

        if (step + 1) % cfg.save_interval == 0 or (step + 1) == cfg.max_steps:
            ckpt_path = (
                Path(cfg.checkpoint_dir)
                / f"step-{step+1:07d}.pt"
            )
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                step=step + 1,
                config=config,
                extra={"tokenizer_path": cfg.tokenizer},
            )
            print(f"[ckpt] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
