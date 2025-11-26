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
from dataclasses import asdict, dataclass
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
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    block_size: int = 256
    dropout: float = 0.1
    weight_tie: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "GPTConfig":
        return GPTConfig(**data)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small GPT model on CPU")
    p.add_argument("--data", required=True, help="Path to training text file")
    p.add_argument(
        "--val-data", help="Optional path to validation text file", default=None
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        help="Path to SentencePiece model (.model)",
    )
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--grad-accum", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--checkpoint-dir", type=str, default="ckpts")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    if tokenizer.vocab_size != args.vocab_size:
        print(
            f"[warn] tokenizer vocab_size={tokenizer.vocab_size} "
            f"differs from provided --vocab-size={args.vocab_size}. Using tokenizer value."
        )
    vocab_size = tokenizer.vocab_size

    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff=args.d_ff,
        block_size=args.block_size,
        dropout=args.dropout,
    )

    model = GPTSmall(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_scheduler(args.max_steps, args.warmup_steps),
    )

    start_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_step = checkpoint.get("step", 0)
        print(f"[info] Resumed from {args.resume} at step {start_step}")

    train_ds = TextDataset(
        args.data,
        tokenizer=tokenizer,
        block_size=args.block_size,
        chunk_bytes=args.chunk_bytes,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    train_iter = iter(cycle(train_loader))

    val_loader = None
    if args.val_data:
        val_ds = TextDataset(
            args.val_data,
            tokenizer=tokenizer,
            block_size=args.block_size,
            chunk_bytes=args.chunk_bytes,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # CPU-only

    model.train()
    for step in range(start_step, args.max_steps):
        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = next(train_iter)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 10 == 0 or step == start_step:
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step+1}/{args.max_steps} | loss {loss_accum:.4f} | lr {lr:.6f}"
            )

        if val_loader and (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, args.eval_batches)
            print(f"[eval] step {step+1}: val_loss={val_loss:.4f}")

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.max_steps:
            ckpt_path = (
                Path(args.checkpoint_dir)
                / f"step-{step+1:07d}.pt"
            )
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                step=step + 1,
                config=config,
                extra={"tokenizer_path": args.tokenizer},
            )
            print(f"[ckpt] saved to {ckpt_path}")


if __name__ == "__main__":
    main()
