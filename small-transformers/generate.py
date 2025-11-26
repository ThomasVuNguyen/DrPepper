"""Text generation script for checkpoints produced by train_model.py."""

from __future__ import annotations

import argparse
import torch

from train_model import GPTConfig, GPTSmall, SentencePieceTokenizer, load_checkpoint


def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Filter a distribution for nucleus/top-k sampling."""
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, -1e10), sorted_logits)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    return logits


def generate(
    model: GPTSmall,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[-model.config.block_size :]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.block_size :]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    return tokenizer.decode(out_ids)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument(
        "--tokenizer",
        help="Path to tokenizer.model (defaults to value stored in checkpoint)",
        default=None,
    )
    p.add_argument("--prompt", default="", help="Prompt text to start generation")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    config = GPTConfig.from_dict(ckpt["config"])

    tok_path = args.tokenizer or ckpt.get("extra", {}).get("tokenizer_path")
    if not tok_path:
        raise ValueError("Tokenizer path must be provided (flag or checkpoint extra)")
    tokenizer = SentencePieceTokenizer(tok_path)

    model = GPTSmall(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device("cpu"))

    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
