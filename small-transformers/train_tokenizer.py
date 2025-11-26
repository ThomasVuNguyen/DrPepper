"""Train a SentencePiece tokenizer for use with train_model.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    p.add_argument(
        "--input",
        type=str,
        default="data/corpus.txt",
        help="Path to raw text corpus",
    )
    p.add_argument(
        "--model-prefix",
        type=str,
        default="data/tokenizer",
        help="Prefix for the output files (creates .model and .vocab)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    p.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Amount of characters covered by the model (1.0 for full coverage)",
    )
    p.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="SentencePiece model type",
    )
    p.add_argument(
        "--input-sentence-size",
        type=int,
        default=2_000_000,
        help="Number of sentences sampled for training",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Threads to use (defaults to all available CPUs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.input)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    prefix_path = Path(args.model_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Hint sentencepiece/OpenMP to use the requested threads.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.num_threads))

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "sentencepiece is required. Install with `pip install sentencepiece`."
        ) from exc

    options = [
        f"--input={corpus_path}",
        f"--model_prefix={prefix_path}",
        f"--vocab_size={args.vocab_size}",
        f"--character_coverage={args.character_coverage}",
        f"--model_type={args.model_type}",
        f"--input_sentence_size={args.input_sentence_size}",
        "--shuffle_input_sentence=true",
        "--pad_id=0",
        "--unk_id=1",
        "--bos_id=2",
        "--eos_id=3",
        f"--num_threads={args.num_threads}",
    ]

    print(
        f"[info] Training tokenizer to {prefix_path}.model with vocab_size={args.vocab_size} "
        f"using {args.num_threads} thread(s). SentencePiece will print progress with ETA."
    )
    spm.SentencePieceTrainer.Train(" ".join(options))
    print("[info] Done. Outputs:")
    print(f"  {prefix_path}.model")
    print(f"  {prefix_path}.vocab")


if __name__ == "__main__":
    main()
