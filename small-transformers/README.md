# Small CPU-Only Transformer Trainer

Train a GPT-style language model on a large `.txt` corpus using only a 12-core CPU and 24GB RAM. This document outlines the recommended architecture, data pipeline, and training loop so you can implement the script confidently.

## Constraints and Targets
- Hardware: 12 vCPUs, 24GB RAM, no GPU.
- Model size: aim for ~40-70M parameters so training fits in RAM.
- Dataset: a very large plain text file; should be processed with streaming/batching to avoid loading it all at once.

## Recommended Stack
- Python 3.10+.
- PyTorch (CPU-only build), `tqdm` for progress, `sentencepiece` or `tiktoken` for tokenization.

Install (CPU-only):
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm sentencepiece
# or: pip install tiktoken if you prefer OpenAI BPE
```

## Repository Layout
- `train.py`: stream a huge text file, tokenize on the fly, and train a small GPT model (CPU-only).
- `generate.py`: load a checkpoint and sample text.
- `README.md`: this guide.

## Data Pipeline
1. **Prepare the raw text**  
   - Place your corpus at `data/corpus.txt` (any path is fine; adjust flags below).
   - Clean up with simple rules (optional): lowercase, normalize whitespace, drop empty lines.

2. **Train or load a tokenizer**  
   - SentencePiece BPE example:
     ```bash
     spm_train --input=data/corpus.txt --model_prefix=tokenizer --vocab_size=32000 --character_coverage=1.0 --model_type=bpe
     ```
   - This produces `tokenizer.model` and `tokenizer.vocab`.

3. **Stream and chunk the data**  
   - Use an `IterableDataset` that:
     - Opens the file once,
     - Reads in moderately sized blocks (e.g., 8-16MB),
     - Tokenizes on the fly,
     - Yields contiguous blocks of length `block_size` (e.g., 256-512 tokens) with next-token targets.
   - Avoid loading the whole file into RAM; rely on streaming + small buffers.

## Model Configuration (fits in 24GB RAM)
Example GPT-like config:
- `vocab_size`: 32000
- `n_layer`: 8
- `n_head`: 8
- `d_model`: 512
- `d_ff`: 2048
- `block_size`: 256-512
- Dropout: 0.1
- Weight tying between token embedding and output head to reduce params.

This is roughly ~45M parameters. With AdamW and reasonable batch sizes it runs within 24GB on CPU.

## Training Loop (implemented in `train.py`)
`train.py` provides streaming data loading, a small decoder-only Transformer, gradient accumulation, warmup+cosine LR, gradient clipping, and checkpointing. Configure everything via CLI flags.

Key training tips for CPU:
- Use mixed precision **only if** your CPU supports bfloat16/AVX512; otherwise stay in float32.
- Keep `batch_size` small (e.g., 8-16 sequences) and rely on gradient accumulation to reach an effective batch size of ~256 sequences.
- Clip gradients (e.g., `max_norm=1.0`).
- Warmup LR schedule (e.g., 1k-5k steps) then cosine decay.
- Checkpoint every N steps to resume easily; store model + optimizer + scheduler states.

Example run command (once `train.py` exists):
```bash
python train.py \
  --data data/corpus.txt \
  --tokenizer tokenizer.model \
  --vocab-size 32000 \
  --n-layer 8 --n-head 8 --d-model 512 --d-ff 2048 \
  --block-size 256 \
  --batch-size 12 --grad-accum 20 \
  --lr 3e-4 --warmup-steps 2000 --max-steps 500000 \
  --checkpoint-dir ckpts/
```

### Checkpointing and resume
Checkpoints are saved to `ckpts/step-XXXXXXX.pt` by default and contain model/optimizer/scheduler state plus the config. Resume with `--resume ckpts/step-XXXXXXX.pt`.

## Evaluation
- Hold out a small slice of the corpus as validation (e.g., last 1-2%).
- Compute perplexity every few hundred steps; stop if it plateaus.
- Optionally sample text from the model to sanity-check.

## Inference
After training, run a simple generation script:
```bash
python generate.py --checkpoint ckpts/step-500000.pt --prompt "Once upon a time"
```
You can override decoding hyperparameters (temperature/top-k/top-p) and tokenizer path (defaults to the path stored in the checkpoint):
```bash
python generate.py \
  --checkpoint ckpts/step-500000.pt \
  --prompt "The future of open models" \
  --max-new-tokens 120 \
  --temperature 0.9 --top-k 40 --top-p 0.95
```

## Next Steps
- Add a tiny unit test that runs a 1-step training pass on a toy string to ensure the pipeline works.
- Add CLI shortcuts or YAML/JSON config loading for experiment tracking.
- Optionally plug in tiktoken if you prefer its tokenizer, or add support for more eval metrics.
