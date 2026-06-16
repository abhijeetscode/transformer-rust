# transformer-rust

GPT-style decoder-only transformer written from scratch in Rust on top of
[candle](https://github.com/huggingface/candle), trained on text (Tiny Shakespeare by
default) with Metal GPU acceleration on macOS.

## Commit rules

- **Do NOT add a `Co-Authored-By` trailer (or any AI attribution) to commit messages.**
- Keep commit messages concise and descriptive of the change.

## Conversation summaries

- Always write a summary of the important points from each conversation to
  `docs/conversations/`. Name files `YYYY-MM-DD-<short-topic>.md`. Capture
  decisions, changes made, and open follow-ups — not every message.

## Project layout

| File                        | Responsibility                                          |
| --------------------------- | ------------------------------------------------------- |
| `src/main.rs`               | Training loop, evaluation, early stopping, checkpointing |
| `src/gpt_model.rs`          | Full GPT model (embeddings → blocks → output head)      |
| `src/transformer_block.rs`  | Pre-norm transformer block + feed-forward network       |
| `src/attention.rs`          | Single attention head + multi-head causal attention     |
| `src/tokenizer.rs`          | Train/load byte-level BPE tokenizer (HF `tokenizers`)   |
| `src/data_loader.rs`        | Raw-text split, tokenizer training, encoding, batching  |
| `src/config.rs`             | Device selection, context size, `VOCAB_SIZE`            |

## Key facts

- Device defaults to **Metal** (`src/config.rs`); swap to `Device::Cpu` there to run on CPU.
- Rust **edition 2024** — needs a recent toolchain (1.85+).
- Tokenizer is a **byte-level BPE** trained on the *training split only* via the HuggingFace
  `tokenizers` crate; `VOCAB_SIZE` (default 1024) caps unique tokens. Base alphabet is the
  256 bytes, so merges = `VOCAB_SIZE - 256 - specials`. Saved to `tokenizer.json`.
- Vocab is kept small on purpose: embedding/output tables scale with `vocab_size × embed_dim`,
  so a 50k vocab dominated params for a ~268k-token dataset.
- Training data lives in `./dataset/` (git-ignored); the training split is also written to
  `./dataset/train_split.txt`. Model checkpoints write to `best_model.safetensors`,
  tokenizer to `tokenizer.json` (all git-ignored).

## Build / run

```bash
cargo run --release
```
