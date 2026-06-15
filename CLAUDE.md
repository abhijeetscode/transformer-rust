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
| `src/data_loader.rs`        | Tokenization, train/test split, batching                |
| `src/config.rs`             | Device selection, context size, tokenizer name          |

## Key facts

- Device defaults to **Metal** (`src/config.rs`); swap to `Device::Cpu` there to run on CPU.
- Rust **edition 2024** — needs a recent toolchain (1.85+).
- Tokenizer is `r50k_base` BPE via the `tiktoken` crate.
- Default model shape (≈ GPT-2 small): `embed_dim` 768, `num_heads` 12, `num_layers` 12,
  `context_size` 1024 — set in `src/main.rs` and `src/config.rs`.
- Training data lives in `./dataset/` (git-ignored). Model checkpoints write to
  `best_model.safetensors` (also git-ignored).

## Build / run

```bash
cargo run --release
```
