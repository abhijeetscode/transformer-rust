# transformer-rust

A GPT-style decoder-only transformer, implemented from scratch in Rust on top of
[candle](https://github.com/huggingface/candle). It trains a character/BPE language
model on text data (Tiny Shakespeare by default) using Metal GPU acceleration on macOS.

This is a learning project: every core piece of the architecture — multi-head causal
self-attention, the transformer block, token/position embeddings, and the training loop —
is written explicitly rather than pulled from a prebuilt model.

## Architecture

The model is a standard GPT decoder stack:

```
tokens ──▶ token embedding  ┐
                            ├─(+)─▶ [ Transformer Block ] × N ─▶ LayerNorm ─▶ Linear ─▶ logits
positions ─▶ position embedding ┘
```

Each **Transformer Block** is pre-norm with residual connections:

```
x ─▶ LayerNorm ─▶ Multi-Head Masked Attention ─▶ (+x) ─▶ LayerNorm ─▶ FFN ─▶ (+) ─▶ out
```

- **Multi-head masked (causal) attention** — `src/attention.rs`. Each head computes scaled
  dot-product attention with a lower-triangular causal mask so a position can only attend to
  itself and earlier positions. Heads are concatenated and projected back to `embed_dim`.
- **Feed-forward network** — two linear layers with a 4× hidden expansion and GELU activation.
- **Embeddings + head** — `src/gpt_model.rs` adds token and learned positional embeddings,
  runs the block stack, applies a final LayerNorm, and projects to vocabulary logits.

### Default hyperparameters

| Parameter      | Value       | Where             |
| -------------- | ----------- | ----------------- |
| `embed_dim`    | 768         | `src/main.rs`     |
| `num_heads`    | 12          | `src/main.rs`     |
| `num_layers`   | 12          | `src/main.rs`     |
| `context_size` | 1024        | `src/config.rs`   |
| tokenizer      | `r50k_base` | `src/config.rs`   |
| optimizer      | AdamW, lr 3e-4 | `src/main.rs`  |

(This is roughly GPT-2 small in shape.)

## Project layout

| File                        | Responsibility                                          |
| --------------------------- | ------------------------------------------------------- |
| `src/main.rs`               | Training loop, evaluation, early stopping, checkpointing |
| `src/gpt_model.rs`          | Full GPT model (embeddings → blocks → output head)      |
| `src/transformer_block.rs`  | Pre-norm transformer block + feed-forward network       |
| `src/attention.rs`          | Single attention head + multi-head causal attention     |
| `src/data_loader.rs`        | Tokenization, train/test split, batching                |
| `src/config.rs`             | Device selection, context size, tokenizer name          |

## Requirements

- Rust (with `edition = "2024"`, so a recent toolchain — Rust 1.85+)
- macOS with a Metal-capable GPU (the device defaults to Metal)

To run on CPU instead, edit `src/config.rs` and swap the device line:

```rust
DEVICE.get_or_init(|| Device::Cpu)
```

## Dataset

Training data is read from `./dataset/tiny_shakespere.txt`. The `dataset/` directory is
git-ignored, so download a corpus into it before training, e.g.:

```bash
mkdir -p dataset
curl -o dataset/tiny_shakespere.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The loader tokenizes the file with the `r50k_base` BPE encoding and splits it 90/10 into
train/test (`get_train_test_data` in `src/data_loader.rs`).

## Running

```bash
cargo run --release
```

Training runs for up to 50 epochs with **early stopping** (patience 3 on validation loss).
Each epoch prints train and validation loss; whenever validation loss improves, the weights
are checkpointed to `best_model.safetensors`.

```
train_data len: ... test_data len: ...
Epoch 1 Train Loss 5.42 Val Loss 5.31
new best (val 5.3100), saved best_model.safetensors
...
early stopping — val loss not improving
```

## Notes

- The full attention scores tensor is materialized per head, so memory scales with
  `context_size²`. For experimentation, lower `CONTEXT_SIZE` in `src/config.rs`.
- Built as an educational reference for how a GPT is wired together end to end in Rust.
