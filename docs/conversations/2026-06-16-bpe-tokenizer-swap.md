# 2026-06-16 — Swap tiktoken → trained byte-level BPE (small vocab)

## Problem

Model had **35.3M params** vs only **268k training tokens** (~2600:1, Chinchilla
wants ~20:1) → severe overfit. Root cause: `r50k_base` vocab (50,257) made the
`vocab × embed_dim` token-embed + output-head tables ~73% of all params. Shrinking
`embed_dim`/`num_layers` barely helped because those two big terms scale with vocab.

## Decision

- Keep BPE, but **train our own small-vocab tokenizer** on the dataset instead of `r50k_base`.
- Rejected: char-level (user didn't want it); `bpe-tokenizer` crate (default-small = 100k,
  *bigger*, and returns strings not ids); karpathy/rustbpe (training is Python-only, and our
  old `tiktoken` crate can't load a custom vocab anyway).
- Chosen: HuggingFace **`tokenizers` crate** (pure Rust, trains on a file, emits integer ids).
- `VOCAB_SIZE = 1024`. Byte-level base = 256, so ~767 merges + 1 special.
- Only special token: `<|endoftext|>` (decoder-only LM needs no pad/unk/mask; byte-level has no OOV).
- Train tokenizer on the **training split only** (split raw text before tokenizing) to avoid leakage.

## Changes

- `Cargo.toml`: removed `tiktoken`, added `tokenizers = "0.23.1"`.
- `src/config.rs`: `VOCAB_SIZE` 256 → 1024; removed `ENCODING_NAME`.
- `src/tokenizer.rs`: `train_tokenizer(train_file)` (byte-level BPE, saves `tokenizer.json`)
  + `load_tokenizer()`. Note: `ByteLevel::alphabet()` returns `AHashSet`, needs
  `.into_iter().collect()` into `std::HashSet` for `initial_alphabet`.
- `src/data_loader.rs`: `get_train_test_data` now splits raw text at a char boundary, writes
  train split to `./dataset/train_split.txt`, trains tokenizer on it, encodes both splits.
- `src/main.rs`: `vocab_size` now from `tokenizer.get_vocab_size(true)`, not tiktoken.
- `.gitignore`: added `/tokenizer.json`. CLAUDE.md key facts updated.
- Also (earlier): user fixed a newline bug — encode whole file, not line-by-line (`.lines()`
  was dropping `\n`).

## Concepts clarified

- GPT-style training concatenates all text into one flat token stream, then cuts fixed
  `CONTEXT_SIZE` windows — so variable line lengths need **no padding**.
- `vocab_size` = total unique tokens INCLUDING the base alphabet. With byte-level base 256,
  `vocab_size=256` leaves zero room for merges (= plain byte-level).

## Status

`cargo build --release` passes (only pre-existing dead-code warnings). Not yet run/trained.

## Follow-ups

- Run training, watch train-vs-val gap with the smaller model.
- Consider caching: skip retraining the tokenizer if `tokenizer.json` already exists.
- Model is still ~big for the data even after vocab shrink — may want fewer layers / smaller embed.
