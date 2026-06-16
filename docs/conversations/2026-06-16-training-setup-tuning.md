# 2026-06-16 — Training setup review & tuning (after BPE swap)

Follows [BPE tokenizer swap](2026-06-16-bpe-tokenizer-swap.md). User started training and
asked for a review + tuned things one lever at a time.

## Decisions / changes

- **Model size cut for the dataset:** `NUM_LAYERS` 12 → 4, `BATCH_SIZE` 3 → 16
  (`EMBED_DIM` 256, `NUM_HEADS` 8, `CONTEXT_SIZE` 256, `VOCAB_SIZE` 1024). ~35M → ~3.6M params.
  12 layers was GPT-2-small depth, overkill for ~270k tokens (overfit + slow). nanoGPT
  char-Shakespeare reference is ~6 layers.
- **Per-epoch loss logging (perf):** replaced per-batch `loss.to_scalar()` (forces a GPU→CPU
  sync every step on Metal) with a detached on-GPU running-sum tensor, synced once per epoch.
  `Tensor::detach()` is infallible (returns `Tensor`, no `?`). User implemented this themselves.
- **Three-way split:** `data_loader::get_train_test_data` → `get_data_splits(filename,
  train_perc, eval_perc)` returning `(train, eval, test)` (80/10/10). Tokenizer still trained on
  train split only. `main.rs`: `let mut varmap`; early stopping now uses `eval_data`; after the
  loop `varmap.load("best_model.safetensors")` reloads the BEST checkpoint and evaluates on
  `test_data` exactly once → prints final held-out test loss.

## Decided NOT to add (yet) — run first, add only if symptoms appear

- **Gradient clipping:** low risk at 4 layers. Add only if loss spikes/NaN early. Method if
  needed: `loss.backward()?` → scale grads in the `GradStore` by `max_norm/global_norm` →
  `opt.step(&grads)?` (splits `backward_step`). Costs one CPU sync/step (or do it sync-free with
  a clamped clip-coef tensor).
- **Dropout:** none in the network. Main overfit lever, but requires threading a `train: bool`
  through every `forward` (must be OFF in eval). Add only if eval loss climbs while train drops.

## Verified facts (candle 0.10.2, git 39355c6)

- `VarMap::save(&self)`, `VarMap::load(&mut self)` — load needs `mut`.
- `Tensor::detach(&self) -> Tensor`. `GradStore`: `get/insert/remove/get_ids`. `Optimizer::step(&grads)`.
- Leakage audit: no train→eval/test leak. Splits disjoint; tokenizer train-only; eval/test do
  no `backward`. Note: eval set drives early stopping (it's a validation set), test is pristine.

## Status

`cargo build --release` passes. Not yet trained — waiting on user to run and share loss curves.

## Follow-ups

- Run, read train/eval curves + final test loss; decide if dropout/clipping needed.
- Tokenizer still retrains every run — consider skip-if-`tokenizer.json`-exists cache.
- `evaluate_model` takes `&Vec<u32>` (clippy: prefer `&[u32]`); hardcodes `chunks(3)`.
