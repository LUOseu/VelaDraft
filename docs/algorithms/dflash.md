# DFlash (reference algorithm, frozen)

Status: **stable / validated** on Qwen2.5-VL-3B and Qwen3-VL-8B (single image).

DFlash is this repo's reference block-parallel drafter: every anchor slot is
denoised with a mask token in one block forward, supervised by a decayed block
cross-entropy over slots `1..block_size-1` (slot 0 is the clean anchor and is
not supervised). The implementation is a port of the read-only reference
`jacky566/Automodel_lao` @ `d21fb5aa` (Apache-2.0, NeMo-AutoModel DFlash
modules); the port map and the exact training semantics are documented in
`docs/REFERENCE.md` and that file is the source of truth.

Freeze policy (v0.2):

- The training objective, block layout, acceptance-length definition and
  checkpoint field names stay unchanged; results remain comparable to every
  run recorded in `BENCHMARKS.md`.
- Config knobs (`drafter.*`) may be added, but existing keys keep their
  semantics.
- The algorithm is exposed through the registry as `dflash`
  (`vlm_dflash/algorithms/dflash.py`); the implementation modules stay in
  `vlm_dflash/dflash/`.

## What a checkpoint records

Written by `train` (see `vlm_dflash/dflash/train.py`), verified by `eval`:

- `algorithm: "dflash"` — algorithm identity (eval/export hard-fail on mismatch);
- `draft_config`, `attention_backend`, `mask_token_id`, `block_size`,
  `num_anchors`, `loss_decay_gamma`, `target_layer_ids`;
- `head` (`tied_embedding` / `target_lm_head`) — must match the target's
  runtime head tie state;
- `identity` — the target/processor fingerprint chain.

## Evidence

Measured acceptance (sequential verification, greedy, token-exact):

- Qwen2.5-VL-3B, 512-sample corpus (rerun 2026-10-11): AL 1.198, 64/64 exact;
- Qwen2.5-VL-3B, 2K: AL 1.630, 128/128 exact, per-slot
  `[.413, .149, .048, .015, .004, .001, 0]`;
- Qwen3-VL-8B, 512-sample corpus: AL 1.322.

See `BENCHMARKS.md` for the table and `docs/REFERENCE.md` for the runtime
verification-mode details.
