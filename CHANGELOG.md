# Changelog

## v0.3.0 (in development)

Algorithm zoo stage 2: DSpark.

Added
- **DSpark** (`algorithm: dspark`, knobs under `algorithm.dspark`): DFlash
  backbone + low-rank Markov logit-bias head (vanilla/gated/rnn) + per-slot
  acceptance-confidence head; semi-autoregressive in-block sampling; upstream
  default `sample_from_anchor: true` (block_size proposals per round) with the
  DFlash-compatible `false` mode as a directly comparable ablation. Port notes
  in `docs/upstream/dspark.md`, usage in `docs/algorithms/dspark.md`.
- `sample_from_anchor` on the frozen DFlash draft model/trainer (default
  `false`; DFlash semantics unchanged) so both slot conventions share one
  verification loop; DSpark checkpoints record the mode.
- **Gradient coverage audit** (`utils.gradient_coverage_report`): per-group
  trainable params / params-with-grad / nonzero-grad / grad-norm, logged at
  step 1 of every training run and written to `train/gradient_coverage.json`.
- Confidence calibration artifact for confidence-head algorithms
  (`train/confidence_calibration.json`: BCE, 10-bin ECE, AUROC vs greedy slot
  acceptance, per-slot predicted/observed acceptance).
- Namespaced algorithm config (`algorithm: {name: ..., <algorithm>: {...}}`)
  alongside the legacy string form; export loading now dispatches on the
  recorded algorithm instead of refusing non-DFlash exports.
- Upstream audit docs for all five algorithms (`docs/upstream/*.md`) plus
  `UPSTREAM_AUDIT.md` / `ALGORITHM_MATRIX.md` governance indexes.
- `configs/dspark/` (2K canonical + COCO smoke).

Changed
- `DFlashDecayLoss` takes `first_slot` (default 1 = DFlash); the trainer seam
  `_apply_algorithm_terms` lets algorithms add logit corrections/auxiliary
  losses without touching the frozen DFlash path.

Fixed
- **DSpark heads are owned by the draft model, not duplicated by the trainer.**
  The duplicate heads kept fp32 weights while the backbone was cast to bf16
  (`RuntimeError: mat1 and mat2 must have the same dtype`), and — the more
  dangerous half — were invisible to the optimizer's master-parameter set and
  to the saved `draft_model.state_dict()`, so head weights would have trained
  on nothing and reloaded at their random init. `DSparkTrainerModule` now binds
  `draft_model.markov_head` / `confidence_head` (single source of truth) and
  validates explicit knob overrides against the model config. Regression tests:
  ownership identity, bf16 cast propagation, disagreement and bare
  `Qwen3DFlashDraftModel` rejection; `gradient_coverage_report` now reports
  `draft_model.<child>` groups so a head cannot hide inside the backbone line.

## v0.2.0 (in development)

Qwen3-VL support, verification modes, scaling evidence, and the algorithm
registry.

Added
- Qwen3-VL-8B-Instruct target adapter (interleaved MRoPE, untied head handled
  end to end: untied targets train against their own `lm_head`).
- Unified CLI: `vlm-dflash {doctor,train,eval,export}` on top of the staged
  scripts; presets `smoke` / `single-gpu-24g` / `single-gpu-48g`.
- Verification modes `sequential` / `parallel` / `strict_parallel` with a
  measured runtime probe: parallel verify is **not bitwise-lossless**
  (documented in `docs/REFERENCE.md`); sequential remains the correctness gate.
- Resume (model/optimizer/scheduler/RNG), self-describing HF-style export,
  `scripts/summarize_runs.py`, and the scaling suite script.
- **Algorithm registry** (`vlm_dflash/algorithms/`): DFlash is now a plugin
  behind a documented interface; checkpoints carry an `algorithm` tag and
  eval/export refuse mismatched checkpoints. No DFlash semantics changed.
- **Benchmark harness skeleton** (`scripts/benchmark.py`,
  `benchmarks/manifests/`): fixed-protocol per-workload reports (JSON + MD).
- Governance docs: `STATE.md`, `SUPPORT_MATRIX.md`, `BENCHMARKS.md`,
  `docs/adding_an_algorithm.md`, `docs/algorithms/dflash.md`,
  `docs/benchmark.md`.
- `pyproject.toml` version aligned with the README (`0.2.0`).

Measured
- Qwen2.5-VL-3B 2K-corpus run: AL 1.630, 128/128 token-exact, per-slot
  acceptance non-zero through slot 5; 5K run in progress.
- Qwen3-VL-8B 512-corpus run: AL 1.322 (sequential verification).

Known limitations
- Single-image only; no video; no serving integration.
- bf16 capture/training is protocol-reproducible, not bit-identical
  (run-to-run AL noise < ~2%).
- Parallel verification is experimental; do not quote it as lossless speedup.

## v0.1.0

First minimal vertical slice: Qwen2.5-VL-3B, single image, target-native
regen, staged fused hidden-state capture, DFlash drafter training (target off
GPU), token-exact acceptance eval, provenance gates, visual-path checks.
