# Changelog

## v0.3.0 (in development)

### Release engineering (v0.3.0-rc1)

- Five-minute demo: `vlm-dflash demo` (tiny license-free synthetic sample,
  token-exact evaluation, random-init acceptance baseline or
  `--drafter <run|ckpt|export>` weights) and `vlm-dflash train-demo` (full
  staged loop + tiny training + eval + export on the same sample).
  `--drafter` demos align the sample's image pixel bounds with the drafter's
  recorded target identity (the provenance gate stays hard) and use a
  drafter-specific run directory.
- Public benchmark: TextVQA adapter (`vlm_dflash/benchmark/adapters.py`,
  `scripts/prepare_benchmark.py`) with a pinned 128-prompt seeded subset,
  per-shard/per-row hashes and a repo-tracked prompt manifest
  (`benchmarks/prompts/textvqa_v1.jsonl`); `02_textvqa.json` is now
  `validated`. Benchmark manifests resolve prompt paths against the repo root.
- Packaging: wheel + sdist build cleanly, fresh-venv install smoke-tested;
  explicit `pyarrow` dependency; `[dev]` extras; version `0.3.0`.
- Export metadata extension: exports now also carry `draft_config.json`,
  `algorithm.json` (with the machine-readable support statement),
  `target_contract.json` and `training_manifest.json` (trainable parameter
  count, GPU); `scripts/export_zoo.py` builds and reload-verifies the six
  canonical zoo checkpoints.
- CI (`.github/workflows/ci.yml`): CPU tests on py3.10/3.12 + package build +
  fresh wheel install + CLI help smoke; issue templates for bugs / algorithm
  and target requests / reproduction reports / runtime compatibility.
- Repo hygiene: CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, CITATION.cff;
  absolute development-host paths removed from configs, presets, tests and
  docs (env-var defaults only); secrets/large-file/license/claim audits
  recorded in `RELEASE_VALIDATION.md`.
- Qwen3-VL reduced validation for DSpark / Domino / JetSpec / DFlash2
  (256-row slice, 16-prompt sequential eval, export + reload) — results in
  `RELEASE_VALIDATION.md`.
- Frozen-reference regression re-run on the release commit (DFlash 2K
  baseline, `tests/reference_dflash/manifest.json`).

Four algorithm-zoo stages land in this release: DSpark (stage 2), Domino
(stage 3), JetSpec (stage 4) and DFlash2 (stage 5).

### Stage 5: DFlash2 (local convolutions + candidate selector, Experimental)

- Upstream audit (`docs/upstream/dflash2.md`): architecture pinned to
  z-lab/dflash `dflash/model.py` @ 07ebd93d via speculators @ 55ebbd5b; the
  training objective is **not published upstream**, so the port implements the
  maintainer implementation's experimental split objective and ships as
  Experimental. (Also corrects the old NeMo pin: `NVIDIA-NeMo/Automodel @
  bb5b34ce` has no dflash2.)
- Added `vlm_dflash/dflash2/`: `GroupedDynamicCausalConv` (grouped,
  token-conditioned, block-causal local convolution with identity init),
  `CandidateSelector`, conv-wrapped decoder layer, `Qwen3DFlash2DraftModel`
  (serving = one parallel draft forward + a sequential top-K candidate walk),
  `DFlash2TrainerModule` (unary forward KL with the exact staged teacher +
  K-way selector CE with rank-K target injection, fixed-exp decay, masked-count
  denominator; serving metrics gated on the un-injected candidate set), factory
  and plugin. NOTICE records the MIT attribution for the ported modules.
- `_make_decoder_layer` hook added to the DFlash draft model (default
  unchanged); `doctor --algorithm dflash2` reports the full-vocab requirement;
  eval summaries record `draft_candidate_selector`; export records
  `method_details` + a runtime note.
- 24 new tests (scalar-loop references for conv/selector math, identity-init
  equivalence with DFlash, serving walk + selector-override checks, objective
  plumbing, fail-loud gates, draft-config round trip).
- COCO smoke: full chain (prepare -> regen -> capture with visual check ->
  train -> eval), token-exact 8/8, AL 1.362 vs init 1.000. 2K canonical:
  AL 1.954 token-exact 128/128, slots `[.495,.243,.118,.060,.027,.013,.006]`;
  selector serving-ablation 2.060 vs 1.965 (unary-only ranking).
- DFlash2 decomposition: local-mixing + selector ablation
  (`conv_kernel_size: 1`, `selector_loss_alpha: 0.0`, same recipe/seed) drops
  AL to 1.798 but still beats DSpark/Domino, bounding the joint conv+selector
  share at +0.156; the objective axis (CE -> forward-KL) stays entangled and
  is not claimed. Selector override rate on the ablated checkpoint is 0.0
  (knob verified off). See `BENCHMARKS.md` for the full table.
- Frozen DFlash reference: `tests/reference_dflash/manifest.json` records the
  pre-zoo baselines (3B-2K AL 1.630, 3B-5K 1.817, 8B-512 1.322, 8B-2K 1.665),
  checkpoint/feature semantics and the regression gates; post-stage
  re-measurements after DSpark..DFlash2 land are recorded there and in
  `STATE.md` (8-prompt registry path reproduced bit-identically at 1.6096).

### Stage 4: JetSpec (causal block-parallel drafting + forward-KL distillation)

- Capture v2: `capture.store_final_hidden` stores the target's final hidden
  state per sample (manifest: `stores_final_hidden`); feature collate carries it
  as the distillation teacher signal. Verified exact: `lm_head(final_hidden)`
  equals the target's own logits bit-for-bit.
- `vlm_dflash/jetspec/`: `kd_loss.py` (temperature-scaled forward KL with
  ignore-index and chunking; ported from the maintainer `KDLoss`),
  `trainer.py` (`causal_in_block` mask, teacher gather at
  `label_indices[..., :-1]`, chunked teacher projection through the frozen
  lm_head, draft-vs-teacher agreement + acceptance metrics), `factory.py`,
  `algorithms/jetspec.py` plugin + configs. 19 new tests.
- Serving fidelity: `inference_in_block_causal` draft-config flag makes
  `spec_generate` serve JetSpec checkpoints with the training causal in-block
  mask. Measured cost of the mismatched runtime: -0.26 AL (7.6% argmax flips on
  the 2K checkpoint). Eval JSONs record the effective serving mode; export
  configs carry the flag and a runtime note (parallel-tree runtime not ported).
- 2K canonical run: 792 steps, KD loss 6.06 -> 3.10 (first/last-10 mean),
  overfit sanity pass, coverage 36/36, visual check pass; eval AL 1.599
  (token-exact 128/128) under causal-mask serving vs 1.338 under the shared
  runtime. Peak training memory 29.7 GB (KD backward retains fp32
  log-probabilities; overfit sanity batch reduced to 8).

### Stage 3: Domino

Added
- **Domino** (`algorithm: domino`, knobs under `algorithm.domino`): DFlash
  backbone + causal GRU correction head (`prefix_gru` + low-rank `embed_proj`
  logit delta applied on the corrected suffix) and the base-anchor curriculum
  `loss = (1 - lambda_base) * final_loss + lambda_base * base_loss` with a
  linear decay of `lambda_base` over `lambda_base_decay_ratio` of the run.
  Defaults follow the maintainer VL recipe (`emb_dim: 256`,
  `gru_hidden_dim: 1024`, `pure_draft_prefix_len: 1`, `shift_label: true`,
  `lambda_base_start: 1.0`, `lambda_base_decay_ratio: 1.0`). Audit in
  `docs/upstream/domino.md`, usage in `docs/algorithms/domino.md`.
- Ablation-only `algorithm.domino.enable_correction_head` switch (not an
  upstream knob): trains the exact DFlash objective on the same architecture
  so the required enabled/disabled comparison isolates the head.
- Shared trainer seams for causal-correction algorithms: `_block_loss_step`
  (dual-logit scoring under one mask/decay convention; underscore-prefixed
  metrics stay internal) and a `set_training_progress(step, total_steps)`
  curriculum hook; `coverage_audit_fraction` lets curriculum-gated heads audit
  gradient coverage after their warm-up instead of at step 0.
- `configs/domino/` (2K canonical + COCO smoke).

Changed
- `gradient_coverage.json` gains additive `audit_step` / `audit_fraction`
  fields and a generic `source`; DFlash/DSpark still audit at the first
  optimizer step, Domino at 10% of the run (λ=1 at step 0 intentionally gives
  the correction head no gradient).

### Stage 2: DSpark

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
- **Verify-backend parity now binds every algorithm, not just DFlash.** The
  base-class `configure_verify_backend` was a no-op, so the evaluator's
  `verify_sdpa_backend: parity` never disabled the reference verify-time SDPA
  narrowing for DSpark: verification ran on the narrowed [EFFICIENT+MATH]
  kernels while `generate()` used flash, and the resulting ~1-ULP bf16 KV
  drift flipped a near-tie at token 8 of `coco_000000562157_686936` (the
  token-exact gate failed on 1/8 COCO smoke prompts for both the trained and
  the random-init checkpoint). The switch now lives in `DrafterAlgorithm`
  (single source of truth for the shared `spec_generate` path) and
  `eval_result.json` records the effective state as
  `verify_backend_switch_enabled`.

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
