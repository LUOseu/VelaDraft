# Changelog

## v0.3.2 (in development)

- Repository renamed to `github.com/LUOseu/VelaDraft` (old `vlm-dflash` URLs
  redirect); the distribution name and CLI stay `vlm-dflash`.

### CI / packaging fix: `num2words` dependency

- `num2words` is now a declared runtime dependency (`pyproject.toml`): the
  transformers SmolVLM processor requires it at construction time (so the
  SmolVLM2 target path needs it), and the CPU-test CI environment - which
  installs `.[tested,dev]` - was missing it, erroring all five
  `tests/test_smolvlm_adapter.py` tests with `ImportError`.

### Schema v2: multi-image / multi-turn

- Data schema v2: rows with `image_turns` (or the `vlm-dflash-data-v2`
  marker) keep the full conversation and a flat `images` list; every image is
  rendered at the start of the turn it is attached to. v1 rows and every v1
  record stay byte-compatible (`vlm_dflash/data/schema.py`,
  `data/sharegpt.py`, `data/prepare.py`).
- Regen generates each assistant turn greedily conditioned on the
  previously *generated* turns, stores per-turn ids plus an incremental
  `full_ids`, and asserts the concatenation against one full-conversation
  render (fail-loud on chat-template drift). Answers that hit
  `max_new_tokens` get the primary stop appended (counted as `forced_stop` in
  `regen_stats.json`).
- Capture records `image_spans` / `n_images` / per-image expected token
  counts, `turn_spans` and a multi-segment `loss_mask`; `index.jsonl` rows
  gain `image_spans` + `n_turns`, manifests gain `schema_version` +
  `num_multi_turn`. The visual-path swap check, eval prompts, online digest
  and FeatureDataset/OnlineFeatureDataset pass the new fields through;
  collate and trainers are unchanged (loss_mask/image_mask were already
  multi-segment capable).
- Algorithms declare `multi_image: true`; the support-matrix footnote scopes
  the evidence.
- Tests: 14 new CPU tests (`tests/test_wp2_multiturn.py`, 349 total).
- Recorded evidence: E2 smoke `R-20261014-08` - synthetic 2-image / 2-turn
  JSONL (16 train + 4 val) through prepare -> regen -> capture ->
  32-step train -> sequential verify; capture visual swap check passed
  (512/512 image tokens across two spans, replay bitwise equal) and eval
  token-exact 4/4 for both `ckpt_final` and `ckpt_init`
  (`configs/smoke_multiturn.yaml`). Smoke-scale only; no quality or speed
  claim follows from it.

### Phase-1 audit deliverables: training signals + multimodal supervision

- Every algorithm now declares its training-signal requirements
  (`TrainingSignals` on `AlgorithmMetadata`: hard label, context layers,
  teacher final hidden states, distribution / confidence objective, stage,
  block convention), rendered to `docs/TRAINING_SIGNALS.md` by
  `scripts/render_training_signals.py` with a drift test so the declaration
  cannot silently diverge from the metadata.
- The multimodal supervision representation gains its phase-1 design note
  (`docs/plan/WP2_DESIGN.md`: the field-by-field producer/consumer contract,
  including the deferred multi-window data-unit rule) and
  `tests/test_multimodal_supervision.py`: supervision never lands on
  `loss_mask == 0` positions, seeded forwards are deterministic, fully
  unsupervised padding rows are inert, image tokens drive the loss and the
  visual gradients while text-only batches leave the visual branch an exact
  no-op, image spans that would leak future turns are rejected, and
  visual-conditioned runs resume exactly (a changed visual branch is
  refused) and export/reload with the visual branch intact.
- The export config and model card record the draft's visual conditioning
  (`visual_conditioning`: cached vectors per layer + adapter type), so a
  serving stack can see that the checkpoint expects image-token spans.
- The route-document §15 audit package lands self-contained in
  `docs/plan/phase1/`: LOCAL_PILOT_REPORT, CLAIM_EVIDENCE (P1-P9),
  RUN_MANIFEST, RESOURCE_ESTIMATE, PROFILING, CACHE_DECISION,
  COMPETITIVE_AUDIT and GO_NO_GO (verdict: NO-GO for the formal
  matrix / big-compute request; the bounded pending list is the
  recommended next spend).
- The online-extraction profiler gains a `--length-buckets` per-row mode
  (median render/forward per token-length bucket, plus
  `forward_ms_per_token`), CPU-tested; the P2 length-distribution
  measurement and the second-seed learning repeat are registered as
  GPU_PENDING #6/#7 in `docs/plan/REVIEW_REMEDIATION.md`.

### DSpark faithful objective (`train_recipe: upstream_dspark`)

- New selectable recipe reproducing the speculators DSpark default objective
  (`vllm-project/speculators @ 55ebbd5b`): a single stage of per-slot forward
  `KL(p_target || p_draft)` over the full vocabulary (decay-weighted, no
  top-k truncation) plus the confidence BCE against the analytic acceptance
  rate `sum_v min(p_draft, p_target) = 1 - TV`.
- The teacher distribution is **reconstructed on demand** from the stored
  target final hidden states through the frozen output head
  (`TeacherLogitsStream`, position `label - 1`, bounded fp32 chunks with
  activation checkpointing). The KL term and the confidence target share one
  teacher chunk stream per step; the confidence target is computed from the
  live (Markov-corrected) drafter logits, detached, and never cached.
- `algorithm.train_recipe` support is now a per-recipe supports-matrix
  (`RECIPE_SUPPORT`): `upstream_dspark` is accepted only for `dspark`,
  `upstream_two_stage` only for `dflash`/`dflare`/`dfly`; every pairing and
  option key outside the matrix fails loud. Recipe options
  (`cold_start_fraction`, `confidence_coef`) are validated per recipe.
- The recipe enters the DSpark training-plan digest (byte-identical digests
  for runs that do not select it), the checkpoint metadata, and the
  config-validation gates: `capture.store_final_hidden=true` is required, the
  confidence head must be enabled, and `algorithm.dspark.confidence_head_alpha`
  must stay at the upstream default 1.0 (`train_recipe.confidence_coef` is the
  tuning knob).
- Plain DSpark runs are unchanged (decayed CE + greedy per-slot acceptance
  indicator). Config: `configs/dspark/qwen2_5_vl_3b_2k_recipe.yaml`; CPU
  evidence: objective vs direct full-vocab reference (loss + gradients),
  chunk invariance, detached analytic target, trainer-level gradient coverage
  and config gates in `tests/test_objectives.py` / `tests/test_dspark.py`.

### DocVQA / ChartQA benchmark adapters

- `scripts/prepare_benchmark.py` gains `--workload docvqa` (`lmms-lab/DocVQA`
  DocVQA/validation) and `--workload chartqa` (`lmms-lab/ChartQA` test):
  deterministic 128-prompt seeded exports (seed 42) with per-shard /
  per-row question/image hashes, pinned prompt set definitions under
  `benchmarks/prompts/`, images named by content suffix. Adapter export is
  covered by CPU tests on synthetic parquet fixtures.
- Manifests `03_docvqa` / `04_chartqa` are `validated`: E3
  (`R-20261014-10`) ran both 128-prompt / 64-token protocols against the
  frozen 3B/2K DFlash checkpoint - DocVQA AL 1.201, ChartQA AL 1.221, both
  token-exact 128/128 (sequential parity). Numbers in `BENCHMARKS.md`.

### Batched online capture

- `feature_mode=online` gains an opt-in batched capture forward
  (`train.online_batch_size: N`, default 1 = the per-sample reference path):
  rows are rendered per sample, one padded target forward serves the batch,
  and each record keeps its row's own length / spans (padding never reaches a
  record). The trainer uses an internal iterator that reproduces the
  per-sample `DataLoader` order (same seeded `RandomSampler` + `BatchSampler`
  sequence), so enabling batching does not change batch composition.
- `train_summary.json` / the online manifest now expose
  `online_batch_size` and `target_samples` / `target_samples_total` next to
  the forward counters (they diverge only when batching is on); a batch size
  > 1 is folded into the online source digest so exact resume cannot switch
  grouping. CPU tests pin batched-vs-per-sample record equality on a tiny
  Qwen2.5-VL target (single-image + schema-v2 multi-image/multi-turn rows)
  and DataLoader order parity across epochs.

### DFlare / DFly (AngelSpec ports)

- New algorithms `dflare` and `dfly`, ported from `Tencent/AngelSpec @
  d3412bed` (MIT) behind the existing algorithm interface:
  - DFlare: per-draft-layer learnable softmax fusion over the captured target
    layers (distance-prior init, upstream semantics) + dedicated
    `k_proj_target`/`v_proj_target` context projections;
  - DFly: DFlash shared-KV layers + FC context + fusion residual + zero-init
    TreeFlash hidden-states correction applied before the frozen LM head
    (`sample_from_anchor=false`); the upstream two-stage LK/D-PACE objective
    and e2e-TV are ported as the config-selectable `algorithm.train_recipe`
    (see the subsection below; documented in `docs/upstream/dfly.md`).
- Feature contract: explicit-id layer slicing — one capture superset
  (`[1, 9, 17, 25, 33]`) serves the matched comparison (`[1, 17, 33]`, the
  frozen DFlash routing) and the upstream 5-layer recipe;
  `contract_defaults_from_config` extracts `algorithm_context_layer_ids` and
  capture-coverage validates the configured algorithm's consumed layers.
- Configs: `configs/dflare/` and `configs/dfly/` each ship the smoke, matched
  2K, layer-rich 2K and Qwen3-VL reduced recipes.
- Training diagnostics: modules can attach the optimizer's fp32 masters
  (`set_master_parameters`); DFlare/DFly fusion metrics are read at fp32
  resolution and log `fusion_prob_l1_drift` against the prior (the bf16 live
  parameter otherwise hides sub-ulp AdamW steps).
- Tooling: `scripts/summarize_fusion_master.py` extracts that fp32 master
  trajectory from a saved checkpoint; `scripts/diag_gradient_coverage.py`
  replays a finished run's feature store against its checkpoint to re-audit
  gradient coverage after training (the step-0 audit sees a zero-init
  residual branch cold, by construction).
- Tests: 31 new CPU tests across `tests/test_dflare.py`,
  `tests/test_dfly.py`, `tests/test_feature_contract.py`; docs:
  `docs/upstream/{dflare,dfly}.md`, `docs/algorithms/{dflare,dfly}.md`,
  `ROADMAP.md`, `ALGORITHM_MATRIX.md`, `UPSTREAM_AUDIT.md`.
- Recorded evidence: COCO 64-row smokes on Qwen2.5-VL-3B (both token-exact,
  exports verified; DFlare AL 1.123, DFly AL 1.438); canonical 2K runs +
  ablations on Qwen2.5-VL-3B (DFlare 1.637 — its dedicated context K/V is
  worth +0.058 while the learned fusion is prior-dominated; DFly 1.682 —
  the hidden-states correction carries +0.113) and Qwen3-VL-8B reduced
  validation (DFlare 1.185, DFly 1.226), all token-exact; numbers in
  `BENCHMARKS.md` and `benchmarks/results/zoo_canonical.{json,csv}`.

### DFly upstream training recipe (LK / D-PACE cold start -> e2e TV)

- New objectives module `vlm_dflash/objectives.py`: exact ports of the
  AngelSpec / LightSeek loss math after the pinned commit (D-PACE position
  weights from per-position confidences, LK in `hybrid` and `alpha` forms,
  the end-to-end multi-step TV term over the block prefix product, per-slot
  CE), all chunked in fp32 so full-vocab tensors never materialize at once.
- `algorithm.train_recipe` selects the objective per run (a name or
  `{name, cold_start_fraction}`); `upstream_two_stage` runs stage 1
  (LK + D-PACE, no CE) for the configured fraction of the planned steps and
  stage 2 (D-PACE CE + e2e TV) after it. Default stays the shared decayed
  block CE (`controlled_ce`), byte-identical plan digests for runs that do
  not set the key.
- Teacher signal: recipe stages consume the target's captured final hidden
  states (`capture.store_final_hidden=true` / `stores_final_hidden=true`);
  training refuses a source without it up front, checkpoints record the
  recipe (`train_recipe`, `objective_profile` = recipe name) and exact resume
  refuses a changed recipe/stage (plan digest + stage gate).
- Config: `configs/dfly/qwen2_5_vl_3b_2k_recipe.yaml` (two-stage recipe on the
  upstream 5-layer context; reuses a same-identity capture-v2 store).
- Tests: `tests/test_objectives.py` (36 tests: bounded resolution schema,
  plan-digest sensitivity, formula parity against direct references, chunk
  invariance, stage switching, teacher requirement, resume gating, and an
  end-to-end tiny-target recipe run + refusal paths).
- Canonical 2K result (`R-20261014-11`): AL **1.839** vs the controlled_ce
  record 1.682 (+0.157 / +9.3%) on the same store, seed and 792-step
  protocol - every slot above the shared-CE run, token-exact 128/128, export
  `objective_profile=upstream_two_stage` reloads token-exact via
  `demo --drafter`. Costs and the side-by-side table: `docs/training-costs.md`
  (new); run details: `docs/plan/RUN_LOG.md`.

### Model Zoo v0.3.1 and the demo reload path

- `zoo-v0.3.1` publishes six canonical archives (DSpark, JetSpec, DFlash2-exp,
  DFlare, DFly on Qwen2.5-VL-3B; DFlash 8B on Qwen3-VL-8B) plus `SHA256SUMS`;
  the README zoo table and the per-archive pages under `docs/zoo/` link them
  as reference artifacts for pipeline validation.
- Clean re-download verification: all six archives re-fetched from the public
  release URL are sha256-identical to the upload set and their inner
  `checksums.json` verify; the DFly and DFlare archives reload through
  `demo --drafter` in fresh directories, token-exact 4/4 vs target-only
  greedy (see `RELEASE_VALIDATION.md`).
- Fix: `demo --drafter` now adopts the drafter's recorded capture layer list
  and, for DFlare/DFly, the consumed context subset, so exports trained on a
  5-layer feature store load out of the box (previously refused by the target
  identity gate); `examples/score_pretrained.py` shares the alignment and
  regression tests cover legacy exports.
- `zoo-v0.3.2` adds the DFly upstream-recipe checkpoint
  (`qwen25vl-3b-dfly-recipe`, 401 MB, code commit `6ad8500`,
  `objective_profile=upstream_two_stage`) as the recipe companion to the
  `zoo-v0.3.1` shared-CE DFly archive; clean re-download verified (sha256,
  inner checksums 7/7, `load_export`, fresh-dir `demo --drafter` token-exact
  4/4).

### Online training: live target feature extraction

- `train --mode online` (`--mode` / `--feature-mode` alias; config
  `train.feature_mode`) runs the frozen target in the trainer process and
  extracts the same per-sample records the capture stage writes to disk.
  Regen answers stay the pre-generated supervision — this is **not** on-policy
  rollout generation, not continual learning and not a serving runtime.
  Presets keep `offline` as the default.
- New `vlm_dflash/target/sample_capture.py` is the shared per-sample capture
  for both the staged capture stage and the online source (same record fields
  and dtypes, same frozen-head resolution, same MRoPE / multimodal-token-type
  handling), so the collator, trainer, loss, audits and eval stay one code
  path.
- New `vlm_dflash/target/online.py`: `OnlineFeatureSource` (frozen-target
  guard, in-process head weights, per-phase extraction counters,
  `feature_source` summary with target residency) and `OnlineFeatureDataset`
  (main-process-only extraction; resumed runs fast-forward consumed batches
  with cheap placeholders instead of re-extracting them).
- Gates and provenance: `--features` + online is refused before any stage;
  `num_workers` must be 0 (CLI + dataset guard); resume refuses mode /
  algorithm / feature-source-digest / step-plan / identity mismatch
  (checkpoints without `feature_mode` are read as offline); checkpoints record
  `feature_mode`, `feature_source_digest`, `feature_backend` and both devices;
  `eval` reports `checkpoint_feature_modes`; `export` records `feature_mode`.
- Docs: `docs/online-training.md` (offline/online trade-off, shared code,
  commands, cost accounting, boundaries, validation status); README and
  README.zh-CN point at it.
- Fix caught by the smoke matrix: `--set` overrides are now merged before
  `validate_config`, so a required algorithm knob supplied on the CLI (e.g.
  JetSpec's `drafter.loss_decay_gamma=null`) validates the config that
  actually runs; regression test in `tests/test_cli.py`.
- Tests: 22 new CPU tests (`tests/test_online_mode.py`) plus the CLI
  ordering regression; full suite green (198 tests). GPU smokes on one A6000:
  Qwen2.5-VL-3B (4-step demo slice, token-exact eval, peak 13.65 GB); all
  five algorithms online on 3B (train → token-exact eval → export; peaks
  13.65-15.53 GB; DSpark calibration on the live source; JetSpec/DFlash2 with
  `capture.store_final_hidden=true`); Qwen3-VL-8B untied head (peak 33.92 GB,
  target resident 17.5 GB, token-exact); online resume verified (3
  fast-forward batches, no re-extraction). Smoke ALs are <=1.09 after 4 steps
  and are not quality results.

### Online review fixes: exact-resume safety, content digests, dtype boundary

Findings from the external review of `aed7bf8` (four confirmed issues), fixed
before any further online work. No algorithm or protocol changes.

- **Resume preflight is read-only.** `train --resume` now validates the
  checkpoint (and, for online, the consumed data) *before* writing anything:
  `config.json` is no longer overwritten before validation, `ckpt_init.pt` is
  never regenerated (a legal resume keeps the original random-init baseline
  byte-for-byte), and a refused resume — changed mode/algorithm/plan/source/
  identity, or rebuilt upstream stages — leaves config, checkpoints, logs,
  provenance and diagnostics byte-identical. The CLI preflight
  (`cli._preflight_resume` + `dflash.train.preflight_resume`) also refuses
  `--resume --force` and legacy runs without stage manifests, and does not
  need to load the target (the online digest comes from the rows + image
  bytes directly).
- **Versioned content digests.** `ONLINE_DIGEST_VERSION`
  (`vlm-dflash-online-source-digest-v2`) hashes the actual consumed content —
  per-row prompt/answer token ids in order, the exact `--limit` slice, image
  bytes (verified against `data/manifest.json`; a stale manifest is a loud
  error, not evidence) and the extraction settings — so a same-length answer
  edit or a same-size image swap changes the digest, while the same content
  copied to another directory keeps its identity. Offline stores get the same
  treatment (`OFFLINE_DIGEST_VERSION`,
  `dflash.features.offline_feature_digest` over manifest + index + shard +
  head bytes).
- **Training-plan resume gate.** New `vlm_dflash/training_plan.py` records a
  versioned digest of the effective plan — seed, micro-batch, accumulation,
  epochs, dataset length, draft dtype, optimizer/schedule settings, drafter
  architecture knobs and the algorithm-namespaced knobs with resolved
  defaults — in every checkpoint. Exact resume refuses any change (including
  shape-compatible ones like seeded shuffles and loss weights) and refuses
  checkpoints that predate the gate instead of fabricating the missing
  fields; the migration policy is documented in `docs/online-training.md`.
- **Dtype boundary.** The online source now hands the trainer embed/head
  copies in the feature-storage dtype (mirroring `capture.save_target_head`)
  instead of the target's native dtype, and the training loop refuses any
  frozen-head/compute-dtype mismatch up front with an actionable error
  (previously an opaque `expected m1 and m2 to have the same dtype` deep in
  the first forward). `train_summary.json` records the effective
  target/storage/drafter/head dtypes (`dtypes` block). Supported combinations
  are exactly "stored head dtype == `train.draft_dtype`"; `capture.dtype`
  values the trainer cannot compute in (`float16`) are refused before
  training.
- **Cleanup and observability.** The online source is closed on every exit
  path (success and exceptions); per-phase target-device CUDA peaks
  (`target_device_peak_gb`) are recorded in the source summary with an
  explicit note on shared vs split devices; the CLI rejects
  `feature_mode=online` + `num_workers != 0` before any stage runs; a resumed
  online run reuses the recorded visual-path check instead of rerunning (and
  rewriting) it.
- Tests: new CPU coverage for digest v2 (token-value edits, image swaps,
  path-stability, stale manifests), the plan gate (field coverage across all
  five algorithms, legacy refusal), resume write-order regressions (online and
  offline: refused resume byte-identical, legal resume preserving
  `ckpt_init`), the real-`build_trainer_module` fp32-target/bf16-storage
  regression and the mismatched-head refusal, and an exact
  offline↔online multi-step update-alignment test (fixed teacher/data/init/
  seed/budget).
- Post-fix GPU rerun set: all five algorithms, the Qwen3-VL-8B untied head
  and the split-device path (target `cuda:0` → drafter `cuda:1`) re-ran on
  the fixed code — train → token-exact eval → export in every run,
  all-bf16 dtype records, plan digests in every checkpoint, source digests
  stable across devices and sensitive to extraction settings
  (`6e2051af43b5` without final hidden, `6e58321b4579` with it; 8B
  `9d2712d52448`), split-device target peak 8.02 GB isolated, 8B peak
  33.92 GB. The superseded pre-fix smoke run dirs were pruned for disk
  space (numbers quoted in this changelog).

### Second review fixes: accumulation stall guard, log reconciliation, startup cleanup, external-store resume

Findings from the external second-pass review of `4e9a538` (four P2 boundary
issues), fixed without touching the training recipe: loss normalization, the
step budget and the training trajectory are unchanged, so no plan-version
bump was needed.

- **Accumulation is progress, not a stall.** The shared loop's stall guard
  counted optimizer steps, so a legal configuration whose accumulation
  window spans epochs (e.g. 1 sample, `grad_accumulation_steps=4`) raised
  "no progress" on the third epoch although every micro-batch had produced
  gradients. The guard now counts valid micro-batches (forward + backward)
  and only stalls after three full epochs with zero valid supervision; the
  error message says exactly that. The budget/loss semantics
  (`ceil(len/accumulation)` planned per epoch, accumulation carried across
  epoch boundaries, every optimizer step averaging exactly `accumulation`
  micro-batches) are documented on `compute_total_steps` and unchanged.
- **Resume reconciles the train log with the checkpoint.** Rows past the
  checkpoint's committed step — including a half-written JSON line — are
  preserved as `train_log.uncommitted.jsonl` (a recovery artifact) before
  the replay appends, so resuming a `[1,2,3]` log with a step-2 checkpoint
  ends with unique ordered steps instead of `[1,2,3,3]`. A run that
  completes now also writes its final resumable state to `ckpt_last.pt`
  (`step == total_steps`), so a later exact resume of a finished run
  replays zero steps instead of re-emitting the logged tail.
- **Startup failures now run the cleanup path.** The online resource guard
  previously started after the source was constructed; an exception during
  source init, manifest access or the startup visual check left the target
  referenced. The `try/finally` now wraps source construction and all
  startup stages; `OnlineFeatureSource.close()` is idempotent, tolerates
  partially-constructed instances (the constructor closes on failure) and
  drops the model/processor/embed/head references (`empty_cache` is
  deliberately not relied on).
- **`--features` resumes follow the real dependency chain.**
  `cli._preflight_resume` now mirrors `cmd_train`: external-store resumes
  check the data stage plus store digest/plan/identity and no longer demand
  `regen`/`capture` outputs that an `--features` run never produced (offline
  regression from the first preflight); online and internal-offline resumes
  still require regen (and capture for offline).
- Tests: 8 new CPU regressions — pending accumulation across epochs is not a
  stall while zero supervision still stops; cross-epoch interrupted resume
  float-aligns with the uninterrupted run; log reconciliation with
  uncommitted tails and half lines; interrupted-run resume keeps log steps
  unique and a completed-run resume is a no-op; visual-check failure and
  constructor failure both close the online source exactly once; the CLI
  preflight dependency chain for `--features`. Full CPU suite green (222
  passed, was 214).
- GPU smoke (second-review set, Qwen2.5-VL-3B online,
  `runs/fix3_resume_online`): SIGKILLed after step 3 was logged (log
  `[1,2,3]`, checkpoint at step 2). Resume reconciled the log (1
  uncommitted row preserved as `train_log.uncommitted.jsonl`), replayed
  steps 3-6 with unique ordered entries (replayed step-3 loss identical to
  the pre-kill value), and the completed run wrote its final resumable
  state (`ckpt_last.pt` step 6); a second resume replayed zero steps with
  the log byte-identical, and eval was `token_exact=True` on both
  `ckpt_final.pt` and `ckpt_init.pt` (feature mode recorded as online).

### README repositioning (docs only)

- `README.md` / `README.zh-CN.md` now present the project as training
  infrastructure: the first screen keeps the support matrix (targets, five
  algorithms, offline/online, resume, eval, export) and the correctness
  evidence (token-exact gates, frozen-target identity, export reload); the
  small-scale AL numbers, the per-algorithm ranking table and the runtime
  speedup figures are no longer displayed there. All historical results stay
  unmodified in `BENCHMARKS.md` / `RELEASE_VALIDATION.md` /
  `docs/measured-runs.md`, which the README links to; Model Zoo entries are
  labelled reference/validation artifacts. No claim is made about the causes
  of smoke-scale acceptance numbers.

## v0.3.1 (2026-10-13)

Distribution fixes, first-run experience, and the post-publish growth review
items. No algorithm, protocol or benchmark changes — every recorded number in
`BENCHMARKS.md` / `RELEASE_VALIDATION.md` is untouched.

### Distribution and identity fixes

- The wheel now ships its resources: `vlm_dflash/presets/*.yaml` and
  `vlm_dflash/resources/constraints-tested.txt` are packaged (setuptools
  package-data) and loaded via `importlib.resources`, so `vlm-dflash train
  --preset smoke` works from a fresh install outside the repo (previously
  `FileNotFoundError` from site-packages). CI asserts the wheel contents and
  loads all three presets from a repo-less CWD.
- Single version source: `vlm_dflash.__version__` drives the distribution
  metadata (pyproject `dynamic = ["version"]`) and a new
  `vlm-dflash --version`; a test pins metadata == module version.
- The teacher/target identity gate no longer requires `target_model_path`:
  the same checkpoint in a new directory passes (content fingerprints,
  processor fingerprint and resolved contract fields still gate), while real
  model/revision/processor changes keep failing loudly. The path stays in the
  records as audit information; stage input digests ignore it for the same
  reason.

### Pipeline reuse safety

- `train` stage caching is now input-addressed: each of data/regen/capture
  records a digest of its inputs (config subset, target identity, upstream
  stage digest, dataset file fingerprints) in `<stage>/stage_manifest.json`.
  Existing outputs with changed inputs are refused before any expensive
  compute (message suggests `--force` or a new run directory); outputs without
  a recorded digest are refused instead of silently reused.
- `train --features <dir>` trains from an existing captured feature store
  (explicit reusable interface; `train_dflash(..., features_dir=...)`), so one
  capture can drive several algorithms. The store's identity, manifest and
  saved target head are re-verified and the reuse (directory, samples, seq
  length, dtype, manifest digest) is recorded in `provenance.json`.

### Docs, metadata and first-run

- First public release of the Model Zoo:
  [zoo-v0.3.0](https://github.com/LUOseu/VelaDraft/releases/tag/zoo-v0.3.0)
  hosts the DFlash and Domino exports (tar.gz + sha256 checksums + model
  cards); both were re-downloaded and re-verified token-exact.
- `demo --drafter <export>` now follows the checkpoint's recorded algorithm
  (scoring a Domino export no longer requires `--algorithm domino`).
- README restructured around the user path (pretrained demo first, pipeline
  SVG, Model Zoo links, split acceptance/inference-cost benchmark tables,
  frozen JSON/CSV results in `benchmarks/results/`), plus a Chinese README
  (`README.zh-CN.md`) and refreshed `examples/` scripts.
- `SUPPORT_MATRIX.md` is now generated from the algorithm registry
  (`scripts/render_support_matrix.py`, drift-tested in CI) and the algorithm
  metadata records the Qwen3-VL reduced validation
  (`reduced_validation` state) consistently across doctor/export/docs.
- Added `docs/serving-bridge.md` (proposal: what a serving bridge must solve;
  no support claim) and `benchmarks/results/zoo_canonical.{json,csv}` +
  `scripts/extract_zoo_results.py`.

## v0.3.0 (2026-10-12)

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
