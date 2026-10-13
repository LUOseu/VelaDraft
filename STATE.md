# STATE

Updated: 2026-10-13 (v0.3.2 in development: online target feature source
pushed as `aed7bf8`; online review fixes pushed as `1786176`; second-pass
review fixes committed locally (not yet pushed); GitHub repo renamed to
`LUOseu/VelaDraft`).

**Status: v0.3.1 tagged and pushed** (tag `v0.3.1` at `eb3f314`); **v0.3.2 in
development — the online target feature source is pushed as `aed7bf8` and
the review fixes as `1786176`** (repo now `github.com/LUOseu/VelaDraft`; old
URLs redirect; distribution and CLI stay `vlm-dflash`).

- v0.3.2 scope (so far): `train.feature_mode: offline|online` — live
  teacher-forced extraction from the frozen target in the trainer process
  (same trainer/collator/provenance; not on-policy, not serving). See
  `CHANGELOG.md` v0.3.2 and `docs/online-training.md`.
- Still frozen: new algorithms, targets, video / multi-image, serving runtime
  and research methods.
- Release gates: `RELEASE_CHECKLIST.md`; evidence: `RELEASE_VALIDATION.md`.

## Current stable features
- Staged pipeline: `prepare -> regen -> capture -> train -> eval` (+ `export`);
  training reads staged features only (target off GPU).
- Runtime model contract + provenance gates (target/processor/identity).
- Qwen2.5-VL-3B and Qwen3-VL-8B adapters; single image; tied and untied heads.
- Verification: `sequential` (bitwise-exact gate), `parallel` /
  `strict_parallel` (experimental throughput; not lossless).
- Unified CLI (`doctor/train/eval/export`), presets, resume, HF-style export.
- Algorithm registry with five drafters behind one interface: DFlash (stable
  frozen reference), DSpark / Domino / JetSpec (validated on the 2K
  apples-to-apples protocol), DFlash2 (experimental objective; upstream does
  not publish it). See `ALGORITHM_MATRIX.md`, `SUPPORT_MATRIX.md`, `BENCHMARKS.md`.
- Gradient coverage audit + per-algorithm calibration/ablation artifacts;
  namespaced algorithm config; export loader dispatch by algorithm.

## Experimental
- Benchmark harness skeleton (`scripts/benchmark.py` + manifests); only the
  scaffold workload (`caption_general`, run's own val split) is runnable;
  TextVQA lands with this release; DocVQA/ChartQA pin protocol without an
  adapter yet.
- Parallel runtime speedup numbers (see BENCHMARKS.md) — experimental.
- Online target feature source (`train --mode online`, v0.3.2-dev): validated
  by CPU tests + 3B/8B GPU plumbing smokes (token-exact, provenance, resume);
  no quality/canonical runs yet, offline remains the default and the only
  path behind the recorded benchmarks (`docs/online-training.md`).

## Online feature source (2026-10-13, v0.3.2-dev)

- `train.feature_mode=offline|online` (CLI `--mode` / `--feature-mode`);
  presets pin `offline`. Online runs the frozen target in the trainer process
  and extracts the capture-stage record contract live (shared
  `target/sample_capture.py`); regen answers stay pre-generated (not
  on-policy, not serving).
- Gates: target must be eval/no-`requires_grad`; `num_workers=0` (CLI +
  dataset guard); `--features` + online refused; resume refuses mode /
  algorithm / digest / step-plan / identity mismatch (pre-feature ckpts read
  as offline); leftover `features/` store ignored with a note.
- Evidence: 22 new CPU tests + a CLI `--set`-before-validation regression +
  full suite green (198 tests). GPU: 3B smoke (`runs/online_smoke_qwen25_3b`,
  3-row demo slice, 4 steps) train OK, live visual check passed, coverage
  36/36, eval token-exact, ckpt + export `feature_mode=online`, peak
  13.65 GB; all five algorithms smoked online on 3B
  (`runs/online_smoke_{dspark,domino,jetspec,dflash2}`, train → token-exact
  eval → export, peaks 13.65-15.53 GB; DSpark calibration ran via
  `online_target_head_batches`; JetSpec/DFlash2 with
  `capture.store_final_hidden=true`); 8B untied smoke
  (`runs/online_smoke_qwen3_8b`) peak 33.92 GB, target resident 16.33 GiB,
  token-exact; resume check: 3 fast-forward batches, 1 extraction forward
  instead of 4 (`resume_fast_forward_reduced_cost=true`). Smoke ALs <=1.09
  after 4 steps — plumbing only, not quality. The post-fix rerun set (all
  five algorithms, the 8B untied head and split devices; smoke 6 in
  `docs/online-training.md`) supersedes these pre-fix smokes; their run dirs
  were pruned for disk space afterwards (numbers quoted in the docs).

### Online review fixes (on top of `aed7bf8`, 2026-10-13)

- Fixed the four confirmed findings from the external online review before any
  further online work (no algorithm/protocol changes):
  1. **Resume preflight is read-only**: `train --resume` validates plan /
     source / mode / algorithm / identity before writing anything — a refused
     resume leaves `config.json`, `ckpt_init.pt`, logs, `provenance.json` and
     the visual diagnostics byte-identical; a legal resume keeps the original
     `ckpt_init.pt` and never regenerates the baseline; `--resume --force`
     and stage rebuilds are refused. The CLI preflight needs no target load
     (digest computed from rows + image bytes).
  2. **Versioned v2 content digest** (`ONLINE_DIGEST_VERSION`): consumed
     prompt/answer token ids in order, the exact `--limit` slice, image bytes
     (verified against `data/manifest.json`; stale manifests are a loud
     error), extraction settings; path-stable across directory moves.
     Offline stores use the equivalent `offline_feature_digest`.
  3. **Versioned training-plan gate** (`vlm_dflash/training_plan.py`) in every
     checkpoint: seed / micro-batch / accumulation / epochs / dataset length /
     draft dtype / optimizer + schedule / drafter + algorithm knobs (resolved
     defaults). Legacy checkpoints are refused for exact resume instead of
     being re-stamped with new digests.
  4. **Dtype boundary**: online head copies are cast to the storage dtype
     (mirroring `capture.save_target_head`); the training loop refuses
     frozen-head/compute-dtype mismatches with an actionable error;
     `train_summary.json.dtypes` records the effective combination;
     per-phase `target_device_peak_gb` and `digest_version` are in the source
     summary; the online source is closed on all exit paths; the CLI rejects
     online + `num_workers != 0` before any stage; a resumed online run
     reuses the recorded visual-path check.
- Evidence: full CPU suite green (214 tests, was 198) including new
  digest/plan/write-order regressions (online and offline failed-resume byte
  hashes; legal resume keeping `ckpt_init`) and an exact offline↔online
  multi-step update-alignment test (fixed teacher/data/init/seed/budget, CPU
  exact float equality). GPU smoke 4 (3B, post-fix, `runs/fix_smoke_online`):
  fresh 3-step online run OK (`dtypes` all bf16, digest v2, plan digest in
  every checkpoint, target-device peak 13.65 GB); refused resume (changed
  seed) exited before any write with `config.json` / `ckpt_init` / `ckpt_last`
  / logs / summary / provenance / visual check md5-identical; legal resume
  passed the preflight, reused the recorded visual check, resumed from step 2
  (2 fast-forward batches → 1 extraction forward, `reduced_cost=true`), kept
  `ckpt_init` unchanged, then eval was token-exact on both checkpoints and
  export recorded `feature_mode=online`. Post-fix rerun set (GPU smoke 6,
  `runs/fix2_smoke_*`): all five algorithms train → token-exact eval →
  export with all-bf16 dtype records and plan digests in every checkpoint
  (source digests `6e2051af43b5` without / `6e58321b4579` with
  `store_final_hidden`, identical across devices); 8B untied head OK (digest
  `9d2712d52448`, peak 33.92 GB); split-device run (target `cuda:0` →
  drafter `cuda:1`) token-exact with the isolated target peak 8.02 GB. GPU
  smoke 5
  (`runs/fix_smoke_align_{offline,online}`) — SAME teacher / data / init /
  seed / budget offline vs online: identical per-step losses
  `[13.25, 16.875, 13.0]` (max |Δloss| = 0.0) and bitwise-identical
  `ckpt_init.pt` / `ckpt_final.pt`; the feature digests differ by mode
  (offline `437166daa78a`, online `6e2051af43b5`) while the training-plan
  digests are equal (`43140c4fdcc5`) — the online path reproduces the staged
  path exactly at fixed teacher/data/init/seed/budget.

### Second review fixes (on top of `1786176`, 2026-10-13, not yet pushed)

- Fixed the four P2 boundary findings from the external second-pass review of
  main `4e9a538`. No plan-version bump: loss normalization, the step budget
  and the training trajectory are unchanged (the loop only changed *when it
  gives up*, what it logs and what it cleans up).
  1. **Accumulation stall guard** (`dflash/train.py::run_training_loop`):
     stall counting now tracks valid micro-batches (successful forward +
     backward) instead of optimizer steps, so a small dataset with a large
     accumulation window (reproduced: 1 sample, accum 4) no longer
     false-stalls on the third epoch; three full epochs with zero valid
     supervision still stop with an explicit message. Budget semantics
     (`ceil(len/accumulation)` planned per epoch, cross-epoch accumulation)
     documented on `compute_total_steps`.
  2. **Resume log reconciliation** (`_reconcile_train_log`, `train_dflash`):
     rows past the checkpoint's committed step (incl. a half-written JSON
     line) are preserved as `train_log.uncommitted.jsonl` before the replay
     appends — resuming `[1,2,3]` at step 2 ends `[1,2,3,4,...]`, no
     duplicates; a completed run now persists its final resumable state
     (`ckpt_last.pt`, `step == total_steps`) so a later exact resume of a
     finished run replays zero steps.
  3. **Startup cleanup**: the online `try/finally` now wraps source
     construction and all startup stages (init, manifest access, visual
     check); `OnlineFeatureSource.close()` is idempotent, handles
     partially-constructed instances (constructor failure → close → re-raise)
     and drops model/processor/embed/head references.
  4. **External-store resume**: `cli._preflight_resume` mirrors `cmd_train`
     for the three feature sources — `--features` resumes require only the
     data stage (plus store digest / plan / identity); online and
     internal-offline runs still require regen (+ capture for offline).
     Fixes the offline regression the first preflight introduced.
- Evidence: 8 new CPU regressions; full CPU suite 222 passed (was 214).
  GPU smoke 7 (`runs/fix3_resume_online`, 3B online, 6 steps, SIGKILL after
  step 3): interrupted state log `[1,2,3]` / ckpt step 2 → resume moved the
  uncommitted row to `train_log.uncommitted.jsonl`, replayed steps 3-6
  (final log `[1..6]` unique, replayed step-3 loss = pre-kill 13.0),
  completion wrote the final state (ckpt step 6); a second resume replayed
  zero steps, log md5 unchanged; eval `token_exact=True` for
  `ckpt_final` / `ckpt_init`, feature modes online; plan digest
  `43140c4fdcc5`.

## Latest measured results
- 3B / 512-corpus rerun: AL 1.198, 64/64 exact (earlier v0.1 run: 1.220).
- 3B / 2K-corpus: AL 1.630, 128/128 exact, slots `[.413,.149,.048,.015,.004,.001,0]`.
- 3B / 5K-corpus: AL 1.817, 128/128 exact, slots
  `[.482,.210,.080,.030,.011,.005,.001]` (1974 steps, CE 6.84 -> 2.58, peak
  30.0 GB).
- 8B / 512-corpus: AL 1.322.
- 8B / 2K-corpus: AL 1.665, 32/32 exact, slots
  `[.413,.160,.063,.027,.010,.002,.0004]` (1996 kept rows, 1500 steps, final
  loss 2.69, visual-path check passed; speedup 0.81x).
- Algorithm zoo / 3B / 2K / 128 prompts (sequential, token-exact): DFlash
  1.630, DSpark 1.735, Domino 1.764, JetSpec 1.599, DFlash2 1.954 (experimental
  objective; conv/selector ablation 1.798).
- DFlash frozen-baseline regression after the zoo landed: full 128-prompt
  re-eval reproduces 1.6298 bit-identically (`tests/reference_dflash/manifest.json`).
- Runtime probe (32 prompts, bf16): see BENCHMARKS.md.

## DSpark stage (2026-10-12)
- `fix(dspark)` head ownership + coverage-audit timing/units + `spec_generate`
  block window + calibration ragged anchors (bc9c0e6).
- `fix(eval)` verify-backend parity now forwarded through every algorithm
  (db7cb8d): DSpark verify runs on the same SDPA backend as `generate()`;
  `eval_result.json` records `verify_backend_switch_enabled`. COCO smoke
  re-run: trained + init both token-exact 8/8 (AL 1.123 vs 1.000).
- DFlash regression check after the eval fix (`dflash_regress8`, 8 prompts):
  AL 1.766, token-exact.
- DSpark 2K canonical: features captured (1050 samples, 7.2 GB, replay
  bitwise, visual-span + downstream change passed). Training 12:59-13:14
  (792 steps, final loss 3.53, overfit sanity drop 0.9997 / slot0 1.0,
  gradient coverage 40/40 with markov/confidence heads covered, drafter
  visual check logits_diff=7.56 passed; confidence calibration BCE 0.590 /
  ECE 0.048 / AUROC 0.745, pred_std 0.213 -> not constant). 128-prompt eval
  running.

## Domino stage (2026-10-12)
- Trainer/factory/plugin landed (`refactor` + `feat(domino)`): dual-logit
  objective `(1-λ)·CE(final) + λ·CE(base)` with the upstream linear λ decay,
  shifted-label slot convention, suffix-only correction, ablation-only
  `enable_correction_head` switch; 11 new unit tests (98 total green).
- COCO smoke (Qwen2.5-VL-3B, 64-row slice, shared capture): overfit sanity
  drop 1.000 / slot0 1.0; gradient coverage 40/40 at 10% of the run with
  prefix_gru + embed_proj covered; drafter visual check passed; eval
  token-exact, AL 1.123 vs init 1.000.
- Domino 2K canonical: trained 13:25-13:42 (792 steps, final loss 2.20,
  visual check logits_diff=7.46 passed); eval AL 1.764 (token-exact 128/128),
  slots `[.443,.180,.078,.036,.016,.009,.005]` - ahead of DSpark (1.735) and
  DFlash (1.630) on the same corpus/protocol.
- Ablations (same corpus/protocol): DSpark `sample_from_anchor: false` AL
  1.719 (Markov head alone +0.089 over DFlash; anchor slot +0.016 on top);
  Domino `enable_correction_head: false` AL 1.662 (correction head +0.102).
  Zoo scoreboard table in BENCHMARKS.md.
- Exports verified for both algorithms (`--out-dir /tmp/export_*`):
  `method: dspark` / `method: domino` with the head fields in `draft_config`.

## JetSpec stage (2026-10-12, stage 4 of the algorithm zoo)
- Capture v2 (`capture.store_final_hidden`): stores the target's final hidden
  state per sample (~+25% storage); feature collate carries it as
  `final_hidden`. Teacher-signal identity verified on a real sample:
  `lm_head(hidden_states[-1])` == the target's own logits, max|diff| 0.0.
- Trainer (`vlm_dflash/jetspec/`): `causal_in_block=True` mask + chunked
  temperature-scaled forward KL (ported semantics; chunked trainer path equals
  the logits-in reference within 1e-5) + teacher gather at
  `label_indices[..., :-1]`; metrics are draft-vs-teacher agreement and
  acceptance length. 19 new tests (134 total green).
- First 2K attempt OOM'd (`CUBLAS_STATUS_ALLOC_FAILED`) in the overfit sanity
  phase: the KD backward retains fp32 log-probabilities of all valid positions
  (16 samples x 256 anchors -> ~3x17 GB fp32 at [P, vocab]). Fixed by
  overfit_sanity.num_samples=8 + kd_chunk_size=1024; peak memory 29.7 GB.
- 2K canonical: 792 steps, KD loss 6.06 -> 3.10 (first/last-10 mean), overfit
  pass (drop 0.992 / slot0 0.98), coverage 36/36, visual check 6.92 logits diff;
  peak 29.7 GB, 20.6 min.
- Serving fidelity (new finding): JetSpec's trained checkpoint relies on its
  causal in-block factorization. Unmasked shared-runtime serving flips 7.6% of
  per-slot greedy argmaxes (max |dlogit| 9.0) and loses 0.26 AL. Added
  `inference_in_block_causal` (draft-config flag; loader forces it for older
  checkpoints; export records it + a runtime note). Eval: AL 1.599 token-exact
  128/128 under causal-mask serving vs 1.338 unmasked; per-slot [.408, .137,
  .040, .011, .003, .001, 0].
- Interpretation: at this scale/corpus JetSpec (1.599) does not beat the frozen
  DFlash CE baseline (1.630); uniform soft-distribution weighting costs
  early-slot accuracy vs decayed hard-label CE. The paper's data-scaling claim
  is not testable at 2K. Exports verified (`/tmp/export_jetspec_2k`,
  `/tmp/export_jetspec_smoke`).

## DFlash2 stage (2026-10-12, stage 5 - zoo complete, Experimental)

- Upstream audit completed (`docs/upstream/dflash2.md`): sources pinned
  (z-lab/dflash `dflash/model.py` @ 07ebd93d, MIT; speculators @ 55ebbd5b,
  Apache-2.0; blog https://inco.ai/blog/dflash2/). The old NeMo pin was wrong -
  `NVIDIA-NeMo/Automodel @ bb5b34ce` contains no dflash2 (re-checked). Honest
  split recorded: architecture official, serving walk inferred, **training
  objective undisclosed upstream** (we implement speculators' experimental
  split objective) -> the algorithm ships as Experimental.
- Port (`vlm_dflash/dflash2/`): GroupedDynamicCausalConv (block-causal grouped
  taps, identity init) + CandidateSelector (predecessor/successor codebooks +
  hidden projection) + conv-wrapped decoder layer + `Qwen3DFlash2DraftModel`
  with the documented serving walk (one parallel draft forward, then a
  sequential top-K candidate walk chained on the selected predecessor);
  trainer = unary forward KL (chunked, exact staged teacher) + K-way selector
  CE with rank-K target injection, fixed-exp decay (gamma 4.0), masked-count
  denominator; serving metrics gated on the un-injected candidate set.
  24 new tests (140 total green).
- COCO smoke (64 rows): prepare/regen/capture (visual check passed) ->
  overfit sanity drop 0.998 / slot0 1.0 -> 48-step train (visual check
  logits_diff 0.31 passed) -> eval token-exact 8/8, AL 1.362 vs init 1.000.
- 2K canonical (`runs/qwen25vl3b_dflash2_2k`, shared corpus/features/protocol):
  792 steps, unary KL 3.41 -> 1.20, selector CE 2.20 -> 0.28, candidate recall
  0.78, overfit 0.995/1.0, coverage 51/51, visual check 7.92; peak 31.1 GB,
  22.6 min. Eval AL **1.954** token-exact 128/128, slots
  `[.495,.243,.118,.060,.027,.013,.006]` - best at this scale (DFlash 1.630,
  DSpark 1.735, Domino 1.764, JetSpec 1.599).
- Decomposition (completed): serving-time selector ablation (16 prompts, same
  checkpoint) AL 2.060 (selector) vs 1.965 (unary-only ranking of the same
  candidates); ~60% of proposal positions are re-picked, gains concentrated in
  deep slots. Local-mixing + selector ablation (`conv_kernel_size: 1`,
  `selector_loss_alpha: 0.0`, run `qwen25vl3b_dflash2_2k_ablate_nomix`, same
  recipe/features/seed): 792 steps, final loss 0.976, overfit + visual checks
  passed. Eval AL **1.798** token-exact 128/128 - the joint conv+selector share
  is +0.156; the ablated run still beats DSpark/Domino, and its
  `selector_override_rate` is 0.0 (ranking knob confirmed off). The objective
  axis (CE -> forward-KL) is *not* isolated; a `unary_loss: ce` control would
  be the next decomposition step if needed.
- Export verified (`/tmp/export_dflash2_2k`): `method: dflash2`,
  `method_details` (conv/selector + experimental-objective tag), runtime note;
  `load_export` rebuilds the selector model and forwards (328.5M params).
- Doctor reports the full-vocab requirement for DFlash2; eval summaries record
  `draft_candidate_selector`.

## Registry refactor verification (2026-10-11, no DFlash regression)
- 61/61 CPU tests green after the refactor.
- `vlm-dflash doctor --algorithm dflash` reports the registry + metadata.
- Real eval through the registry path (`--tag registry_check`, 8 prompts,
  sequential): AL 1.610, token-exact, algorithm tag recorded in the result -
  consistent with the 128-prompt 1.630 protocol run.
- First benchmark-harness run (`caption_general`, 8 prompts): AL 1.610, exact,
  report written to `benchmarks/reports/` (smoke only - 8 prompts).
- Packaging: `vlm_dflash-0.2.0` wheel + sdist build; wheel contains the
  algorithms/benchmark subpackages and the `vlm-dflash` console script.

## DFlash stage-5 regression (2026-10-12, after DFlash2 code landed)
- 8 prompts, registry paths, same frozen 2K checkpoint
  (`eval_result_post_dflash2_stage_regression.json`): AL 1.6096 - bit-identical
  to the pre-DFlash2 registry check (1.6096) and token-exact; `ckpt_init`
  1.002. The `_make_decoder_layer` seam did not move DFlash.
- Full canonical re-eval, 128 prompts / 128 tokens
  (`eval_result_post_dflash2_full128.json`): AL **1.6298**, bit-identical to
  the frozen pre-zoo baseline (1.6298365840801265); both checkpoints
  token-exact (128/128). Machine-readable record:
  `tests/reference_dflash/manifest.json`.

## v0.3.0-rc1 release engineering (2026-10-12/13)

Scope frozen (harden/document/validate/package only). Evidence:
`RELEASE_VALIDATION.md`; gates: `RELEASE_CHECKLIST.md`.

- Five-minute demo landed: `vlm-dflash demo` (synthetic sample, token-exact
  eval, random-init baseline or `--drafter <run|ckpt|export>`) and
  `vlm-dflash train-demo` (full staged loop + tiny training + eval + export).
  Measured on the A6000: demo 45 s / 8.2 GB; demo --drafter zoo 42 s / 8.5 GB
  (AL 1.313 vs 1.000); train-demo 3 min 13 s / 8.2 GB (loss 13.0 -> 0.013,
  eval AL 1.138 vs 1.000, 4/4 token-exact). `--drafter` demos align the
  sample's image pixel bounds with the drafter's recorded identity and use a
  drafter-specific run dir (provenance gate unchanged).
- TextVQA public workload validated: adapter + pinned 128-prompt seed-42
  subset (`benchmarks/prompts/textvqa_v1.jsonl`, shard/row hashes in
  `benchmarks/manifests/02_textvqa.json`); DFlash 1.263 / DSpark 1.253 /
  Domino 1.268 AL, token-exact 128/128 (32 new tokens, sequential);
  reports under `benchmarks/reports/`.
- Frozen baseline re-eval on the release commit: **1.6298365840801265**,
  bit-identical to the frozen reference (128/128 token-exact); 16-prompt
  release regressions token-exact — DSpark 1.827, Domino 1.854, JetSpec
  1.661, DFlash2 2.047 (same checkpoints).
- Qwen3-VL reduced validation (256-row slice, 192 steps, 16-prompt eval,
  export + reload, all token-exact): DSpark 1.2212, Domino 1.2278, JetSpec
  1.1185, DFlash2 1.2104 (init 1.000) — SUPPORT_MATRIX Qwen3-VL column
  upgraded for all four (dflash2 stays experimental by objective).
- Model zoo: 6 canonical exports with reload self-check
  (243.8M / 321.6M / 292.9M / 243.8M / 328.5M / 629.2M params).
- Packaging/CI/hygiene: wheel + sdist build, fresh-venv CLI smoke; CI
  workflow (CPU tests + package build); issue templates;
  CONTRIBUTING / SECURITY / CODE_OF_CONDUCT / CITATION.cff; absolute host
  paths removed from configs/presets/tests/docs; secret / large-file /
  license / claim audits recorded in `RELEASE_VALIDATION.md`.
- Tests: 155/155 CPU tests pass; explicit `pyarrow` dependency; manifest
  prompt paths now expand `${VAR:-default}` (was `expandvars`-broken).


## Known bugs / caveats
- bf16 capture/training: protocol-reproducible, not bit-identical.
- `parallel` verification exactness is numerically unreachable with the
  current block-forward path (see docs/REFERENCE.md); do not advertise as
  lossless.

## Blocked / waiting
- PyPI publishing and Hugging Face Hub hosting: not done; the code is public
  (`github.com/LUOseu/VelaDraft`) and the Model Zoo weights are hosted as the
  `zoo-v0.3.0` GitHub release. Do not write PyPI/HF instructions that do not
  exist yet.
- Server disk policy in effect: feature stores are pruned once a run is
  recorded ("deterministic to recapture"); the zoo 2K stores live in
  `runs/qwen25vl3b_jetspec_2k/features` (capture-v2 superset), symlinked by
  the dspark/domino/dflash2 2K runs. `/home` is at ~99% (278G free as of
  2026-10-12; the cleanup reclaimed ~23G: tmp debug scripts, /tmp exports,
  a stale code copy, smoke resume states, duplicate feature store).

## Next highest-value tasks
1. DFlash2 objective isolation (only if the 1.954 is quoted as a method
   claim): a `unary_loss: ce` control separates the forward-KL objective axis
   from conv + selector; features and recipe are ready
   (`runs/qwen25vl3b_jetspec_2k/features`, ~25 min GPU per run + eval).
2. Serving-fidelity follow-up from the JetSpec stage: decide whether the
   `inference_in_block_causal` mask path should be exercised by a
   `strict`-style test on a GPU CI run (currently unit-tested + smoke-tested).
   First serving-bridge step is scoped in `docs/serving-bridge.md` (proposal).
3. Deterministic benchmark prompt file for `caption_general`; land one public
   dataset adapter (TextVQA first) to move a workload from planned -> validated.
4. Publish the remaining zoo exports (DSpark / JetSpec / DFlash2-exp / 8B
   DFlash) as release assets once hosting is decided (HF Hub needs an account
   decision).

(Done in v0.3.1: Qwen3-VL reduced-scale validation recorded for
DSpark/Domino/JetSpec/DFlash2 with metadata retagged to `reduced_validation`;
`SUPPORT_MATRIX.md` regenerated from `AlgorithmMetadata` via
`scripts/render_support_matrix.py`.)
