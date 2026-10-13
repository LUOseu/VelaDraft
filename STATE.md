# STATE

Updated: 2026-10-13 (v0.3.2 in development: online target feature source
pushed; GitHub repo renamed to `LUOseu/VelaDraft`; DFlare/DFly AngelSpec
ports landed behind the same interfaces — canonical 2K runs, ablations and
Qwen3-VL reduced validations recorded).

**Status: v0.3.1 tagged and pushed** (tag `v0.3.1` at `eb3f314`); **v0.3.2 in
development — the online target feature source is pushed as `aed7bf8`** (repo
now `github.com/LUOseu/VelaDraft`; old URLs redirect; distribution and CLI
stay `vlm-dflash`).

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
- Algorithm registry with seven drafters behind one interface: DFlash (stable
  frozen reference), DSpark / Domino / JetSpec (validated on the 2K
  apples-to-apples protocol), DFlash2 (experimental objective; upstream does
  not publish it), DFlare / DFly (AngelSpec ports — canonical evidence in
  the stage section below). See `ALGORITHM_MATRIX.md`, `SUPPORT_MATRIX.md`,
  `BENCHMARKS.md`.
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
  after 4 steps — plumbing only, not quality.

## DFlare / DFly stage (2026-10-13, stage 6 — AngelSpec ports, evidence recorded)

- Scope: port `Tencent/AngelSpec @ d3412bed` DFlare (learnable per-draft-layer
  fusion + dedicated context K/V) and DFly (DFlash shared-KV backbone + FC
  context + fusion residual + zero-init hidden-states correction) into the
  algorithm-zoo interfaces. Both keep the frozen DFlash block-CE objective,
  slot convention and recipes; audits in `docs/upstream/dflare.md` /
  `docs/upstream/dfly.md`, port notes in `docs/algorithms/dflare.md` /
  `docs/algorithms/dfly.md`.
- Feature contract: explicit-id layer slicing — a single capture superset
  `[1, 9, 17, 25, 33]` serves the matched comparison (context `[1, 17, 33]`,
  identical to the frozen DFlash routing) and the upstream 5-layer recipe;
  `contract_defaults_from_config` extracts `algorithm_context_layer_ids` and
  the capture-coverage check validates the configured algorithm's consumed
  layers (not always the draft routing).
- Tests: `tests/test_dflare.py` (12), `tests/test_dfly.py` (12),
  `tests/test_feature_contract.py` (7); full CPU suite 228 passed / 6 skipped.
  Gradient coverage: training-time step-0 audits are 42/42 for DFlare and
  38/42 for DFly — the 4 zero-grad tensors are the zero-init correction's
  gate/up path (`down_proj` alone is kicked off at step 0, unit-tested);
  the new `scripts/diag_gradient_coverage.py` replays each canonical run's
  feature store against `ckpt_final.pt` and reports 42/42 nonzero on the
  first replayed backward (both algorithms).
- GPU smokes (Qwen2.5-VL-3B, COCO 64-row): DFlare train + overfit + visual
  check, eval token-exact 8/8, AL 1.123 vs init 1.000; export verified
  (`method: dflare`, `method_details.separate_context_kv=true`). DFly smoke
  eval token-exact 8/8, AL 1.438; correction delta 0 → 0.75, fusion entropy
  stable (prior-dominated at smoke scale).
- Fusion diagnostics honesty: fusion logits train through the fp32 optimizer
  masters, but the live bf16 parameter quantises single AdamW steps, so in-log
  fusion metrics move coarsely; the trainer now attaches fp32 masters and
  logs fp32-resolution `fusion_*` metrics plus `fusion_prob_l1_drift` (the
  2K canonical run predates this and has a `train/fusion_master_summary.json`
  extracted from its step-594 `ckpt_last.pt`: fusion probabilities moved
  L1 0.011 from the distance prior, per-layer entropy
  [0.496, 0.671, 0.396] vs prior [0.441, 0.668, 0.439]).
  Ablation checkpoints extracted at the same step-594 save: shared-KV drift
  0.0133, no-correction 0.0026, fixed-prior 0.0 — the knob turns the
  mechanism off exactly; 8B reduced (step 144) drift 0.0014 / 0.0003,
  correction `down_proj` |w| 0.0025.
- Canonical runs recorded (Qwen2.5-VL-3B 2K, 128 prompts, all token-exact;
  numbers in `BENCHMARKS.md`): DFlare matched `[1,17,33]` **AL 1.637**
  (frozen baseline 1.630, re-checked bit-identically), fixed-prior ablation
  1.637, shared-KV ablation **1.579** (the dedicated context K/V is the
  effective component; the learned fusion is prior-dominated), layer-rich
  `[1,9,17,25,33]` **1.670**; DFly matched **AL 1.682** (hidden-states
  correction worth +0.113 — no-correction ablation 1.569), layer-rich
  **1.709**. Qwen3-VL-8B reduced (192 steps, 16 prompts): DFlare 1.185,
  DFly 1.226. All 9 run exports re-rendered with the final algorithm
  metadata; fusion master summaries extracted before pruning `ckpt_last.pt`
  resume states (~33 GB freed).

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
  On 2026-10-13 `/home` hit 100% (a concurrent eval died with ENOSPC while
  writing its result); freed ~9.6G by deleting `ckpt_last.pt` (fp32 AdamW
  masters, ~3.2G each) from completed runs — resume states for finished runs
  only. The DFlare/DFly 2K pair share one 12G capture superset store
  (`runs/qwen25vl3b_dflare_2k/features`, symlinked by dfly/ablations); every
  other new run reuses it, and `ckpt_last.pt` is pruned after each run's eval
  so the remaining stage-6 runs need only ~1.5G of net new space each.

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
