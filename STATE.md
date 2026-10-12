# STATE

Updated: 2026-10-12 (v0.2 + DSpark stage in development; nothing pushed yet).

## Current stable features
- Staged pipeline: `prepare -> regen -> capture -> train -> eval` (+ `export`);
  training reads staged features only (target off GPU).
- Runtime model contract + provenance gates (target/processor/identity).
- Qwen2.5-VL-3B and Qwen3-VL-8B adapters; single image; tied and untied heads.
- Verification: `sequential` (bitwise-exact gate), `parallel` /
  `strict_parallel` (experimental throughput; not lossless).
- Unified CLI (`doctor/train/eval/export`), presets, resume, HF-style export.
- Algorithm registry with DFlash as the only registered plugin (frozen).
- DSpark plugin (experimental): Markov logit-bias head + confidence head on
  the frozen DFlash backbone; `sample_from_anchor` both modes; CPU/synthetic
  tests green; canonical VLM runs pending (see ALGORITHM_MATRIX.md).
- Gradient coverage audit + confidence calibration artifacts; namespaced
  algorithm config; export loader dispatch by algorithm.

## Experimental
- Benchmark harness skeleton (`scripts/benchmark.py` + manifests); only the
  scaffold workload (`caption_general`, run's own val split) is runnable;
  TextVQA/DocVQA/ChartQA manifests pin protocol but have no adapter yet.
- Parallel runtime speedup numbers (see BENCHMARKS.md) — experimental.

## Latest measured results
- 3B / 512-corpus rerun: AL 1.198, 64/64 exact (earlier v0.1 run: 1.220).
- 3B / 2K-corpus: AL 1.630, 128/128 exact, slots `[.413,.149,.048,.015,.004,.001,0]`.
- 3B / 5K-corpus: AL 1.817, 128/128 exact, slots
  `[.482,.210,.080,.030,.011,.005,.001]` (1974 steps, CE 6.84 -> 2.58, peak
  30.0 GB).
- 8B / 512-corpus: AL 1.322; 8B / 2K: running (suite, started 23:31; ETA
  ~03:00).
- 8B / 2K-corpus: AL 1.665, 32/32 exact, slots
  `[.413,.160,.063,.027,.010,.002,.0004]` (1996 kept rows, 1500 steps, final
  loss 2.69, visual-path check passed; speedup 0.81x).
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
- Decomposition so far: serving-time selector ablation (16 prompts, same
  checkpoint) AL 2.060 (selector) vs 1.965 (unary-only ranking of the same
  candidates); ~60% of proposal positions are re-picked, gains concentrated in
  deep slots. Local-mixing ablation training (`conv_kernel_size: 1`,
  `selector_loss_alpha: 0.0`, run `qwen25vl3b_dflash2_2k_ablate_nomix`) is the
  control for the convolution; eval pending.
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

## Known bugs / caveats
- bf16 capture/training: protocol-reproducible, not bit-identical.
- `parallel` verification exactness is numerically unreachable with the
  current block-forward path (see docs/REFERENCE.md); do not advertise as
  lossless.

## Blocked / waiting
- 8B-2K scaling job on the cluster (started 23:31 after pruning the completed
  8B-512 / 3B-refresh features to pass the disk pre-check); `/home` stays
  near-full, the suite pre-checks and skips rather than failing.
- Hugging Face / PyPI publishing: pending release audit.

## Next highest-value tasks
1. Finish the DFlash2 decomposition: local-mixing ablation (running) answers
   whether the convolution (vs the KL objective) carries the gain; if positive,
   a `unary_loss: ce` control isolates the objective axis. Then fill the
   ablation rows in `BENCHMARKS.md`.
2. Algorithm-zoo wrap-up for the GOAL report: regenerate the support matrix
   from `AlgorithmMetadata` (currently hand-maintained), and add Qwen3-VL
   reduced-scale validation for DSpark/Domino (already `experimental` there).
3. Serving-fidelity follow-ups from the JetSpec stage: decide whether the
   `inference_in_block_causal` mask path should be exercised by a
   `strict`-style test on a GPU CI run (currently unit-tested + smoke-tested).
4. Deterministic benchmark prompt file for `caption_general`; land one public
   dataset adapter (TextVQA first) to move a workload from planned -> validated.
5. Model cards for released checkpoints (export already emits most fields).
