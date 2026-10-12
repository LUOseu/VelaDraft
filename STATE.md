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
1. JetSpec stage: capture v2 (store final hidden state per sample) ->
   `vlm_dflash/jetspec/` (chunked forward-KL + causal mask + teacher metrics) ->
   tests -> smoke -> 2K run. Audit and plan: `docs/upstream/jetspec.md`.
2. DFlash2 stage after JetSpec (experimental training objective; must be
   labeled Experimental per GOAL §8).
3. Deterministic benchmark prompt file for `caption_general`; land one public
   dataset adapter (TextVQA first) to move a workload from planned -> validated.
4. Model cards for released checkpoints (export already emits most fields).
