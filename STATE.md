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
- Runtime probe (32 prompts, bf16): see BENCHMARKS.md.

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
1. Record the 8B-2K scaling numbers when the suite finishes; update
   README/BENCHMARKS (5K already recorded).
2. Deterministic benchmark prompt file for `caption_general`; land one public
   dataset adapter (TextVQA first) to move a workload from planned -> validated.
3. Model cards for released checkpoints (export already emits most fields).
