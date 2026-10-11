# STATE

Updated: 2026-10-11 (v0.2 in development; nothing pushed yet).

## Current stable features
- Staged pipeline: `prepare -> regen -> capture -> train -> eval` (+ `export`);
  training reads staged features only (target off GPU).
- Runtime model contract + provenance gates (target/processor/identity).
- Qwen2.5-VL-3B and Qwen3-VL-8B adapters; single image; tied and untied heads.
- Verification: `sequential` (bitwise-exact gate), `parallel` /
  `strict_parallel` (experimental throughput; not lossless).
- Unified CLI (`doctor/train/eval/export`), presets, resume, HF-style export.
- Algorithm registry with DFlash as the only registered plugin (frozen).

## Experimental
- Benchmark harness skeleton (`scripts/benchmark.py` + manifests); only the
  scaffold workload (`caption_general`, run's own val split) is runnable;
  TextVQA/DocVQA/ChartQA manifests pin protocol but have no adapter yet.
- Parallel runtime speedup numbers (see BENCHMARKS.md) — experimental.

## Latest measured results
- 3B / 512-corpus rerun: AL 1.198, 64/64 exact (earlier v0.1 run: 1.220).
- 3B / 2K-corpus: AL 1.630, 128/128 exact, slots `[.413,.149,.048,.015,.004,.001,0]`.
- 3B / 5K-corpus: **running** (scaling suite; will be appended to BENCHMARKS.md).
- 8B / 512-corpus: AL 1.322; 8B / 2K: queued behind the 5K job.
- Runtime probe (32 prompts, bf16): see BENCHMARKS.md.

## Known bugs / caveats
- bf16 capture/training: protocol-reproducible, not bit-identical.
- `parallel` verification exactness is numerically unreachable with the
  current block-forward path (see docs/REFERENCE.md); do not advertise as
  lossless.

## Blocked / waiting
- 5K and 8B-2K scaling jobs on the cluster; `/home` disk pressure can skip
  jobs (the suite pre-checks and skips instead of failing).
- Hugging Face / PyPI publishing: pending release audit.

## Next highest-value tasks
1. Finish + record 5K (and 8B-2K) scaling numbers; update README/BENCHMARKS.
2. Deterministic benchmark prompt file for `caption_general`; land one public
   dataset adapter (TextVQA first) to move a workload from planned -> validated.
3. Model cards for released checkpoints (export already emits most fields).
