# Benchmark workloads

These manifests drive the historical HF checkpoint-diagnostic harness. For
paired native vLLM request E2E, use `vlm-dflash native-eval` and the
[protocol guide](../docs/benchmark.md). Prepared-input HF ratios remain labeled
diagnostics; they are not promoted into the native E2E headline.

Fixed protocols for the canonical drafter benchmark. One manifest per
workload; the harness refuses to guess protocol values.

- `status: validated` — run end-to-end at least once against a real checkpoint.
- `status: scaffold` — runnable now, but the prompt set is provisional (e.g.
  the run's own held-out split rather than a fixed public benchmark file).
- `status: planned` — dataset adapter not landed yet; the manifest only pins
  the protocol the workload *will* use. The harness lists these and skips them.

Run:

```bash
python scripts/benchmark.py --run-dir runs/<run> --workload caption_general
python scripts/benchmark.py --run-dir runs/<run> --list
```

Reports land in `benchmarks/reports/` (gitignored); promote numbers into
`BENCHMARKS.md` only after checking the report's `protocol` block matches the
table header it is being compared under.
