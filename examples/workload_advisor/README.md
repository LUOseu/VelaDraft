# Workload advisor examples

All data here is **synthetic** and generated deterministically by
`generate_fixtures.py`. It exercises the CPU contract, statistics, decision and
report paths; it is not accelerometer evidence and must not be copied into a
real results table.

- `mini/manifest.json` – 30 groups (6 tuning, 24 holdout) across invoice
  fields, chart QA and general VQA, with frozen target/decode/scorer/bucket
  settings and SHA256 for rows/labels.
- `mini/rows.jsonl` – prompt-only request rows; labels live separately in
  `mini/labels.jsonl`.
- `mini/media/` – 1×1 PNG placeholders with valid image magic.
- `fixtures/records.jsonl` – 3 sessions × 24 holdout groups × 3 candidates:
  AR, a token-exact speculative candidate (`dflash-exact`) and a faster
  candidate that fails token comparison (`dfly-fast-wrong`). Every record is
  `synthetic=true` and carries null TTFT/ITL with `no_streaming` /
  `no_token_visibility` reasons.
- `fixtures/records_ar_better.jsonl` – same shape but the candidate is slower
  than AR, exercising `retain_ar`.
- `fixtures/legacy_report.json` – an old `vlm-dflash-benchmark-v1` aggregate
  report used to exercise `import-legacy` / `aggregate_only` handling.

Regenerate:

```bash
python examples/workload_advisor/generate_fixtures.py
```

The generator is deterministic: rerunning it must produce identical bytes.
`tests/test_workload_advisor_*.py` cover bad JSON/paths/leaks/missing media,
unsupported inputs, protocol-mismatched AR cache, non-exact candidates,
aggregate-only imports, null streaming metrics, deterministic replay and the
share-package redaction.
