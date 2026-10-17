# Workload advisor evidence

## CPU fixture run (synthetic)

`cpu_fixture/` was produced by:

```bash
python scripts/workload_advisor.py report \
  --records examples/workload_advisor/fixtures/records.jsonl \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir docs/workload_advisor_evidence/cpu_fixture --share
```

It contains `report.md`, `summary.json`, `decision.json`,
`records.synthetic.jsonl` and the redacted share package. `decision.json` has
`synthetic=true` and
`recommended_action=insufficient_evidence`; the fast-but-wrong candidate is
excluded for correctness and quality, and the exact candidate is excluded only
because the evidence is synthetic. The fixed bootstrap seed (7) and 24
independent holdout groups are recorded. Replaying the same records produces
byte-identical `decision.json` and `summary.json`.

## Backend blocker (no GPU work)

`blocked/` records an attempted `run` on this host:

```bash
python scripts/workload_advisor.py run \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir docs/workload_advisor_evidence/blocked
```

Exit code 3. `blocker.json` lists `torch_missing`, `no_gpu`, `target_missing`
and `checkpoint_missing` (the fixture manifest points at synthetic paths).
`validation.json` and `plan.json` are present; there is intentionally **no**
`records.jsonl`. No GPU was allocated, no lock was taken, and no timing was
invented. GPU-hour budget used: 0; first-round cap: 4 GPU-h; extension cap:
8 GPU-h (not requested).

`environment.json` is the real host snapshot (macOS/arm64, Python 3.12.13, no
`nvidia-smi`, no torch/transformers). `capabilities.json` declares the backend
capability matrix with `state=blocked`.

## Minimal rerun plan on a ready A6000 host

1. Point `target.model_path` at the local Qwen2.5-VL-3B checkpoint and the two
   candidate `checkpoint` paths at the published DFlash/DFly weights; update
   their identity hashes in the manifest.
2. Split the real private rows into 16 tuning / 48 holdout groups and freeze
   the scorers/thresholds/buckets before looking at holdout.
3. Run `dry-run`; confirm the 4 GPU-h cap, 432 measured runs + 36 warmups for
   three candidates and three sessions, and a free GPU UUID.
4. Run the pilot; the backend takes UUID locks, counts all allocated GPU time,
   writes a RunRecord for every request/error and stops at 4 GPU-h.
5. Run `report --share`; only then decide whether the evidence justifies the
   optional 8 GPU-h extension. Record the reason and remaining budget.
