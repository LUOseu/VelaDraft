# Workload advisor delivery ledger

## Repository state

- worktree: `veladraft-workload`
- branch: `platform/workload-advisor`
- baseline: `7711eda504f4f28653a11cbb164b2ea673b4d682`
- PR 1 commit: `72e1796` — CPU contract, statistics, decision, reports
- PR 2 commit: `88c8318` — HF backend, GPU lease/ledger, blocker evidence

## Pull requests

- PR 1 (base `main`): https://github.com/LUOseu/VelaDraft/pull/14
- PR 2 (stacked on PR 1): https://github.com/LUOseu/VelaDraft/pull/15

PR 2's diff contains only `backend_hf.py` and the backend/blocker documentation.
The public CLI hook is an optional patch at
`docs/workload_advisor_public_hook.patch`; applying it is not required to run
`scripts/workload_advisor.py`, and the global CLI is unchanged in both PRs.

## Change list

PR 1:

- `vlm_dflash/workload_advisor/` contract, validation, conversion, records,
  statistics, decision, report, plan, readiness, CLI;
- `scripts/workload_advisor.py` entry point;
- `tests/test_workload_advisor_{contract,decision,backend,cli}.py`;
- deterministic fixtures and example workload;
- CPU evidence in `docs/workload_advisor_evidence/cpu_fixture/`;
- public hook patch.

PR 2:

- `vlm_dflash/workload_advisor/backend_hf.py`;
- `docs/workload_advisor.md`;
- blocked-run evidence in `docs/workload_advisor_evidence/`.

## Test commands and results

```bash
.venv-workload/bin/python -m unittest discover -s tests -p 'test_workload_advisor*.py'
# Ran 44 tests ... OK
```

Also executed:

```bash
python scripts/workload_advisor.py validate --manifest examples/workload_advisor/mini/manifest.json
# status ok, 30 requests / 30 groups (6 tuning, 24 holdout)

python scripts/workload_advisor.py dry-run --manifest examples/workload_advisor/mini/manifest.json --out-dir docs/workload_advisor_evidence/blocked
# 216 measured runs + 36 warmups, 4 GPU-h cap, predicted_speedup=null, ranking_allowed=false

python scripts/workload_advisor.py run --manifest examples/workload_advisor/mini/manifest.json --out-dir docs/workload_advisor_evidence/blocked
# exit 3; blocker.json written; no records.jsonl

python scripts/workload_advisor.py report --records examples/workload_advisor/fixtures/records.jsonl --manifest examples/workload_advisor/mini/manifest.json --out-dir docs/workload_advisor_evidence/cpu_fixture --share
# exit 0; synthetic=true; insufficient_evidence; byte-stable replay
```

## Evidence classification

- **Real host measurement:** none for speculative decoding. The host has no
  CUDA device (`nvidia-smi` absent, no torch), so no GPU timing exists.
- **Real host/environment evidence:** `environment.json`, `readiness.json`,
  `blocked/blocker.json`, `blocked/plan.json`, `blocked/validation.json`.
- **Synthetic CPU fixture evidence:** `cpu_fixture/` and all
  `examples/workload_advisor/fixtures/*.jsonl`; every record has
  `synthetic=true` and is excluded from real result tables.
- **Legacy aggregate evidence:** `fixtures/legacy_report.json` imports as
  `aggregate_only`; it cannot produce CIs and resolves to
  `insufficient_evidence`.

## Budget and dependencies

- First-round GPU cap: 4 GPU-h.
- Consumed: 0 GPU-h (no allocation, no UUID lock).
- Remaining: 4 GPU-h; optional cumulative extension to 8 GPU-h not requested.
- Dependencies to unblock PR 2: CUDA host with `nvidia-smi`, torch/transformers,
  local Qwen2.5-VL-3B weights, and the published DFlash/DFly checkpoints.
- PR 2 status: `blocked` (backend implemented, not GPU-validated). It must not
  be reported as a real speedup until the 4 GPU-h pilot is executed and the
  correctness/quality/CI gates pass.
