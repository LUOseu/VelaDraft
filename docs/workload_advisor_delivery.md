# Workload advisor delivery ledger

## Repository state

- worktree: `veladraft-workload`
- baseline: `7711eda504f4f28653a11cbb164b2ea673b4d682`
- PR #14 branch: `platform/workload-advisor-cpu`
- PR #15 branch: `platform/workload-advisor-backend` (stacked on #14)
- PR #14 fix commits: `5cf3db0`, `8d2e168`; evidence refresh `7ee3df8`
- PR #15 backend fix commit: `7049cd8`

## Pull requests

- PR #14 (base `main`): https://github.com/LUOseu/VelaDraft/pull/14
- PR #15 (stacked on #14): https://github.com/LUOseu/VelaDraft/pull/15
- PR #14 checks: https://github.com/LUOseu/VelaDraft/pull/14/checks
- PR #15 checks: https://github.com/LUOseu/VelaDraft/pull/15/checks

PR #14 owns CPU contract/statistics/decision/share/evidence fixes. PR #15 adds
the HF backend, execution fixes, backend tests and backend/blocker
documentation on top of that branch. The public CLI hook remains an optional
patch at `docs/workload_advisor_public_hook.patch`; the global CLI is unchanged.

## Rework closure

| ID | Fix commit | CPU regression evidence | Remaining limitation |
| --- | --- | --- | --- |
| A1 strict pairing and holdout-only recommendation | `5cf3db0` | `DecisionTests`: equal-speed baseline, deleted sessions, duplicate AR rows, request/hash mismatch, all-tuning, tuning-does-not-move-holdout-CI | Synthetic records only; real private holdout evidence still required |
| A2 input/correctness/quality gates and exit consistency | `5cf3db0` | `CLITests`: manifest protocol mismatch and invalid failed-row hard failures; `DecisionTests`: failed AR reference and label coverage | No real task scorer run; labels/scorers need customer-owned data |
| A3 JSON scorer crash isolation | `7049cd8` | `ScorerTests`: object/array/bad JSON/required/type cases; scorer exception keeps a valid audit row with `quality.reason=scorer_error` | CPU-only scorer; no GPU output quality measurement |
| A4 applied loader/template/pixel identity and drafter artifact checks | `7049cd8` | `LoaderTests`: revision, processor revision, template and pixel bounds reach the spy loader; hash/template/pixel mismatch rejected. `CheckpointTests`: head/algorithm/source/revision/block-size/hash checks and corrupt-checkpoint isolation | Actual HF snapshot and parity behavior still needs a ready GPU host |
| A5 honest local-client E2E boundary | `7049cd8` | `TimingTests`: preprocess, H2D, generate, actual-device sync and token commit are inside E2E; generate-only remains separate; failures retain elapsed or null+reason | Fake clock; no real CUDA synchronization measurement |
| A6 device, lease, budget, failure lifecycle and persistence | `7049cd8` | `LifecycleTests`: logical->physical UUID mapping, one-card lock/ledger, external occupancy, invalid budget before load, lease conflicts, target-preflight rejection code, all-failed status and flushed records/ledger | Fake occupancy/lease; no real external process or multi-GPU host test |
| A7 structural share redaction | `5cf3db0` | `DecisionTests` and `CLITests` scan nested records/summary/report for prompts, answers, paths and sentinels; share manifest uses `structural_allowlist_v1` | Allowlist covers the declared schema; new arbitrary extensions are intentionally dropped |
| A8 ShareGPT terminal-answer isolation | `5cf3db0` | `ContractTests`: single-turn answer becomes a private label, multi-turn history excludes the terminal answer, conversion report reflects the policy | No model evaluation of converted private rows |
| A9 structured latency and serving-profile normalization | `5cf3db0`, `8d2e168` | `ContractTests`: structured latency round-trip/rejection, generate-only never relabeled E2E, known serving profile normalizes; queue depth retained | Unknown/ambiguous serving profiles are rejected; no serving re-run |

## Executed commands and exit codes

Run on the stacked PR #15 checkout, macOS/arm64, Python 3.12.13:

```bash
.venv-workload/bin/python -m unittest discover -s tests -p 'test_workload_advisor*.py'
# Ran 85 tests ... OK; exit 0

.venv-workload/bin/python -m pytest -q \
  tests/test_workload_advisor_backend.py \
  tests/test_workload_advisor_contract.py \
  tests/test_workload_advisor_cli.py \
  tests/test_workload_advisor_decision.py
# No module named pytest; exit 1. The venv has no pip, so this was not hidden
# or replaced with a fabricated pass. CI owns the pytest environment.

.venv-workload/bin/python scripts/workload_advisor.py validate \
  --manifest examples/workload_advisor/mini/manifest.json
# status ok, 30 requests / 30 groups (6 tuning, 24 holdout); exit 0

.venv-workload/bin/python scripts/workload_advisor.py dry-run \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir docs/workload_advisor_evidence/blocked
# 216 measured runs + 36 warmups, 4 GPU-h cap, predicted_speedup=null,
# ranking_allowed=false; exit 0

.venv-workload/bin/python scripts/workload_advisor.py run \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir docs/workload_advisor_evidence/blocked
# backend_not_ready; exit 3; blocker/plan/validation written; no records.jsonl

.venv-workload/bin/python scripts/workload_advisor.py report \
  --records examples/workload_advisor/fixtures/records.jsonl \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir docs/workload_advisor_evidence/cpu_fixture --share
# exit 0; synthetic=true; insufficient_evidence; deterministic replay
```

The CPU-only PR #14 branch runs `63` tests with the same unittest discovery
command because the backend test module is introduced by PR #15.

## Evidence classification

- **Real host measurement:** none for speculative decoding. The host has no
  CUDA device (`nvidia-smi` absent, no torch), so no GPU timing exists.
- **Real host/environment evidence:** `environment.json`, `readiness.json`,
  `blocked/blocker.json`, `blocked/plan.json`, `blocked/validation.json`.
- **Synthetic CPU fixture evidence:** `cpu_fixture/` and all
  `examples/workload_advisor/fixtures/*.jsonl`; every record is marked
  `synthetic=true` and is excluded from real result tables.
- **CPU fake/spy backend evidence:** A3-A6 regression tests only. They prove
  control flow and record semantics, not GPU speed or HF parity.
- **Legacy aggregate evidence:** `fixtures/legacy_report.json` imports as
  `aggregate_only`; it cannot produce CIs and resolves to
  `insufficient_evidence`.

## Budget and dependencies

- First-round GPU cap: 4 GPU-h.
- Consumed: `0 / 4 GPU-h` (no allocation, no target load, no UUID lock).
- Remaining: 4 GPU-h; optional cumulative extension to 8 GPU-h not requested.
- Dependencies to unblock PR #15: CUDA host with `nvidia-smi`, torch and
  transformers, local Qwen2.5-VL-3B weights, and the published DFlash/DFly
  checkpoints with matching artifact hashes.
- PR #15 status: `blocked` (execution paths CPU-reworked, not GPU-validated).
  It must not be reported as a speedup or `validated` until the minimal GPU
  parity smoke and the correctness/quality/CI gates pass.
