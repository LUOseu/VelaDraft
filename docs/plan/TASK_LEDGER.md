# VelaDraft plan - task ledger

Statuses: `DONE` / `IN_PROGRESS` / `PENDING` / `GPU_PENDING` / `BLOCKED`.
Baseline `9137de0`. Every "done" line points at code/tests/runs, not prose.

## Capability gap table (verified against the worktree on 2026-10-14)

| Capability | Verified baseline state | This plan |
| --- | --- | --- |
| 7 algorithms unified train/eval/export | `SUPPORT_MATRIX.md` generated from `AlgorithmMetadata`; all 7 through Export (DFly/DFlash2 training marked experimental) | keep interfaces |
| Offline features + online live target capture | `vlm_dflash/target/{capture,online}.py`, `docs/online-training.md` | WP4 batch online capture |
| DFlare/DFly review fixes | merged (`ff406bb` via PR #2 = baseline) | read-only |
| DFly upstream two-stage recipe | ported on `dfly-upstream-recipe` (WP1, not yet on main) | WP1 |
| Image input | single image + text only (`multi_image: unsupported` in every `AlgorithmMetadata`) | WP2 |
| TextVQA benchmark | adapter + pinned 128-prompt manifest validated (`benchmarks/manifests/02_textvqa.json`) | reuse (WP3) |
| DocVQA / ChartQA | manifests `planned`; no adapter | WP3 |
| Serving bridge | `docs/serving-bridge.md` = proposal only (SGLang/vLLM thin adapter scoped) | WP5 |
| Target family | Qwen2.5-VL-3B + Qwen3-VL-8B only | WP6 non-Qwen |
| Docs homepage | capability-first README (AL ranking removed in `8d6c54c`) | keep |

## Work packages

| WP | Deliverable | Status | Evidence |
| --- | --- | --- | --- |
| WP0 | Baseline audit, task ledger, budget, capability list | DONE | this directory; CPU suite green (321 tests) |
| WP1 | DFly staged recipe + reference numeric validation + resume + export metadata | DONE (PR opened for review) | code `66b595a`, docs `c425850`, zoo export entries `6cf8141`, smoke `R-20261014-01`; E1 PASSED twice (`R-20261014-03..06` and the duplicate set `R-20261014-07`, bit-exact select-parity / resume / export metadata) |
| WP2 | Multi-image / multi-turn: schema v2, data, features, supervision, position protocol | PENDING | - |
| WP3 | Reuse benchmark harness; DocVQA/ChartQA adapters; training-cost report | PENDING | TextVQA adapter reused; manifests 03/04 planned |
| WP4 | Batched online capture + at most one profiled I/O optimization + measurement report | PENDING | online capture exists (`target/online.py`) |
| WP5 | One fixed runtime (SGLang or vLLM) DFlash real deployment loop | PENDING | `docs/serving-bridge.md` proposal scoped the first step |
| WP6 | One non-Qwen VLM family, DFlash first | PENDING | adapters: qwen25_vl, qwen3_vl only |
| WP7 | Reference weights, fixed experiments, model cards, paper outline + evidence table | PENDING | candidate claims in CLAIM_EVIDENCE.md |

## Fixed experiment protocol (E0-E4, reconstructed)

Executed in order; each has a fixed run ID, budget and success criterion in
RUN_LOG.md before launch. Negative results are kept and recorded.

| ID | What | Success criterion | Budget |
| --- | --- | --- | --- |
| E0 | CPU regression suite on the working branch (no GPU) | `pytest tests/ -q` exit 0 | 0 GPU-h |
| E1 | WP1 validation on the real target: (a) `train_recipe: controlled_ce` select-parity vs legacy CE loss trace; (b) 4->8-step resume round-trip across the stage boundary gate; (c) export + reload metadata records `objective_profile`; (d) peak-memory re-measure | (a) same loss trace within bf16 protocol reproducibility; (b) resume log matches straight-through run; (c) export reload token-exact on a tiny prompt set, metadata present; (d) fits 48 GB | 0.3 GPU-h |
| E2 | WP2 multi-image/multi-turn end-to-end smoke (prepare -> capture -> train -> eval) | token-exact eval on the smoke protocol; schema v2 round-trip | 0.5 GPU-h |
| E3 | WP3 DocVQA + ChartQA adapter validation against a frozen DFlash checkpoint | token-exact vs the in-repo verifier on pinned prompt sets; reports under `benchmarks/` | 0.7 GPU-h |
| E4 | DFly 2K canonical recipe run vs matched controlled_ce baseline | both train + eval; record AL + token-exactness; no positive-outcome requirement | 1.0 GPU-h + reserve |

(WP4-WP6 measurement/verification jobs are budgeted in RESOURCE_LEDGER.md and
logged individually in RUN_LOG.md; they are not additional protocol rows.)
