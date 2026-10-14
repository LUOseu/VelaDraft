# Local pilot report — CVPR-route phase 1 (2026-10-14, CPU only)

Self-contained report of the phase-1 execution requested by the route
document §15. Branch `cvpr-route` of `github.com/LUOseu/VelaDraft`; the
phase used **0 GPU-hours** (no large-compute request; per §6 the formal
matrix needs the gate in `GO_NO_GO.md` first). Everything a reader needs to
audit this phase is inside this directory plus the two docs it cites by
relative path.

## Baseline and what changed

- Baseline reviewed: `origin/main` `e15256d` (merged PR #4), containing the
  review-remediation through `5db73c7`. Phase work happens on `cvpr-route`,
  pushed after each item.
- `a80f564` — DSpark faithful objective (`train_recipe: upstream_dspark`):
  per-slot forward KL over the full vocabulary + confidence BCE against the
  analytic acceptance rate `sum_v min(p,q)`, teacher logits reconstructed
  chunk-wise from stored final hidden states; per-recipe supports-matrix,
  config gates, resume/plan-digest integration, docs, CPU reference tests.
- `64cef1d` — per-method training-signal declarations (`TrainingSignals` on
  every algorithm) + generated `docs/TRAINING_SIGNALS.md` + drift test.
- `b5691e5` — multimodal supervision representation: the phase-1 design note
  (`../WP2_DESIGN.md` field-by-field contract), 9 supervision-invariant tests
  (`tests/test_multimodal_supervision.py`), and the export/model-card record
  of visual conditioning.

## §15 item status (1-7)

| Item | Deliverable | Status this phase | Where |
| --- | --- | --- | --- |
| 1 | R1-R6 review items | Done in the previous round (merged); re-checked against `950b394`; no re-modification | `../REVIEW_REMEDIATION.md` (traceability + acceptance mapping) |
| 2 | Training-signal list per method | Delivered: declarations drive a generated doc (no drift possible) | `docs/TRAINING_SIGNALS.md`, `tests/test_algorithms.py`, commit `64cef1d` |
| 3 | DSpark faithful objective vs upstream | Delivered on CPU with a pinned upstream reference (speculators `55ebbd5b`); GPU smoke registered as pending | `docs/algorithms/dspark.md`, `docs/upstream/dspark.md`, `tests/test_objectives.py`, `tests/test_dspark.py`, `UPSTREAM_AUDIT.md` |
| 4 | Multimodal supervision: design note + minimal implementation + invariance/loss/grad/resume/export tests | Delivered (implementation already existed from schema v2; this phase added the design note, the invariant tests, and the export record) | `../WP2_DESIGN.md`, `tests/test_multimodal_supervision.py`, `vlm_dflash/export/hf.py` |
| 5 | Limited profiling by visual/text length | Delivered as an assembled study (measured shares + explicit formulas); no new GPU measurement, gaps named | `PROFILING.md` |
| 6 | all-online vs all-offline vs simple fixed-space cache | Delivered as a decision document: simple plans cover the recorded geometry; the mixed planner is **not** implemented and the revisit triggers are recorded | `CACHE_DECISION.md` |
| 7 | LOCAL_PILOT_REPORT / CLAIM_EVIDENCE / RUN_MANIFEST / RESOURCE_ESTIMATE / GO_NO_GO, self-contained | This directory | `LOCAL_PILOT_REPORT.md`, `CLAIM_EVIDENCE.md`, `RUN_MANIFEST.md`, `RESOURCE_ESTIMATE.md`, `GO_NO_GO.md` |
| (constraint) | Competitive audit of SpecForge / Speculators / AngelSpec / NeMo Automodel / TIDE | Delivered with pinned commits and file-level evidence | `COMPETITIVE_AUDIT.md` |

CPU regression on this branch: **453 passed / 6 skipped** (443 at the
`191da28` package commit, +10 length-bucket profiler tests; baseline before
this phase: 434/6).

## The four-way classification (route §15 final-reporting rule)

**Implemented (code + CPU tests at the named commits)**
- `upstream_dspark` recipe end-to-end on the shared trainer core: objective
  math, trainer wiring, config gates, checkpoint metadata, plan-digest
  integration (`a80f564`).
- Training-signal declarations for all seven algorithms (`64cef1d`).
- Supervision invariants and export record for visual drafts (`b5691e5`).

**Independently verified (not by trusting the implementation's own output)**
- Objective formulas re-derived against the pinned upstream source
  (speculators `55ebbd5b`): KL/confidence references, double-precision
  references, gradient separation, chunk invariance.
- E1's duplicate concurrent validation set (two sessions, bit-exact) for the
  DFly recipe; E4's AL delta is a *report* result, not a verification.
- Supervision semantics checked by invariants that fail independently of the
  code path (padding rows inert, text-only batch an exact no-op, late images
  rejected, resume reproduces the uninterrupted run bit-for-bit in fp32).
- Upstream competitor claims verified by reading pinned code, not by their
  READMEs (`COMPETITIVE_AUDIT.md`).

**Report evidence (recorded GPU runs; scale stated)**
- E2 multi-image smoke, E3 DocVQA/ChartQA, E4 DFly recipe (single seed),
  WP4 A/B (with the superseded metric marked), WP5 bridge (blocked at 5/32),
  WP6 SmolVLM2 smoke. Inventory: `RUN_MANIFEST.md`.

**待 GPU 验证 (registered, not run; commands in `../REVIEW_REMEDIATION.md`)**
- Post-R3/R4 multi-image re-check (#1), paired online re-timing (#2),
  SmolVLM export closure (#3), same-engine vLLM diagnosis (#4), the new
  `upstream_dspark` GPU smoke (#5), length-bucketed profiling (#6) and the
  second recipe seed (#7). Estimated 2.2-3.75 GPU-h total
  (`RESOURCE_ESTIMATE.md`).

**Technical hypotheses (stated, not evidenced)**
- That the visual-context budget question (route §4.2's successor idea) can
  produce a measurable cost/acceptance trade-off across DFlash/DFlare/DFly -
  untested; ViSpec-style query compression is prior art for the mechanism
  (`COMPETITIVE_AUDIT.md`).
- That a fixed-space cache policy adds measurable value over
  shortest-first materialization at corpus scale - analysis says the gain is
  a bounded constant, so the hypothesis is currently *negative*
  (`CACHE_DECISION.md`).
- That the vLLM divergence is a fixable semantic gap rather than a
  fundamental mismatch - unresolved (#4).

## What phase-2 should do first (if and when GO)

1. Clear the registered pending GPU items (bounded, 2.2-3.75 GPU-h) - in
   particular #2 (paired timing), #5 (DSpark smoke) and #7 (second seed),
   because they feed the §6.1 execution-benefit, faithfulness and
   learning-effectiveness gates.
2. Re-run the competitive audit re-pins before writing any comparison table.
3. Only then evaluate the §6.1 gate again (`GO_NO_GO.md`) before any
   multi-seed or multi-node request.
