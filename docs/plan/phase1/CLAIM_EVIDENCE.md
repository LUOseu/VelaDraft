# Phase-1 claim / evidence ledger (self-contained)

Vocabulary used throughout this package (matches the route document's final
reporting rule):

- **Implemented**: code exists at a named commit with CPU tests that pin the
  stated behaviour.
- **Independently verified**: checked by a procedure that does not trust the
  implementation's own output (formula oracles re-derived from pinned upstream
  source, invariant tests, duplicate runs by a second process, direct reading
  of the competitor code).
- **Report evidence**: recorded GPU runs with archived artifacts; scales and
  seeds are stated; single-seed unless noted.
- **GPU-pending**: registered, not scp'ed (list in `../REVIEW_REMEDIATION.md`).
- **Hypothesis**: believed, deliberately not evidenced.

Supersedes nothing: the plan-level `../CLAIM_EVIDENCE.md` stays the
work-package ledger; this file is the phase-1 (CVPR-route) view.

| # | Claim (as it may be worded) | Status | Evidence | Gaps / restrictions |
| --- | --- | --- | --- | --- |
| P1 | A single pipeline trains, evaluates and exports seven parallel drafter families against frozen VLM targets, with capability and evidence separated machine-readably | Implemented + report evidence | `SUPPORT_MATRIX.md` (generated from `AlgorithmMetadata`, drift-tested); zoo releases `zoo-v0.3.1`/`zoo-v0.3.2` re-downloaded and validated (`R-20261014-09`/`13`); 443 CPU tests green on `cvpr-route` | "Seven" is a method-count statement, not a quality ranking; audit (`COMPETITIVE_AUDIT.md`) shows comparable breadth exists elsewhere (NeMo Automodel covers MTP/EAGLE/DFlash-family on LLMs) |
| P2 | DFly's upstream two-stage recipe (LK/D-PACE cold start -> end-to-end TV) is selectable and numerically faithful to the upstream loss math, and improves the measured AL over shared CE at the 2K protocol | Implemented + independently verified (CPU) + report evidence (single seed) | Formula oracles and chunk invariance in `tests/test_objectives.py` (re-derived from AngelSpec `d3412bed` + speculators exports); two independent E1 sets bit-exact (`R-20261014-03..07`); canonical run `R-20261014-11`: AL **1.839 vs 1.682** (+9.3%), every slot up, token-exact 128/128, export `objective_profile=upstream_two_stage` + reload demo | **The objective is not a VelaDraft novelty**: AngelSpec trains it, SpecForge ships LK/D-PACE/TV loss terms at `fdcc2edd`. Single seed; no multi-seed or scale claim. |
| P3 | Multi-image / multi-turn supervision is represented explicitly (image spans, per-turn masks, native positions) and consumed without leaking future turns; the representation survives resume and export | Implemented + independently verified (CPU) + report evidence (protocol scale) | Schema v2 + 14 tests (`tests/test_wp2_multiturn.py`, `tests/test_multiturn_layout.py`); supervision invariants incl. late-image rejection, exact-resume, export round-trip (`tests/test_multimodal_supervision.py`, commit `b5691e5`); E2 smoke `R-20261014-08` (2-image/2-turn, 512/512 image tokens over 2 spans, token-exact 4/4 both checkpoints); design note `../WP2_DESIGN.md` | Smoke scale only; post-R3/R4 protocol re-run is GPU-pending #1; audit: NeMo Automodel main already ships ViSpec multimodal draft training for Qwen2.5-VL - "multimodal drafter training exists" cannot be claimed as new (framing must be DFlash-family + controlled evidence + resources) |
| P4 | DSpark's faithful upstream objective (`train_recipe: upstream_dspark`: per-slot forward KL over full vocab + confidence BCE against the analytic acceptance rate) is implemented with formula-level tests | Implemented + independently verified (CPU); **no GPU run yet** | `tests/test_objectives.py` (KL/TV reference cases, gradient separation, chunk invariance, plan-digest sensitivity), `tests/test_dspark.py` (trainer analytic mode, alpha gate, config gates); convention differences documented in `docs/algorithms/dspark.md` | GPU smoke is the new pending item (#5 in `../REVIEW_REMEDIATION.md`); the confidence target is *reconstructed* from stored final hidden states - a reconstruction claim, not bit-exact upstream reproduction |
| P5 | DocVQA/ChartQA workloads run end-to-end against a frozen checkpoint with a token-exact gate | Report evidence (single checkpoint, protocol scale) | `R-20261014-10`: AL 1.201 / 1.221, token-exact 128/128, pinned manifests + shard hashes | One checkpoint, one seed; other algorithms not measured on these workloads |
| P6 | Batched online capture reduces per-row target forwards 4x with numerically identical training steps | Implemented + report evidence (with a superseded metric) | `R-20261014-12`: 16 -> 4 forwards, losses bit-identical, peak unchanged; CPU order/record-equality tests | The extraction-wall "-18.8%" is **superseded/invalid** (R2 timer boundary); paired re-measurement is GPU-pending #2; no claim of a measured runtime saving yet |
| P7 | A real serving engine accepts and runs an exported DFlash draft | PARTIAL - blocked at the token-exact gate | `R-20261014-14`: vLLM 0.22.1 loads + runs (token-exact 5/32, AL 1.4059 vs in-repo 1.6924); `R-20261014-15`: MRoPE-vs-1D excluded as the driver; `R-20261014-16`: small margins observed at 4 flips, root cause unresolved | **No deployment or speed claim**; gate stays blocked at 5/32; same-engine diagnosis is GPU-pending #4 |
| P8 | The stack works on a non-Qwen VLM family (first step) | Report evidence (single-image smoke, smoke scale) | `R-20261014-17`: SmolVLM2-2.2B through prepare->regen->capture->train->eval, capture 1377/1377 tokens, eval token-exact 4/4 | Export closure (fresh-process reload) is GPU-pending #3; multi-image/online untested for the family; audit notes competitor VLM draft scope is Qwen-VL only (a scope advantage, not a quality claim) |
| P9 | Cost structure of the recorded protocol is understood (teacher vs training vs eval shares; feature bytes by length; cache-plan comparison) | Analysis on measured data (`PROFILING.md`, `CACHE_DECISION.md`) | All inputs are recorded runs; every number tagged [M]/[D]/[F] with its source | No new measurement; length-bucketed teacher timing and paired online re-timing are pending; the mixed-materialization planner is **not justified** by the simple baselines (decision recorded) |

## Claims explicitly NOT made (audit-driven)

1. First/only multimodal drafter training, or that the multimodal supervision
   representation is novel on its own (NeMo ViSpec on main, Qwen-VL scope).
2. Novelty for LK / D-PACE / end-to-end-TV objectives, chunked teacher
   reconstruction, training-time real acceptance evaluation, sequence packing,
   or image-span query compression (precedents pinned in
   `COMPETITIVE_AUDIT.md`).
3. Any speedup in a real engine (P7 blocked), any multi-seed learning claim,
   any scaling-law language, any AL comparison across targets.
4. The superseded WP4 extraction-wall percentage, and any projection that is
   not labeled as such.

## Open risks that could invalidate a future claim

- Competitor movement: SpecForge and NeMo Automodel both pushed within days of
  this audit; the comparison table must be re-pinned at writing time.
- Exactness: the vLLM divergence (P7) is unexplained; if it is a semantics
  difference rather than a fixable bug, the deployment story changes.
- Resource environment: shared-host contention (1.7x wall variance) and 61 GB
  free disk bound what can be replicated on the current hardware.
