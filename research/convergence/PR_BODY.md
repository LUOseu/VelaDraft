# D1 convergence round: regroup identity bypass fix + eval accounting + benchmark/A100 tooling

Branch `research/rsi-d1-conv` (base `origin/main` @ `6d99f04`). Regroup stays
default-off; nothing here changes shipped training semantics unless
`train.regroup.mode` is set explicitly.

## Code changes

- **`vlm_dflash/dflash/train.py` — identity-layout bypass for `train.regroup`**
  (`f65e67f`). The only remaining identity seam between the regroup machinery and
  the shipped path was the loss reduction (bf16 mean vs fp32 preserved-weight
  sum); the previously reported "2048 vs ~1k forward width" seam does not apply
  (feature batches are length-trimmed and collated to batch max). With
  `train.regroup.identity_bypass=true` (default), identity layouts re-run the
  shipped `module(...)` call verbatim; the legacy machinery path stays available
  as an explicit control (`identity_bypass=false`). Verified bitwise on the
  real training loop (3-step smokes, W3 store: loss/grad_norm bitwise identical
  to regroup-off; the machinery path reproduces the historical ±1 ulp signature).
- **`vlm_dflash/eval/run.py` — per-prompt acceptance accounting** (`5750c23`):
  `eval_samples.jsonl` now carries `verify_steps` / `accepted_tokens` /
  `draft_tokens` per prompt (prerequisite for the preregistered paired
  non-inferiority stats).
- **`scripts/convert_train_state_ckpt.py`** (`bc87e8f`): mid-run `ckpt_last.pt`
  snapshots are exact-resume payloads (`vlm-dflash-train-state-v1`); eval/export
  need `vlm-dflash-ckpt-v1` headers. The converter grafts the train-state
  weights onto `ckpt_init.pt`'s provenance block — stability-curve checkpoints
  become evaluable without re-training.
- Tests: `tests/test_regroup.py` (bypass bitwise on canonical + sorted plans,
  legacy control close, split never bypasses), `tests/test_convert_train_state_ckpt.py`.

## Research tooling (default-off, no product behavior)

- `research/convergence/` — frozen pre-registration, settle-rule analyzer,
  quality gates (hierarchical paired bootstrap), run/chain/stability/harness
  launchers, evidence.
- `research/convergence/a100/` — config-driven experiment matrix for the full
  A100 round: `run_matrix.py` (dry-run / run / resume / summarize; 7 drafters ×
  {controlled, native} × seeds × regroup factor, ~37 GPU-h), `RUNBOOK.md`
  (tracks, isolation, adaptation gates), `regroup_adaptation_smoke.sh`.

## Evidence so far

- Identity bypass: bitwise smokes (off vs canonical/8+bypass) + machinery
  control signature reproduced (see `research/convergence/STATE.md` F1/F2).
- Capacity probe (batch 16, A6000 48GB): canonical micro-16 OOM; canonical
  micro-8 fits at 24.21 GiB peak / 12.6% padding; split sub2 9.37 GiB / 4.9%;
  grid sub2 9.37 GiB / 3.3% with wall −18.8% vs canonical micro-8
  (`research/convergence/evidence/capacity/`).
- Systems A/B series (3 reps × arms, `evidence/systems_table.md`): speedup vs
  canonical on W3 k4 = split 1.061 / token 1.111 / dual 1.110 / cost 1.133 /
  grid 1.131, peak 25.05 → 10.64 GiB (−57%), padding 11.8% → 2.7–3.6%; MIX k4
  grid 1.156 (pad 20.1% → 6.0%); k2 gains shrink (grid 1.097); **W1 k4 all
  regroup arms slower (0.865–0.873) — low-memory-pressure no-gain boundary**.
  Identity controls cp/ref ≈ 1.000 (loss-identical; machinery overhead ≈ 0).
- Stability/L* (frozen prereg §3, canonical-only): settled(400)=False (0.73%),
  settled(800)=False (6.04%); fresh 1600-step extension settled(1600)=False
  (3.07%) → **L* = 1600 with the explicit "not settled" disclosure**; claim
  scoped to the matched budget. AL declines at longer budgets (1.20@400 →
  1.155@800 → 1.119@1600) — small-store overfit; the arm comparison is a
  matched-budget test.
- Confirmatory quality legs (canonical ×4, grid2 ×4 at L*=1600, val64, paired;
  frozen gates in `PREREGISTRATION.md` §5, hierarchical bootstrap B=10,000):
  per-seed relΔAL = +0.015% / +0.430% / −3.042% / −0.501%; mean −0.775%,
  one-sided 95% lower bound **−2.000%** → **AL gate FAIL** (margin −1%);
  Δacc = −0.00113 (lo −0.00352) → acc gate passes; verdict **FAIL** (both must
  hold). Attribution: step-1 seams 4.4e-4–2.8e-3 (same supervision/anchors);
  seed-level endpoint chaos in the *not settled* regime (curve moves ~3%/800
  steps) — a certification failure, not an established systematic regression
  (t≈−1.0 across seeds). All 16 confirmatory+pilot legs token_exact=True.
  Artifacts: `evidence/quality_data.json`, `evidence/quality_verdict.json`,
  per-leg raw under `evidence/legs/`.
- Pilots (token2 C-only, split2 plain split ×2 seeds; exploratory, non-gating):
  −0.032% (lo −1.06%) and +0.147% (lo −0.35%).
- Full cost table (14 legs): `evidence/costs_table.md` — train wall/GPU-h, peak
  (25.3 → 10.0 GiB for sub2 layouts, −60.5%), eval per-sample breakdown.
- Second-drafter applicability: dflare **bypass bitwise PASS**; regroup
  machinery refused (aux metrics) — code boundary: dspark/domino
  (`sample_from_anchor`), dflash2/jetspec (aux loss), dflare (aux metrics),
  dfly untested (`evidence/second_drafter/`, STATE F9).
- A100 matrix dry-run re-validated (32 entries, 37.3 GPU-h est, 0 errors;
  `evidence/a100_dryrun.txt`).

## Reproduce

```bash
# local tests
PYTHONPATH=. python -m pytest tests/test_regroup.py tests/test_convert_train_state_ckpt.py -q
# server (4×A6000): stability curve -> L* (frozen rule)
bash research/convergence/scripts/vconv_stability_leg.sh conv_stab_canon_s42 1 800 200 42
python research/convergence/analyze_settle.py --leg evidence/legs/conv_stab_canon_s42 \
  --extra-result evidence/legs/conv_stab_canon_s42/eval/eval_result_stab_curve.json
# extension leg (1600) + extension verdict ({800,1600} clause)
bash research/convergence/scripts/vconv_stability_leg.sh conv_stab1600 1 1600 400 42
python research/convergence/analyze_settle.py --leg evidence/legs/conv_stab1600 --extension
# confirm wave at L* (canonical x4 + grid2 x4 + pilots; retry driver)
VCONV_EVAL_CKPTS=ckpt_final.pt VCONV_KEEP_CKPT_FINAL=1 \
  bash research/convergence/scripts/vconv_confirm_driver.sh 1600 0,1,2 pilot 8
# systems series + capacity probe
bash research/convergence/scripts/vconv_harness_series.sh series1
bash research/convergence/scripts/vconv_capacity_probe.sh
# finalizers (fetch + frozen gates / systems table)
bash research/convergence/scripts/finalize_quality.sh
bash research/convergence/scripts/finalize_systems.sh
```

No merge to main; unverified features stay default-off.

## Outcome

**HOLD for the optimization claim; systems/capacity/applicability results stand.**
The pre-registered non-inferiority gate fails at the affordable local budget
(L*=1600, explicit "not settled" disclosure from the frozen settle rule). The
regrouping machinery stays default-off; the benchmark base, the identity-bypass
fix, the eval accounting, the applicability boundary map and the A100 matrix
are ready for the full round. Historical verdicts (d1-006/d1-007/d4xd1) are
preserved unchanged in `research/convergence/HISTORY.md`.
