# D1 convergence round — report (rolling)

Single-agent convergence round for direction D1: physical training regrouping by
visual context length C × valid anchor count A. Branch `research/rsi-d1-conv`
(base `origin/main` @ `6d99f04`). Status: **closed.** Systems/capacity/applicability
tables below are final; the frozen confirmatory quality gate is a **FAIL**
(non-inferiority not established at the registered budget) — the negative result
is the registered outcome and the optimization stays default-off.

## 1. What was fixed first (round prerequisites)

| item | status | evidence |
| --- | --- | --- |
| Identity-leg seam isolated to the loss reduction (not forward width) | done | STATE F1; `vlm_dflash/dflash/train.py` `_regrouped_forward` |
| Identity bypass: regroup identity layouts re-run the shipped call verbatim | done, bitwise | commit `f65e67f`; 3-step W3 smokes (off vs canonical/8+bypass bitwise equal); legacy machinery control reproduces the historical ±1 ulp signature; dflare also bitwise PASS |
| Per-prompt acceptance accounting in `eval_samples.jsonl` | done | commit `5750c23` |
| Train-state → eval checkpoint converter (stability curves without re-training) | done, tested | commit `bc87e8f`; `tests/test_convert_train_state_ckpt.py` |

## 2. Systems closure (A6000 48GB, harness, 3 reps × 100 batches)

Final numbers from `evidence/systems_table.md` (paired per-logical-batch medians vs
canonical, 3 reps; raw rows `evidence/harness/`). Speedup vs canonical:

| arm | W3 k4 | W3 k2 | MIX k4 | W1 k4 |
| --- | --- | --- | --- | --- |
| canonical | 1.000 | 1.000 | 1.000 | 1.000 |
| split (plain equal split) | 1.061 | 1.026 | 1.082 | 0.867 |
| token (C-only sort) | 1.111 | 1.067 | 1.128 | 0.867 |
| dual (C + A sort) | 1.110 | 1.069 | 1.123 | 0.865 |
| cost (C + w·A) | 1.133 | 1.097 | 1.157 | 0.872 |
| grid (2D quantiles) | 1.131 | 1.097 | 1.156 | 0.873 |
| cp / ref (identity controls) | 1.000 / 0.999 | 0.999 / 1.000 | 0.995 / 0.994 | 0.976 / 0.979 |

Peak alloc 25.05 → 10.64 GiB (−57%) at W3 k4; 24.0 → 10.36 GiB on MIX;
25.05 → 15.60 GiB (−38%) at W3 k2. Context padding waste on W3 k4:
11.8% → 2.7% (`token`/`dual`) / 3.6% (`cost`/`grid`); MIX: 20.1% → 4.1% / 6.0%.
One-batch loss parity vs canonical (bf16 vs fp32-preserved reduction): med
≈ 0.21%, grad-norm max ≈ 0.8% — the machinery identity path costs ≈ 0%
(cp/ref ≈ 1.000, loss-identical).

Decomposition: plain sub-batching +2.6–8.2%; sorting by C adds +2.6–4.6pp
(MIX k4: 1.082 → 1.128); the A/cost axis adds **+2.2–3.0pp on heterogeneous
stores** (W3 k4 1.111 → 1.131; k2 1.067 → 1.097; MIX 1.128 → 1.156) but ≈ 0 on
low-memory-pressure W1. **No-gain boundary:** on W1 (short sequences, 10 GiB
canonical peak) every regroup arm is *slower* (0.865–0.873) — sub-batching
overhead is not repaid when padding/memory pressure is low.

## 3. Capacity (logical batch 16, single-step)

| layout | fits? | peak | padding waste | wall (med, 4 batches) |
| --- | --- | --- | --- | --- |
| canonical, 16-row chunk | **OOM 4/4** | — | — | — |
| canonical, 2×8 chunks | yes | 24.21 GiB | 12.6% | 1301 ms |
| split sub2 (8×2) | yes | 9.37 GiB | 4.9% | 1514 ms |
| grid sub2 (8×2 sorted) | yes | 9.37 GiB | 3.3% | 1057 ms |

Peak −61% at equal semantics; batch 16 unlockable only with sub-batching.
(Details + raw rows: `evidence/capacity/`.)

## 4. Quality (frozen prereg; confirm wave)

L* verdict (frozen rule, canonical-only data): settled(400)=False (0.73%),
settled(800)=False (6.04%) → fresh 1600-step extension leg (`conv_stab1600`);
same rule on {800, 1600}: **3.07% → NOT settled**. **L* = 1600** under the
prereg's explicit not-settled disclosure; the quality claim is scoped to the
matched budget. Canonical curves (held-out AL, seed 42, all token_exact):
800-step schedule 200=1.2017, 400=1.2105, 600=1.1632, 800=1.1373; 1600-step
schedule 400=1.2015, 800=1.1547, 1600=1.1193. AL *and* acceptance decline at
longer budgets (acc 0.029 → 0.022 → 0.017; train loss ≈ 8e-4) — overfitting on
the 192-row store. The arm comparison is a matched-budget non-inferiority test
in this regime; absolute quality is a small-store artifact (the A100 2K-recipe
round is where absolute quality is judged). Artifacts: `evidence/settle_decision.json`,
`evidence/settle_decision_ext1600.json`, `evidence/legs/conv_stab1600/`.

Confirmatory set (canonical ×4 vs grid2 ×4 at L\*=1600, val64, paired; all legs
token_exact=True; frozen gates: AL rel > −1%, dacc > −0.5pp, hierarchical
bootstrap B=10,000, seed 20261015):

| seed | canonical AL | grid2 AL | rel ΔAL | canonical acc | grid2 acc | Δacc |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 1.119269 | 1.119437 | +0.015% | 0.01715 | 0.01729 | +0.00031 |
| 43 | 1.141853 | 1.146761 | +0.430% | 0.02054 | 0.02127 | +0.00097 |
| 44 | 1.183648 | 1.147642 | **−3.042%** | 0.02668 | 0.02136 | −0.00544 |
| 45 | 1.106339 | 1.100796 | −0.501% | 0.01533 | 0.01446 | −0.00035 |

**Verdict: FAIL.** Mean relΔAL = −0.775%, one-sided 95% bootstrap lower bound
= **−2.000%** (gate > −1.0%) → AL gate fails; dacc = −0.00113, lower bound
−0.00352 → acc gate passes; both must hold → FAIL (frozen `PREREGISTRATION.md` §5;
`evidence/quality_verdict.json`). Secondary t-based CI across seeds:
mean −0.775%, sd 1.559%, CI ≈ [−2.61%, +1.06%] — same story.

**Attribution (RSI checks, no post-hoc loosening).** Step-1 loss seams of the
pairs are 4.4e-4 / 2.8e-3 / 1.9e-3 (same order for every seed) → same
supervision, data order and anchors; the residual is the fp32-preserved vs
bf16-mean reduction seam, which then amplifies chaotically over 1600 steps of
bf16 SGD (as the prereg allows; cf. the bitwise cross-GPU control, STATE F12).
Seed 44 shows grid2 *training* slightly better (last10 1.92e-3 vs 1.98e-3) but
held-out AL 3% worse — an endpoint divergence in the still-fast-moving
(unsettled) held-out regime, not a semantic change and not a bug. Context: the
L\* extension curve itself moves ≈3% per 800 steps, so a ±1% non-inferiority
claim at this budget is below the regime's trajectory sensitivity.

Pilot controls (token/sub2, split/sub2 ×2 seeds; exploratory, non-gating):
<placeholder — filled by the final `finalize_quality.sh` run>.

## 5. Applicability boundary (second drafter)

Runtime smokes on real stores + code-level guards:

| drafter | regroup machinery | why |
| --- | --- | --- |
| dflash | **works** (verified) | fixed-anchor dflash objective; no aux metrics |
| dflare | bypass-safe, **machinery refused** | aux metrics from fusion diagnostics |
| dflash2 | refused | aux metrics (unary_kl) |
| jetspec | refused | aux metrics (kd_loss) |
| dspark | refused (guard) | `sample_from_anchor=True` |
| domino | refused (guard) | shift_label → `sample_from_anchor=True` |
| dfly | not tested | gate as dflash-family; no regroup code path claimed |

Negative results are recorded as boundaries, not as failures of the mainline
benchmark: D1 is a dflash-family optimization; adapting another drafter requires
group-aware auxiliary accounting, with the smoke gates in
`a100/regroup_adaptation_smoke.sh` as the acceptance test.

## 6. Repro + artifacts

- Tests: `PYTHONPATH=. python -m pytest tests/ -q --ignore=tests/gpu` → **523
  passed, 6 skipped, 0 failed** @ `b74532d` (local).
- Commands: stability leg, settle analyzer, confirm wave driver, harness series,
  capacity probe (see `scripts/`; exact invocations in STATE.md).
- SHAs: branch tip + per-leg `repo_head` in server launch logs.
- PR: <placeholder link> (no merge to main; regroup default-off).
- A100: `a100/RUNBOOK.md` + `a100/run_matrix.py` (dry-run validated: 32 entries,
  ~37 GPU-h; two tracks; isolation checks; regroup gates).

## 7. Open items / next

- Confirm wave completion + verdict JSON (`evidence/quality_verdict.json`).
- 3-rep harness aggregation (`analyze_systems.py`).
- Full-pipeline wall/GPU-hour accounting from per-leg logs (train + eval split).
- Pilot token/split interpretation (+2 seeds if the boundary is interesting).
