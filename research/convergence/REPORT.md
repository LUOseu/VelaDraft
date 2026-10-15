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

Confirmatory set (canonical ×4 vs grid2 ×4 at L\*=1600, val64, paired; all 8
legs pass the prereg validity list — train summaries valid, token_exact=True,
64 prompts/leg; frozen gates: AL rel > −1%, dacc > −0.5pp, hierarchical
bootstrap B=10,000, seed 20261015). The AL columns are the per-prompt means the
prereg registers (`R_s = mean_p(dAL) / mean_p(AL_canon)`); the run-level pooled
means are kept in `quality_verdict.json` as `*_aggregate`:

| seed | canonical AL (pp mean) | grid2 AL (pp mean) | rel ΔAL | canonical acc | grid2 acc | Δacc |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 1.128909 | 1.128658 | −0.022% | 0.01715 | 0.01729 | +0.00031 |
| 43 | 1.151123 | 1.158462 | +0.638% | 0.02054 | 0.02127 | +0.00097 |
| 44 | 1.195482 | 1.159236 | **−3.032%** | 0.02668 | 0.02136 | −0.00544 |
| 45 | 1.115507 | 1.113776 | −0.155% | 0.01533 | 0.01446 | −0.00035 |

**Verdict: FAIL.** Mean relΔAL = −0.643%, one-sided 95% bootstrap lower bound
= **−1.980%** (gate > −1.0%) → AL gate fails; dacc = −0.00113, lower bound
−0.00352 → acc gate passes; both must hold → FAIL (frozen `PREREGISTRATION.md` §5;
`evidence/quality_verdict.json`). Secondary t-based CI across seeds:
mean −0.643%, sd 1.630%, CI ≈ [−3.24%, +1.95%] — same story.

**Tool-review re-analysis (2026-10-15, PR #9 review).** The first table computed
`rel` from the run-level pooled `mean_al` (sum(new_tokens)/sum(verify_steps))
instead of the prereg's per-prompt statistic. `analyze_quality.py` now
implements the registered functional exactly and gates every leg on the §5
validity list (invalid/incomplete evidence → HOLD with reasons, never a silent
PASS/FAIL). Corrected numbers: mean −0.775% → **−0.643%**, lower bound
−2.000% → **−1.980%**; the reviewer's independent recomputation reproduced
both. The verdict stays **FAIL** and the optimization stays default-off.

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
| arm | mean relΔAL | bootstrap lo | mean Δacc | lo | mechanical verdict (2 seeds) |
| --- | --- | --- | --- | --- | --- |
| token2 (C-only) | +0.015% | −1.053% | +0.00013 | −0.00186 | FAIL (2-seed, exploratory) |
| split2 (plain split) | +0.355% | −0.345% | +0.00066 | −0.00042 | PASS (2-seed, exploratory) |

Reading across all four layouts: the pooled confirmatory mean (−0.64%) is not
significantly different from 0 (t ≈ −0.79, sd(seed rel) = 1.63%), and the 2-seed
pilots land within ±0.36% of canonical. The FAIL is a **certification** outcome
(lower bound south of the registered −1% margin), driven by seed-level endpoint
scatter — not an established systematic regression. Conversely, token2 does not
rescue the gate either, and per the frozen prereg §4 no layout is swapped in
post-hoc; these rows are reported as exploratory only.

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

- Tests: `PYTHONPATH=. python -m pytest tests/ -q --ignore=tests/gpu` → **541
  passed, 6 skipped, 0 failed** @ `533bfa8` (adds the PR#9-review regressions:
  runner config binding/benchmark naming/stage-done/summarize mapping, analyzer
  validity+registered statistic, converter provenance; later commits are
  docs/evidence only).
- Commands: stability leg, settle analyzer (default + `--extension`), confirm
  wave driver, harness series, capacity probe, finalizers (`finalize_quality.sh`,
  `finalize_systems.sh`, `analyze_costs.py`); exact invocations in STATE.md /
  PR body.
- SHAs: all training/eval legs ran at server repo pin `bc87e8f` (no code diffs in
  `vlm_dflash/`/`scripts/`/`tests/` through the branch tip; per-leg `repo_head`
  recorded in launch logs). Branch tip at close-out: see PR.
- Cost table: `evidence/costs_table.md` (14 legs: train wall/GPU-h, peak, eval
  per-sample breakdown, token_exact). Wave walls are contended wall-clock (3
  concurrent legs + tenant flicker); paired per-step numbers come from the
  harness (`evidence/systems_table.md`). Pooled wave train wall: canonical
  1798 s mean vs grid2 1728 s (−3.9%); peak 25.3 → 10.0 GiB (−60.5%).
- PR: https://github.com/LUOseu/VelaDraft/pull/9 (open, HOLD; no merge to main; regroup default-off everywhere).
- Review round (2026-10-15): all six PR-review findings fixed @ `533bfa8`
  (per-algorithm pinned configs, `prepare` data binding + preflight, per-workload
  benchmark reports, validity-gated analyzer with the registered statistic,
  converter provenance); end-to-end smoke (prepare → train → eval → 2 workloads
  → resume → summarize) under `evidence/a100_smoke/` (plumbing only; 2 steps,
  quality metrics not meaningful by design).
- A100: `a100/RUNBOOK.md` + `a100/run_matrix.py` (dry-run validated: 32 entries,
  37.3 GPU-h est., 0 errors; controlled + native tracks; train/val/test
  isolation; regroup gates + adaptation smoke).

## 7. Outcome / next

- **Status: HOLD for the optimization claim.** Systems/capacity/applicability
  results are final and positive; the registered quality gate is FAIL; regroup
  ships default-off (unverified for quality at this budget).
- Natural next steps on A100 (see runbook): re-run the same confirmatory design
  at the 2K-step registered recipe where the held-out curve is expected to be
  settled (the L\* "not settled" disclosure and the FAIL are the same fact: at
  1600 steps the regime moves ~3%/800 steps, below a ±1% margin); consider a
  pre-registered margin informed by a fresh settle curve, and keep the same
  paired multi-seed gates.
