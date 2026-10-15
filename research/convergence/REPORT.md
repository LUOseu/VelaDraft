# D1 convergence round — report (rolling)

Single-agent convergence round for direction D1: physical training regrouping by
visual context length C × valid anchor count A. Branch `research/rsi-d1-conv`
(base `origin/main` @ `6d99f04`). Status: **in progress — numbers filled as the
confirm wave completes.**

## 1. What was fixed first (round prerequisites)

| item | status | evidence |
| --- | --- | --- |
| Identity-leg seam isolated to the loss reduction (not forward width) | done | STATE F1; `vlm_dflash/dflash/train.py` `_regrouped_forward` |
| Identity bypass: regroup identity layouts re-run the shipped call verbatim | done, bitwise | commit `f65e67f`; 3-step W3 smokes (off vs canonical/8+bypass bitwise equal); legacy machinery control reproduces the historical ±1 ulp signature; dflare also bitwise PASS |
| Per-prompt acceptance accounting in `eval_samples.jsonl` | done | commit `5750c23` |
| Train-state → eval checkpoint converter (stability curves without re-training) | done, tested | commit `bc87e8f`; `tests/test_convert_train_state_ckpt.py` |

## 2. Systems closure (A6000 48GB, W3 store, harness, 3 reps × 100 batches)

Figures from `evidence/systems_table.md` (paired per-logical-batch medians vs canonical):

| layout (sub-batch) | wall speedup | peak alloc | context padding waste | loss rel (med) |
| --- | --- | --- | --- | --- |
| canonical | 1.000 | 25.05 GiB | 11.8% | — |
| split (plain equal-k split, unsorted) | 1.056 | 10.64 GiB | 5.5% | 0.22% |
| token (C-only sorted) | 1.108 | 10.65 GiB | 2.7% | 0.22% |
| dual (C + A) | 1.103 | 10.64 GiB | 2.7% | 0.22% |
| cost (fitted C + w·A) | 1.131 | 10.65 GiB | 3.6% | 0.22% |
| grid (2D quantiles) | 1.131 | 10.65 GiB | 3.6% | 0.22% |
| cp / ref (sanity) | 0.999 / 1.000 | 25.05 GiB | — | 0.22% |

<placeholder: final 3-rep aggregated numbers incl. w3_k2, mix_k4, w1_k4>

Decomposition: splitting alone ≈ +5.6%; sorting by C ≈ +5.2pp more; the anchor
axis adds ≈ +2.3pp on top of C (`token` → `dual/grid/cost`). At k=2 the gains
shrink (<placeholder>) — the value is concentrated where rows are highly
heterogeneous.

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

L* verdict: settled(400)=False (0.73%), settled(800)=False (6.07%) → fresh
1600-step extension leg (`conv_stab1600`) → same rule on {800, 1600}:
<placeholder>. Canonical curve (held-out AL, seed 42):
200=1.2017, 400=1.2105, 600=1.1632, 800=1.1370, 1600=<placeholder> — peaks near
step 400 then declines (overfit on the 192-row store; train loss 0.026 with
held-out AL falling). If unsettled at 1600 the prereg's disclosure path applies
(L*=1600, claim scoped to the matched budget).

<placeholder: confirm table — canonical ×4 vs grid2 ×4 at L*: per-seed AL/acc +
hierarchical bootstrap gates (-1% AL rel; -0.5pp acc) → PASS/FAIL/HOLD>

Pilot controls (token/sub2, split/sub2 ×2 seeds): <placeholder>.

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
