# PREREGISTRATION — D1 convergence quality confirmation

Status: **FROZEN before any candidate (regroup-arm) training result was read.**
Date: 2026-10-15 (Asia/Shanghai). Role: single convergence agent (VelaDraft).
Supersedes the D1 lane's d1-006/d1-007 gates *for this round's claim only*; the
historical verdicts stand unchanged (d1-006 FAIL vs the frozen 2-seed band,
d1-007 M1 stopped at P1; see `HISTORY.md`).

## 1. Claim under test

At the frozen budget L* (see §3), training with the C×A dual-axis physical
regrouping candidate (`train.regroup.mode=grid, sub_batch=2`, `fp32_weights=true`,
`identity_bypass=true`) is **non-inferior** to the shipped canonical path on the
held-out drafter quality (AL and draft-token acceptance rate) within the
registered tolerance in §5, on the frozen W3 store / val64 held-out split.

Non-inferiority only: this can never claim "lossless". All observed deltas are
reported with uncertainty; the historical out-of-tight-band FAIL is retained.

## 2. Protocol (frozen)

- Store: `/home/lyh/rsi-d1/runs/d1_w3_probe` (192 rows, real multi-turn; hardlinked per leg).
- Held-out: `d1q_val64` = 64 prompts disjoint from the 192 training rows; never trained on.
- Train: `scripts/train_dflash.py`, config `e2e_w3_conv.json` (slice 192, micro_batch 8,
  grad-accum 1, lr 6e-4, cosine + 4% warmup, bf16, flex_attention), `--skip-sanity`,
  `feature_mode=offline`, `log_every_steps=1`.
- Eval: `scripts/eval_dflash.py --num-prompts 64 --set eval.max_new_tokens=128`,
  greedy, sequential verification (token-exact path). Metrics per checkpoint:
  `mean_al` (AL = new_tokens / target_verification_steps, bonus-inclusive),
  `acceptance_rate` (accepted/drafted), `per_slot_acceptance`, `verify_steps`.
- Every leg records `eval_samples.jsonl` (per-prompt rows) — the primary analysis unit.
- Identical data order and anchor RNG stream across arms: same seed, same
  collate/order code, one anchor sampler call per logical batch; anchor tables
  are replayed (not resampled) inside the regroup path; `train.seed` fixes
  init + order + anchors. The parity oracle (`tests/test_regroup.py`) and the
  R4 removability audit back the same-supervision/equivalence claims.

## 3. Length L* — selection rule (canonical-only data)

Stability leg: canonical, seed 42, 800 steps (schedule total = 800), held-out
evaluated at steps {200, 400, 600, 800} via mid-run checkpoint copies.

- settled(L) := |AL(L) − AL(L/2)| / AL(L/2) ≤ 0.5% (canonical held-out AL).
- L* := 400 if settled(400); else 800 if settled(800); else 1600 (extend with a
  fresh canonical 1600-step leg, same rule on {800, 1600}; if still unsettled,
  L* = 1600 with an explicit "not settled" disclosure and the claim scoped to
  the matched budget).
- The rule is applied to canonical data only, before reading any regroup-arm result.

## 4. Arms and seeds (frozen)

| arm | train.regroup | seeds | role |
| --- | --- | --- | --- |
| canonical | off | 42 43 44 45 | reference |
| grid/sub2 | grid, sub_batch=2, fp32, bypass | 42 43 44 45 | **candidate** (confirmatory) |
| token/sub2 | token, sub_batch=2, fp32, bypass | 42 43 (pilot) | C-only control |
| split/sub2 | canonical, sub_batch=2, fp32, bypass | 42 43 (pilot) | plain-split control |

`fp32_weights=true` is the validated wave-2 reduction (the fp32 preserved-weight
path); `identity_bypass=true` is required for the fixed control semantics.
Candidate choice (grid/sub2) was fixed by the systems-only rule: best measured
step-wall gain among the regroup modes/configs that passed semantics gates
(W3 k=4 `grid` +18.8% / `cost` +19.0% vs k=2 +10.7%, wave-2; re-measured this
round). k=2 is *not* promoted post-hoc if k=4 fails the gate; any k=2 result is
reported as exploratory only.

## 5. Gates (frozen)

Per-prompt paired deltas (same prompts, same seed): dAL = AL_arm − AL_canon,
dacc = acc_arm − acc_canon. Per-seed: mean over the 64 prompts; relative drop
R_s = mean_p(dAL) / mean_p(AL_canon).

- **AL gate**: one-sided 95% CI lower bound of mean_s(R_s) > −1.0%.
- **Acceptance gate**: one-sided 95% CI lower bound of mean_s(dacc) > −0.005
  (0.5 percentage points).
- CI: hierarchical bootstrap, B=10,000 — resample prompts within each seed
  (paired) and seeds across seeds; percentile one-sided bound. Secondary
  (reported, not gating): t-based CI across seeds.
- Both gates must hold on the confirmatory set (canonical + grid/sub2, 4 seeds)
  for verdict **PASS (non-inferior within registered tolerance)**. Else **FAIL**.
- Leg validity (else the leg is undetermined, not silently dropped):
  `loss_decreased=true`, `nan_seen=false`, `skipped_no_anchor_batches=0`,
  finite final loss, `token_exact=true` for the evaluated checkpoint.
- No post-hoc threshold changes. If evidence is insufficient (wide CI, invalid
  leg, budget exhaustion) the verdict is **HOLD**.

## 6. Systems gates (reported, not the confirmatory statistic)

- Paired per-step wall and peak memory: harness A/B (`r3_ab_step.py`, arms
  canonical/split/token/cost/grid; W3 + MIX + W0 low-waste control; 3 independent
  rep rounds; alternating arm order; warmup batches untimed; peak via
  `torch.cuda.max_memory_allocated`).
- Engineering target (task brief): same-quality peak memory −25% OR drafter
  training wall −10%. Targets not met are reported as such; no multiplication of
  per-stage speedups; full-pipeline wall and GPU-hours reported separately from
  the train stage.
- Capacity case: a configuration that OOMs under canonical and runs under regroup
  (d1-006 evidence re-cited; optionally re-verified).

## 7. What would falsify the candidate

- Either gate fails → FAIL (negative result recorded; feature stays default-off).
- Supervision-set divergence, effective-batch change, or data-order divergence
  discovered in any leg → the affected comparison is invalid (HOLD).
- Token-exact failure on the eval path → leg invalid.

## 8. Evidence retention

Raw per-leg: `train_log.jsonl`, `train_summary.json`, `eval_result.json`,
`eval_samples.jsonl`, `config.json`, `provenance.json`, guard-annotated launch
logs. Aggregates and the analysis script output are committed under
`research/convergence/evidence/`. Exact code SHA recorded per leg (`repo_head`
in the launch log + `git_sha` fields).
