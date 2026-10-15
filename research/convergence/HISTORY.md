# HISTORY — frozen negative verdicts from the D1 lane (kept as-is)

This round does **not** re-litigate these verdicts, and none of the numbers below
are recomputed or relaxed here. The convergence preregistration
(`PREREGISTRATION.md`) tests a different, narrower claim — matched-budget
non-inferiority at L\* with hierarchical confidence intervals — and says so
explicitly. All raw evidence stays in the original lanes (paths per entry).

## d1-006 (L-W3-D1Q) — held-out quality vs the frozen canonical band: **FAIL** (frozen 2026-10-15 05:47)

- Frozen band (canonical 42/43 `ckpt_final` differences; 200 steps, W3 store, val64):
  `band_AL = 1.8152127954278452e-3`, `band_acc = 2.79171630289541e-4`.
- Canonical references: s42 AL=1.1653666 / acc=0.0239032; s43 AL=1.1635514 / acc=0.0236240.
- Candidate (grid/2, fp32 preserved weights): s42 dAL=3.2633e-3 (**1.80×** band) /
  dacc=4.7229e-4 (**1.69×**); s43 dAL=8.8134e-3 (**4.86×**) / dacc=1.2925e-3 (**4.63×**)
  → out of band on both seeds and both metrics (same direction) → FAIL; long training stays gated off.
- Receipts: `veladraft-scale/research/rsi/_coord/evidence/c0_d1q_receipt_verdict_20261015.md`,
  `..._harvest_receipt_20261015.md`; lane raw: `vlm-dflash-d1/research/rsi/d1/evidence/d1q_band_summary.json`.

## d1-007 (L-W3-D1R7) — mechanism attribution: **M1 STOPPED at P1** (H4 control), bug report #01

- Stop rule was frozen before running: P1 out-of-band → stop M1, file a bug report; no GPU work after the stop.
- P1 = `canonical/8/false` @ seed 43 (only knob flipped vs canonical = regroup machinery ON; identity
  layout, same dtype): eval AL=1.1588582 / acc=0.0229341 → |ΔAL|=4.6932e-3 = **2.59×** band;
  |Δacc|=6.8989e-4 = **2.47×** band → out of band, negative. Consumed 0.37/2.0 GPU-h.
- Formal A/B/C disposition (lane preference C) remained pending at close-out.
- Evidence: `vlm-dflash-d1/research/rsi/d1/proposals/d1-007-bugreport-01.md` (incl. 9-file sha256 table);
  lease receipt `veladraft-scale/research/rsi/_coord/proposals/d1-007-c0.md`.
- Relation to this round: the convergence identity fix (`identity_bypass`, commit `f65e67f`) makes the
  identity-layout machinery seam impossible by construction for dflash (bitwise smokes, STATE F2), which
  is why every convergence leg runs with `identity_bypass=true`. The historical verdict stands for the
  feature as configured then; the P1 drift mode is structurally removed for identity layouts only.

## d4xd1-003 / 004 / 005 — D4×D1 combination & isolation: **correctness blockers stand**

- d4xd1-003 (2×2 combination, FINAL 04:10): F1 gate (steps 2..101, max loss_rel ≤ 7.5e-3) —
  W3 `bucket_on` **FAIL: max 1.2475e-2 @step 11** (4 steps over; median 1.854e-3); MIX all pass
  (6.1262e-3 @37, 0 over). F3 peak ratio PASS (0.5324 W3 / 0.5408 MIX); interaction bit-level 0.
  Mapping executed: W3 out-of-band → the combined numerical-compatibility claim does **not** hold
  (correctness blocker; combination temporarily banned). Actual ≈0.76 GPU-h.
  Receipt: `veladraft-scale/research/rsi/_coord/proposals/d4xd1-003-c0.md`.
- d4xd1-005 (isolation arm L-W3-D4xD1B, FINAL 04:24): F1B (`fp32_weights=false`) **FAIL: max
  loss_rel 2.2727e-2 @step 11, 17/100 steps over gate, median 4.32e-3** → blocker **deepened**;
  “不得用于事后放宽任何门”. Receipt: `.../proposals/d4xd1-005-c0.md`.
- These constrain scope, not this round's claim: the convergence rounds use the validated wave-2
  semantics (`fp32_weights=true` + `identity_bypass=true`, matched supervisions) on the offline W3
  store only.
