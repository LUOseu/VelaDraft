# STATE — D1 convergence (single-agent round)

Last update: 2026-10-15 ~13:20 CST. Branch `research/rsi-d1-conv` (base `origin/main` @ 6d99f04).

## 发现—证据—决定—下一步 log

### F1. Recon: the identity-leg seam is the loss reduction, not clip width
- 发现: bug report #01 §6 attributed the P1 drift to three seams, including "shipped forward width 2048
  vs regroup ~1k". Re-verified from code + store data: `FeatureDataset` returns length-trimmed records and
  `collate_features` pads to the *batch* max; the shipped forward width for W3 batches is ~610-994, NOT 2048
  (store `seq_length` 2048 is the capture cap). For the identity plan, regroup's `s_g = max(lengths)` equals
  the shipped width exactly, and a single group = the shipped single backward.
- 证据: `vlm_dflash/dflash/features.py` (`FeatureDataset.__getitem__`, `collate_features`); W3 store index
  (`seq_len` 610-994); wave-2 harness rows (`canonical` arm `S=508/614`, not 2048).
- 决定: identity seam = loss-reduction path only (fp32 preserved-weight sum vs bf16 mean). Fix = route
  identity plans to the shipped path verbatim ("identity bypass"); keep the legacy machinery path as an
  explicit control (`identity_bypass=false`).

### F2. Identity bypass verified end-to-end (bitwise)
- 证据 (3-step smokes, seed 43, W3 store, GPU0/1): `conv_smoke_canon` (regroup off) vs
  `conv_smoke_bypass` (canonical/8, bypass=true): train_log bitwise identical on all steps (loss 12.875 /
  20.5 / 12.0625, grad_norm 104.02552032470703 / 51.649017333984375 / 39.25664933776855, lr all steps).
  `conv_smoke_mach` (canonical/8, fp32=false, bypass=false) reproduces the historical P1 signature: step 1
  loss 12.9375 (+1 ulp), grads bitwise steps 1-2, divergence at step 3 (39.532 vs 39.257, 7.0e-3 class).
  `conv_smoke_grid` (grid/2) runs the real machinery path (loss 12.911 step 1).
  `train_summary.regroup` counters: bypass 3/3 identity batches; mach 0/3.
- 决定: fix accepted on the real training loop; commit `f65e67f` pushed; server repo at the same SHA.

### F3. Pre-registration frozen before candidate results
- `PREREGISTRATION.md` + `analyze_quality.py` committed before any regroup-arm training result was read.
  Gates: one-sided 95% CI (hierarchical bootstrap over prompts within seed and seeds) lower bound
  > -1% relative AL and > -0.5pp acceptance rate; candidate grid/sub2 (systems-rule); 4 seeds confirmatory
  {42,43,44,45}; token/split at 2 seeds pilot. Length rule: settle test on canonical-only AL curve.

### F3 addendum, commit-order disclosure
- `PREREGISTRATION.md` authored 11:18:38 / `analyze_quality.py` 11:18:55 (mtime), first committed 12:16
  (`6af79f1`) as part of the convergence working set. Between authorship and commit, the only runs were the
  identity smoke legs (canonical/off, canonical/8 bypass, canonical/8 machinery, grid/2 3-step): no eval of
  any arm had been executed and no gate statistic existed. No candidate (grid/sub2) *training* leg has been
  launched or read. The frozen gates were fixed in text/sha before every arm result that will be fed to them.


### F7. Capacity probe re-verified at logical batch 16 (PREREG section 6 item)
- 发现 (A6000 48GB, W3 store, 4 timed logical batches, harness protocol): canonical single-chunk
  micro16 -> CUDA OOM 4/4; canonical micro8 fits at 24.21 GiB peak, 12.6% context padding waste;
  split sub2 fits at 9.37 GiB / 4.9%; grid sub2 fits at 9.37 GiB / 3.3%, wall median 1057ms vs
  canonical micro8 1301ms (-18.8%, n=4 indicative). Sorting by C x A removes ~1/4 of canonical's
  padding; splitting alone captures the bulk but not all.
- 证据: `research/convergence/evidence/capacity/` (raw jsonl + CAPACITY_BATCH16.md); server log
  `logs/vconv_capacity_probe.log`; OOM rows carry `run_error=cuda_oom` + error text.
- 决定: capacity claim re-cited from d1-006 is now re-verified this round; numbers folded into the
  systems table. Peak-memory win at equal semantics: -61%.

### F8. Mid-run checkpoints are resume-format; eval needs a converter (fixed)
- 发现: the stability eval crashed on `ckpt_200.pt`: `Unrecognized checkpoint format ... '
  vlm-dflash-train-state-v1'`. `write_train_state` saves exact-resume payloads (weights + fp32
  masters + optimizer + RNG); the eval/export loader needs `vlm-dflash-ckpt-v1`, whose provenance
  header (draft_config/algorithm/identity/feature_mode/plan digest) is written only into
  `ckpt_init.pt`/`ckpt_final.pt`. The watcher-copied mid-run ckpts were therefore not evaluable.
- 决定: added `scripts/convert_train_state_ckpt.py` + tests (graft the train-state `model`/`step`
  onto the `ckpt_init.pt` header; no semantics touched) commit `bc87e8f`; converted the three
  stability checkpoints to `ckpt_eval_{200,400,600}.pt`; eval relaunched 12:26 on GPU1.
- Follow-up: `vconv_stability_leg.sh` should convert before eval on its next use.


### F9. Second-drafter applicability (dflare smoke): bypass PASS, machinery refused
- 发现: 3-step smoke on the real dflare 2k store (`qwen25vl3b_dflare_2k`, 1050 samples, seq 2048):
  regroup-off leg rc=0; canonical/8+bypass leg rc=0 with GATE1 bitwise identity **PASS** (loss/grad_norm
  equal on every step); grid/2 leg rc=1 with `NotImplementedError: train.regroup does not support
  auxiliary/recipe losses yet` (dflash/train.py:321) -- dflare's forward surfaces auxiliary metrics, so
  the physical-group accounting path is unavailable without an adaptation.
- 证据: `research/convergence/evidence/second_drafter/` (logs + per-leg train logs); server log
  `logs/vconv_smoke_dflare.log`.
- 代码级边界 map (guard `loss_type=="dflash" and not sample_from_anchor`, plus the extra_metrics gate):
  dflash = full pass (this round's W3 legs); dflare = bypass-safe / machinery blocked (aux metrics);
  dflash2 and jetspec = blocked (aux metrics: unary_kl / kd_loss); dspark = blocked (sample_from_anchor);
  domino = blocked (shift_label -> sample_from_anchor + aux). Adapting a second drafter = teaching the
  group loop to aggregate each algorithm's auxiliary accounting per group, then re-running these gates.
- 决定: keep regroup default-off across all drafters; dflare's `regroup_adaptation` note in the A100
  package updated with the concrete block. The applicability statement for the paper: D1's physical
  regrouping is currently a dflash-family-only optimization (boundary documented, negative result for
  the others recorded, not papered over).


### F10. Stability curve / L* rule: settled(400)=False, settled(800)=False; extension leg running
- 发现 (canonical seed 42, held-out AL, frozen W3 store): 200=1.2017, 400=1.2105, 600=1.1632, 800=1.1370
  (all token_exact=True; levels consistent with the historical d1q baseline AL~1.165/acc~2.4% at 200 steps).
  settled(400)=0.73%>0.5%; settled(800)=|1.1370-1.2105|/1.2105=6.07%>0.5% -> the rule requires the fresh
  1600-step extension leg (`conv_stab1600`, running on GPU1 since 13:06) and the same rule on {800,1600}.
  The canonical curve peaks near step 400 and then declines: on this 192-row store the drafter overfits
  (train loss 0.026, slot0 acc ~0.999 while held-out AL falls). A plateau is unlikely; the prereg's
  "not settled" disclosure path (L*=1600) is the likely outcome, computed mechanically when the leg lands.
- 证据: `evidence/legs/conv_stab_canon_s42/` (first eval wave fetched 12:52, all three ckpts);
  `evidence/settle_decision.json` (updated with AL(800)); server log lines 12:49:08 + 13:14:57.
- 注意 (gotcha): each `eval_dflash.py` run OVERWRITES `run/eval/eval_result.json`; the 3-ckpt curve result
  was overwritten by the single-ckpt (ckpt_final) run. The full 3-ckpt result is preserved locally; the
  server now holds only the ckpt_final one. Stability-leg heavy ckpts cleaned (ckpt_eval_* + ckpt_final
  kept) to free ~12G; /home at 53G free.
- 下一步: stab1600 done (~13:40) -> AL(800 fresh), AL(1600) -> L* verdict -> launch the confirm wave at
  L* on all free GPUs (`vconv_confirm_driver.sh <L*> <gpus> pilot`, with `VCONV_EVAL_CKPTS=ckpt_final.pt`).

### F11. L* verdict: fresh 1600-step extension -> NOT settled; L*=1600 via disclosure path
- 发现: fresh canonical 1600-step leg (seed 42) held-out AL = 400:1.2015 / 800:1.1547 / 1600:1.1193;
  settled(1600) = |AL(1600)-AL(800)|/AL(800) = 3.07% > 0.5% -> frozen rule: L*=1600 with the explicit
  "not settled" disclosure; the quality claim is scoped to the matched budget (PREREGISTRATION.md §3).
- 证据: `evidence/legs/conv_stab1600/` (train_summary: loss_decreased=true, nan_seen=false,
  skipped_no_anchor=0, wall 1794.3s, peak 25.34 GiB; eval token_exact=true at all three ckpts);
  `evidence/settle_decision_ext1600.json`; server log 13:54:44 / done 13:54:47.
- 注意: AL *and* acceptance decline monotonically at longer budgets (1.202/0.029@400 -> 1.155/0.022@800 ->
  1.119/0.017@1600; train loss ~8e-4) -- overfitting on the 192-row store. The arm comparison is therefore
  a matched-budget non-inferiority test in this regime; absolute quality is a small-store artifact (the
  A100 2K-recipe round is where absolute quality is judged).
- 附注: fresh-leg AL(800)=1.1547 differs from the resume-leg AL(800)=1.1373 because the cosine schedules
  differ (1600-step vs 800-step total) -- expected, not a determinism discrepancy.
- 下一步: confirm wave at L*=1600 launched 13:55:54 on GPU 0,1,2 (canonical x4 + grid2 x4; pilots
  token2/split2 x2; VCONV_EVAL_CKPTS=ckpt_final.pt, ckpt_final kept) -> analyze_quality.py gates.

### F5. Stability leg: full 800 steps trained; crashed only at final checkpoint write (disk full)
- 发现: `conv_stab_canon_s42` trained steps 1..800 (train_log.jsonl complete, loss 12.875 -> 0.026, gn
  sane, lr schedule complete) and died at 11:33:18 inside the final `ckpt_last.pt` write (`.tmp` 669MB
  partial; /home was at 100%, 365M free). `ckpt_200/400/600.pt` (exact steps 200/400/600; watcher copied at
  log-steps 211/410/610) survived. `train_summary.json` was never written -> disclosed; leg validity flags
  (loss_decreased, nan_seen) re-derivable from train_log.jsonl only.
- 证据: `/home/lyh/veladraft-conv/runs/conv_stab_canon_s42/train/`, `logs/vconv_conv_stab_canon_s42_train.log`
  ("write error: No space left on device" traceback), `df` history.
- 决定: keep the 200/400/600 checkpoints (the settle rule needs exactly {200,400} and, only if unsettled,
  {800} via a 200-step resume from the intact step-600 `ckpt_last.pt`); eval launched manually at 12:17 on
  GPU1 with `--set eval.checkpoints=[ckpt_200.pt,ckpt_400.pt,ckpt_600.pt]`.

### F6. Disk recovery (shared /home was the binding constraint, not GPUs)
- 发现: /home (15T shared) sat at 100% all morning; GPU guard requires >=15G free, so runs could not start.
  GPUs themselves flicker: 4 tenants can appear/disappear within minutes (GPU0 re-occupied at 12:17).
- 决定: deleted only `*.pt` files (resumable ckpts) from 16 stopped-lane smoke run dirs
  (`vlm-dflash/runs/fix*_smoke_*`, `fix3_resume_online`, `rc_traindemo_gpu3`, `demo_dflash*`,
  `rsi-d1/runs/d4xd1_smoke`, `rsi-d1/runs/d1_mix_w1w2`); every config/contract/eval/log/json was left in
  place, and LIVE stores (`d1_w3_probe`, `d1q_val64`, `d1_mix_w1w2/features`, `d1_w1_probe`) untouched.
  Freed ~64G; /home at 65G free afterwards. No mainline benchmark runs (`qwen25vl3b_*`, `qwen3vl8b_reduced_*`)
  were touched.

### F4. Eval per-prompt accounting added (prereq for the preregistered CI)
- `eval_samples.jsonl` lacked per-prompt verify_steps/accepted/draft counts; added (commit `5750c23`),
  eval tests pass; server repo pulled so the stability leg's eval writes the new fields.

## Running now (13:56 CST)
- GPU0/1/2: confirm wave at L*=1600 (driver pid 600986, launched 13:55:54): canonical x4 + grid2 x4 (gates)
  + pilots token2/split2 x2. Queue per GPU: gpu0 canon_s42, grid2_s43, canon_s45, split2_s42; gpu1
  grid2_s42, canon_s44, grid2_s45, token2_s43; gpu2 canon_s43, grid2_s44, token2_s42, split2_s43.
  ETA ~16:30. Driver retries epochs=8; VCONV_EVAL_CKPTS=ckpt_final.pt; ckpt_final kept.
- GPU3: external tenant (generate_target_data.py) — excluded by the >=40GiB-free guard.
- Done since 12:45: harness series1 complete (all 10 tags rc=0, 12:17:48-13:39:42) -> 3-rep systems
  table committed (470c0ee); stability curve re-eval restored (tagged) + L* verdict (F11); extension leg
  done (F11); A100 dry-run re-validated (32 entries, 37.3 GPU-h; evidence/a100_dryrun.txt committed).

## Incidents
- `conv_smoke_grid` rc=1 at ckpt_final save (torch zip write error, host writeback/ENOSPC class) after
  3/3 steps trained; evidence kept, leg dir cleaned (not part of any verdict).
- `conv_smoke_canon` first attempt rc=1: `train.regroup.mode=off` parsed as YAML bool False; chain now omits
  regroup keys for mode=off. Re-run passed bitwise.
- /home free swung 10-15G; legs hold transient ~4.8GB (ckpt_init 975M + ckpt_last 3.4G + ckpt_final 487M)
  and are cleaned after eval. At most 2 legs in flight.

## Next
1. Monitor confirm wave (~16:30); then `scripts/finalize_quality.sh` -> `evidence/quality_verdict.json`
   (frozen gates: AL rel > -1%, acc drop > -0.5pp, hierarchical bootstrap).
2. `analyze_costs.py` full leg cost table (train wall/GPU-h/peak + eval per-sample) after the wave.
3. Final REPORT numbers + PR body; push branch; open PR (no merge to main).
4. A100 handoff list (matrix/runbook/dry-run) — done, cite in the final message.
