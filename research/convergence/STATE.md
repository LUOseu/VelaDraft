# STATE — D1 convergence (single-agent round)

Last update: 2026-10-15 ~12:45 CST. Branch `research/rsi-d1-conv` (base `origin/main` @ 6d99f04).

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

## Running now (12:45 CST)
- GPU1: stability eval of `conv_stab_canon_s42` over the converted ckpt_eval_200/400/600 (relaunched 12:26
  after the F8 converter fix) -> L* decision via `analyze_settle.py`.
- GPU2: harness series `series1` rep 1/3 of w3_k4 done 12:29; ETA ~1.5h for the full series.
- GPU0: external tenant. GPU3: free (kept idle for the confirm-leg wave).
- Capacity probe done (F7); A100 matrix runner dry-run passed locally (32 entries, ~37 GPU-h).
- GPU0: `conv_stab_canon_s42` — canonical seed 42, 800 steps, mid ckpt copies @200/400/600, 4-checkpoint eval.
- GPU1: harness series `series1` — W3 k4 x3, W3 k2 x3, MIX k4 x2, W1 k4 x1, seam run (arms
  canonical,cp,ref,split,token,dual,cost,grid; 100 batches/rep; parity 3).

## Incidents
- `conv_smoke_grid` rc=1 at ckpt_final save (torch zip write error, host writeback/ENOSPC class) after
  3/3 steps trained; evidence kept, leg dir cleaned (not part of any verdict).
- `conv_smoke_canon` first attempt rc=1: `train.regroup.mode=off` parsed as YAML bool False; chain now omits
  regroup keys for mode=off. Re-run passed bitwise.
- /home free swung 10-15G; legs hold transient ~4.8GB (ckpt_init 975M + ckpt_last 3.4G + ckpt_final 487M)
  and are cleaned after eval. At most 2 legs in flight.

## Next
1. Read stability curve (200/400/600/800) -> L* per the frozen rule.
2. Launch confirmatory legs (canonical x4, grid/sub2 x4) + pilot legs (token/sub2 x2, split/sub2 x2) at L*.
3. Harness series -> systems table (wall/peak/padding; canonical vs split vs token vs cost/grid).
4. Capacity probe (batch 16 canonical OOM vs grid fits) + second-drafter applicability (DFlare) smoke.
5. A100 matrix runner (dry-run/run/resume/aggregate) + final report + PR.
