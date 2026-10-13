# VelaDraft plan - resource ledger

## GPU budget

- First-round cap: **8.0 GPU-hours** (plan default; no other budget given).
  1 GPU job at a time; single job <= 2 h wall; <= 2 fix-retries per failing
  debug task.
- Charged so far: **4.10 GPU-h** (R-20261014-01 0.044 + R-20261014-02 0.002
  + E1 R-20261014-03..05 0.29 + duplicate concurrent E1 set R-20261014-07
  0.26 + zoo re-download validation R-20261014-09 0.04 + E2 R-20261014-08
  0.55, two passes + E3 R-20261014-10 0.15 + E4 R-20261014-11 1.45 + WP4
  measurement R-20261014-12 0.21 + zoo v0.3.2 validation demo R-20261014-13
  0.02 + WP5 R-20261014-14..16 0.55 + WP6 R-20261014-17 0.53).
  Remaining: **3.90 GPU-h**. Shared file between the two concurrent
  sessions - re-read before editing.
- First 0.5 GPU-h is measurement budget (throughput / peak memory / storage);
  the two smoke runs consumed part of it.

| Allocation (draft) | GPU-h |
| --- | --- |
| E1 WP1 validation (R-20261014-03/04/05) | 0.25 used -> actual 0.29 + 0.26 (duplicate set) |
| WP4 batched-online + profiling | 1.0 budgeted -> 0.21 actual (R-20261014-12: A/B 3m13s + 2m41s, two profiling passes) |
| E2 WP2 multi-image smoke | 0.5 budgeted -> 0.55 actual (two passes; second pass overran the cap) |
| E3 WP3 DocVQA/ChartQA validation | 0.7 budgeted -> 0.15 actual |
| E4/WP7 DFly 2K recipe canonical (+eval) | 1.0 budgeted -> 1.45 actual (overfit sanity + storm-inflated main loop + 3-config eval; recorded) |
| WP6 non-Qwen DFlash smoke | 1.5 budgeted -> 0.53 actual (R-20261014-17: regen + capture + 32-step train + dual-checkpoint eval + margin probes) |
| WP5 runtime load test | 1.0 budgeted -> 0.55 actual (R-14 vLLM bridge check 0.11 inc. smoke + R-15 in-repo arms 0.36 + R-16 margin probe 0.08) |
| Reserve | 2.0 |

## Devices

- `a6000xuyang` (ssh `oab-remote`, 10.65.14.8): 4x RTX A6000 48 GB, idle at
  audit time; shared server - other users' jobs may appear.
- Local Mac M4 (16 GB): CPU tests/docs only.

## Disk (measured 2026-10-14)

| Path | Size | Free | Note |
| --- | --- | --- | --- |
| `a6000xuyang:/home` | 15 TB | **61 GB (100% used), drifting down** | shared; 130 GB free at 01:30 (audit), 58 GB at 02:52; E4's renewable `ckpt_last.pt` (3.7 GB) pruned 02:55 -> 61 GB. Host-wide writeback stalls hit the E2 eval ~02:10 and slowed E4's train mid-run. Keep net-new small. |
| `a6000xuyang:/` | 1.8 TB | 53 GB (97%) | system |
| Local Mac | - | check before large downloads | - |

### Drift-prone estimates per job kind

| Job kind | Net-new storage | Note |
| --- | --- | --- |
| Recipe training (existing store, warm) | ~5 GB (init 1.0 + last 3.7 + final 0.5) | `ckpt_last.pt` prunable after eval (renewable); prune this plan's own after use |
| New 2K capture with `store_final_hidden` | ~9-12 GB | reuse via symlink; do not duplicate per algorithm |
| New model download (WP6 candidate) | ~7-16 GB | check `/home` free first |
| Eval / export outputs | < 0.1 GB | fine |

### Memory calibration (recorded runs)

| Job | Config | Peak GPU |
| --- | --- | --- |
| Recipe (upstream_two_stage) train, sanity off | micro-batch 8, seq 2048 | 28.6 GB / 48 GB |
| Recipe train + overfit sanity (16 samples) | micro-batch 8, seq 2048 | 46.5 GB / 48 GB (R-01 smoke and the E4 canonical run agree) |
| controlled_ce train | micro-batch 8 | 31.4 GB (1-step probe) |
| Online smoke train (batch 1 or capture group 4) | micro-batch 4, seq 1024, 3B target resident | 13.67 GB target-device peak (R-20261014-12; batching moved it +0.004 GB) |

Rule: keep recipe micro-batch at 8; use grad accumulation for effective batch.

## Cleanup rules (applied)

Only this plan's registered, reproducible temp artifacts are pruned
(e.g. `runs/qwen25vl3b_dfly_2k_recipe_smoke/train/ckpt_last.pt` once E1 is
done and the checkpoint is no longer needed for resume tests). Never touch
other users' files, baselines or unique checkpoints.
Applied again on 2026-10-14 02:55: E4's `train/ckpt_last.pt` (3.7 GB,
renewable) deleted after its eval + export landed; the run's `ckpt_final` /
`ckpt_init`, the export dir and the small-file evidence
(`recipe-smoke-config/e4_evidence.tar.gz`) are kept.
