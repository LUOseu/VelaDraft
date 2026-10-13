# VelaDraft plan - resource ledger

## GPU budget

- First-round cap: **8.0 GPU-hours** (plan default; no other budget given).
  1 GPU job at a time; single job <= 2 h wall; <= 2 fix-retries per failing
  debug task.
- Charged so far: **1.34 GPU-h** (R-20261014-01 0.044 + R-20261014-02 0.002
  + E1 R-20261014-03..05 0.29 + duplicate concurrent E1 set R-20261014-07
  0.26 + zoo re-download validation R-20261014-09 0.04 + E2 R-20261014-08
  0.55, two passes + E3 R-20261014-10 0.15). Remaining: **6.66 GPU-h**; E4
  (in flight, budgeted 1.0) is charged when its row lands. Shared file
  between the two concurrent sessions - re-read before editing.
- First 0.5 GPU-h is measurement budget (throughput / peak memory / storage);
  the two smoke runs consumed part of it.

| Allocation (draft) | GPU-h |
| --- | --- |
| E1 WP1 validation (R-20261014-03/04/05) | 0.25 used -> actual 0.29 + 0.26 (duplicate set) |
| WP4 batched-online + profiling | 1.0 |
| E2 WP2 multi-image smoke | 0.5 budgeted -> 0.55 actual (two passes; second pass overran the cap) |
| E3 WP3 DocVQA/ChartQA validation | 0.7 budgeted -> 0.15 actual |
| E4/WP7 DFly 2K recipe canonical (+eval) | 1.0 |
| WP6 non-Qwen DFlash smoke | 1.5 |
| WP5 runtime load test | 1.0 |
| Reserve | 2.0 |

## Devices

- `a6000xuyang` (ssh `oab-remote`, 10.65.14.8): 4x RTX A6000 48 GB, idle at
  audit time; shared server - other users' jobs may appear.
- Local Mac M4 (16 GB): CPU tests/docs only.

## Disk (measured 2026-10-14)

| Path | Size | Free | Note |
| --- | --- | --- | --- |
| `a6000xuyang:/home` | 15 TB | **61 GB (100% used), drifting down** | shared; 130 GB free at 01:30 (audit), 61 GB at 02:20; a host-wide writeback stall hit the E2 eval ~02:10. Keep net-new small. |
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
| Recipe train + overfit sanity (16 samples) | micro-batch 8, seq 2048 | 46.5 GB / 48 GB |
| controlled_ce train | micro-batch 8 | 31.4 GB (1-step probe) |

Rule: keep recipe micro-batch at 8; use grad accumulation for effective batch.

## Cleanup rules (applied)

Only this plan's registered, reproducible temp artifacts are pruned
(e.g. `runs/qwen25vl3b_dfly_2k_recipe_smoke/train/ckpt_last.pt` once E1 is
done and the checkpoint is no longer needed for resume tests). Never touch
other users' files, baselines or unique checkpoints.
