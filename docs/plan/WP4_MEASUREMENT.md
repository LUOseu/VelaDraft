# WP4 measurement - batched online capture + extraction profiling

Run: `R-20261014-12` (2026-10-14, GPU0 of `a6000xuyang`, plan owner
C-01a13621). Design: `docs/plan/WP4_DESIGN.md`. Code:
`vlm_dflash/target/{sample_capture,online}.py`, `vlm_dflash/dflash/train.py`
(commits `95d6f2e`, `c28ad14`); CPU coverage in `tests/test_online_mode.py`
(batched-vs-per-sample record equality on a tiny Qwen2.5-VL target incl. a
schema-v2 row, DataLoader order parity, digest coupling, fast-forward
accounting, Subset index mapping, capture-group chunking).

## Protocol (pinned before launch)

- Rows: the E2 WP2 smoke slice - 16 train rows
  (`~/vlm-recipe-smoke/runs/e2/smoke_multiturn`, synthetic 2-image / 2-turn,
  lengths 600-608 tokens), seq 1024, target Qwen2.5-VL-3B.
- Same data stage for both legs: B reuses A's `data` + `regen` (both stage
  manifests verified "unchanged inputs, skipping"), so the consumed rows are
  byte-identical.
- Config `configs/smoke_multiturn.yaml` (micro-batch 4, sanity off), 4 steps
  = 1 epoch, `--mode online`. A: `train.online_batch_size=1` (per-sample
  reference); B: `train.online_batch_size=4` (one padded forward per
  training batch).
- Metrics from `train/train_summary.json` -> `feature_source`.
- Follow-up profiling (same rows, no training):
  `scripts/profile_online_extraction.py --groups 1,2,4,8 --render-repeats 3
  --forward-repeats 4`; per group the median of 4 passes (the first pass is
  a ~2x warmup outlier and the median drops it).

## A/B result (registered runs, 4 steps each)

| Metric | A: batch 1 | B: group 4 | Delta |
| --- | ---: | ---: | --- |
| `target_forwards_total` | 16 | 4 | 4.0x fewer calls |
| `target_samples_total` | 16 | 16 | 0 |
| `extraction_wall_seconds_total` | 2.842 | 2.308 | **-18.8%** |
| extraction ms/sample | 177.6 | 144.3 | -18.8% |
| train-phase target device peak | 13.669 GB | 13.673 GB | +0.004 GB |
| step losses | 13.5 / 15.5625 / 14.4375 / 16.375 | identical | bit-equal |
| train `wall_seconds` (incl. 3 resumable checkpoint writes) | 121.6 | 124.4 | noise on this disk-bound host |

One target forward serves 4 rows with no measurable quality or memory change.
The extraction-wall saving (-19%) is smaller than the 4x call reduction
because prefill compute dominates and is nearly linear in rows on this slice
(3B target, ~600 tokens + 2 images per row).

## Profiling evidence (render vs forward)

| Phase | Per row |
| --- | ---: |
| render (`prepare`: decode + processor + drift check) | 16.5-18.6 ms |
| forward, group 1 | 139.3 ms |
| forward, group 2 | 136.4 ms |
| forward, group 4 | 129.3 ms |
| forward, group 8 | 133.2 ms |

Render breakdown (8 rows, real processor): PIL decode ~2.0-2.4 ms; processor
(resize/normalize/tensorize + template) ~10-19 ms. Prepared tensors are
~4.6 MB/row (968 patches x 1176 fp32), which is what any prepared-input cache
would have to hold.

Raw: `~/recipe-smoke-config/wp4_profile.log`, run logs
`wp4_ab1.log` / `wp4_ab4.log` / `wp4_ab_chain.log`.

## Decision: the "at most one I/O optimization" resolved to zero

- Prepared-input (render) cache: removes 10-19 ms/row on repeat epochs but
  costs ~4.6 MB/row (~4.8 GB for the 2K corpus; a capped LRU in shuffled
  order only partially hits). Ceiling ~5-11% of extraction wall.
- Decode-only cache: PIL decode is ~2 ms of the ~17 ms render - not measured
  to pay.
- Prefetch (overlap render with the previous forward): same ~11% ceiling of
  extraction, at the cost of threading the capture path.
- On this slice extraction is ~2% of the training wall; projected at ~20% for
  a 2K online run, a render-side optimization caps out at ~2% of step time.
  Per the plan's acceptance rule (no default-enabled optimizations without
  measured benefit, "at most one"), the batched capture - measured at -19%
  of extraction with zero quality change - is the WP4 optimization; the
  render-side candidates are documented here for a future round if many-epoch
  online 2K runs make extraction hot.

## Cost

0.21 GPU-h charged (R-20261014-12): A 3m13s + B 2m41s + two profiling passes
(~3.0 min and ~3.5 min wall, both dominated by the 2.5 min target load).
First-round budget after this row: 3.00 / 8.00 GPU-h used.
