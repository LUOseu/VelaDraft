# WP4 measurement - batched online capture + extraction profiling

> **R2 correction (2026-10-14, review `950b394`) - the extraction-wall
> percentage in this report is superseded/invalid.** The per-sample leg timed
> ``build_sample_record`` (render + forward + finalise); the batched leg timed
> only ``finalize_prepared_records`` (forward + finalise) - the prepare/render
> work was real but uncharged in leg B, so the two
> ``extraction_wall_seconds_total`` numbers are not the same metric. The
> 16 -> 4 target-forward counts, the bit-identical losses, the peak-memory
> readings and the profiling table remain valid as recorded; the "-18.8%"
> comparison does not. The unified boundary (prepare + forward timed on both
> paths, CUDA-synchronised) is implemented and CPU-tested (R2,
> `docs/plan/REVIEW_REMEDIATION.md`); the paired GPU re-measurement is a
> GPU_PENDING item. Raw logs are kept unchanged.

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
| `extraction_wall_seconds_total` (invalid comparison - R2: leg A included prepare/render, leg B did not) | 2.842 | 2.308 | not a gain; re-measure |
| train-phase target device peak | 13.669 GB | 13.673 GB | +0.004 GB |
| step losses | 13.5 / 15.5625 / 14.4375 / 16.375 | identical | bit-equal |
| train `wall_seconds` (incl. 3 resumable checkpoint writes) | 121.6 | 124.4 | noise on this disk-bound host |

One target forward serves 4 rows with no measurable quality or memory change.
The per-call reduction (16 -> 4) is the recorded, still-valid result; the
extraction-wall saving this report originally derived from the mismatched
timers must not be quoted and is being re-measured under the unified boundary.

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
  measured benefit, "at most one"), the batched capture - measured at 4x
  fewer target forwards with zero quality change (the extraction-wall
  percentage is superseded, see the R2 correction note) - is the WP4 optimization; the
  render-side candidates are documented here for a future round if many-epoch
  online 2K runs make extraction hot.

## R2 metric boundary (current definition)

Both extraction paths now charge the same boundary (`vlm_dflash/target/online.py`):

| Counter | Boundary |
| --- | --- |
| `prepare_seconds` | image decode + processor/render + prompt-drift checks |
| `forward_seconds` | teacher-forced target forward + output finalisation (one call per capture group) |
| `wall_seconds` (= `extraction_wall_seconds_total`) | prepare_seconds + forward_seconds, timed per row on both the per-sample and batched paths |
| `fast_forward_items` | resumed items answered with placeholders (no target work, never counted in wall time) |

The target device is CUDA-synchronised around every timed stage, so the
numbers cover completed GPU work (not just kernel launches). The startup
visual-path check still runs its fixed capture forwards outside these
counters. CPU fake-clock tests pin that equal work yields equal totals across
the per-sample and batched paths (`tests/test_online_mode.py`).

## Cost

0.21 GPU-h charged (R-20261014-12): A 3m13s + B 2m41s + two profiling passes
(~3.0 min and ~3.5 min wall, both dominated by the 2.5 min target load).
First-round budget after this row: 3.00 / 8.00 GPU-h used.
