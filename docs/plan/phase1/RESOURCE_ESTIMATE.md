# Phase-1 resource estimate (self-contained)

Accounting basis: `docs/plan/RESOURCE_LEDGER.md` + `docs/plan/RUN_LOG.md`
(measured), and the route document's rule that big-compute requests require
the §6 gate ("不自行开始大规模训练"). This phase (2026-10-14, `cvpr-route`):
CPU only.

## Spent and remaining (as of 2026-10-14)

| Item | GPU-h |
| --- | ---: |
| Charged before this phase (R-20261014-01..17) | 4.10 / 8.00 |
| **This phase (audit, tests, docs, cache/profiling studies)** | **0.00** |
| Remaining against the first-round cap | **3.90** |

Storage (measured 2026-10-14): GPU host `/home` 61 GB free (shared, drifting;
`RESOURCE_LEDGER.md`). Local Mac: tests and docs only.

## Estimates for the registered pending items (not authorised, order = review)

Method: per-stage measured walls from `docs/training-costs.md` and
`../WP4_MEASUREMENT.md`; ranges reflect shared-host contention (up to ~1.7x)
and first-call kernel compile. "GPU-h" = allocation x wall.

| # | Item (source) | Estimate | Basis |
| --- | --- | ---: | --- |
| 1 | Qwen multi-image/multi-turn post-R3/R4 correctness run (`REVIEW_REMEDIATION.md` GPU_PENDING #1) | 0.4-0.6 | E2 smoke measured 0.55 for two passes; same slice |
| 2 | Paired online/offline extraction re-timing under the R2 boundary (#2) | 0.15-0.3 | WP4 A/B legs 3m13s + 2m41s + profiling ~6.5 min |
| 3 | SmolVLM export -> fresh-process reload -> eval (#3) | 0.1-0.2 | R-17 eval legs only; capture/train already recorded |
| 4 | Same-engine first-divergence diagnosis inside vLLM (#4) | 0.3-0.6 | R-14..16 checked 0.55 for the whole bridge program |
| 5 | `train_recipe: upstream_dspark` GPU smoke (new, this phase) | 0.2-0.4 | one short recipe run on the reused store + export; mirrors R-01 scale |
| 6 | Length-bucketed profiling (vision vs text vs finalisation split, `PROFILING.md` §7) | 0.15-0.25 | one profiling pass per length bucket on the smoke slice |
| | **Total** | **1.3-2.35** | fits the 3.90 remaining with reserve |

Explicitly excluded from this table: any multi-seed quality matrix - that is
phase-2 scope and requires the §6 gate (below).

## Phase-2 estimate if the §6 gate is ever approved (NOT requested here)

Scale: one canonical 2K train+eval run measured 1.0-1.3 GPU-h (train
9.8-24 min + eval ~28 min; capture ~3 min when the store is not reused).

| Scope (route §6 row) | Runs | Estimate |
| --- | ---: | ---: |
| P2 profiling at scale (single/multi-image lengths) | 4-6 | 0.8-1.5 |
| P3 plan comparison (online / offline / fixed-space) under identical supervision | 4-8 | 4-10 |
| P4 learning pilot: 2 methods x 2 seeds x {recipe, baseline} = 8 runs | 8 | 8-11 |
| P5 deployment rehearsal (one engine, one target) | 3-5 | 1.5-3 |
| Reserve for retries / contention | - | 3 |
| **Total** | | **~17-28 GPU-h** |

These are ranges, not a request: the route document's §6.1 thresholds are the
decision rule and are evaluated in `GO_NO_GO.md`. If the execution-benefit
threshold is not met, "open-source engineering continues; no multi-node
request follows."

## Predictability notes (risks priced into the ranges)

- Shared host: identical protocols varied up to 1.7x; `/home` was 100% full
  with writeback storms during E4. Re-check free space and idle GPUs before
  launching anything.
- Disk per job kind (measured estimates): recipe training on an existing store
  ~5 GB net new; new 2K capture with `store_final_hidden` 9-12 GB; new model
  download 7-16 GB.
- Commit pinning: every estimate assumes the configs/commits recorded in
  `RUN_LOG.md`; anything re-tuned invalidates the comparison and the estimate.
