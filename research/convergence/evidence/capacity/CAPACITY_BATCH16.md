# Capacity + padding probe — logical batch 16, W3 store (A6000 48GB), 2026-10-15

Protocol: `research/convergence/harness/r3_ab_step.py --batch-size 16 --warmup 1
--batches 4 --parity 0`, one frozen anchor table per logical batch, one optimizer
update per logical batch in every arm. `canonical micro16`: shipped layout in a
single 16-row physical chunk. `canonical micro8`: shipped 2-chunk accumulation.
`split/grid sub2`: 8 groups of 2 rows (positional split vs C x A-sorted), preserved
weights (fp32), the same supervision set and update count as the canonical arms.

| layout | wall (median of 4) | peak alloc | context padding waste | note |
| --- | --- | --- | --- | --- |
| canonical micro16 | — | — | — | **CUDA OOM 4/4 batches** (micro 16 > 48 GiB) |
| canonical micro8 | 1301.2 ms | 24.21 GiB | 12.6% | fits; reference for peak |
| split sub2 | 1513.8 ms | 9.37 GiB | 4.9% | fits; equal-k split control |
| grid sub2 | 1057.0 ms | 9.37 GiB | 3.3% | fits; −18.8% wall vs canonical micro8 |

Readings:
- Capacity headroom: a logical batch 16 that the shipped single-chunk layout
  cannot run fits under the regroup layout (and under plain splitting).
- Peak memory at equal semantics: 24.21 -> 9.37 GiB (−61%).
- Padding: 12.6% (canonical) -> 4.9% (split) -> 3.3% (grid); sorting by C x A
  removes about one quarter of canonical's waste; splitting alone does not
  capture the sorted-order share.
- Wall: 4 batches only, indicative; the full 3-rep table comes from the harness
  series. Split is slower than canonical micro8 here (more, smaller kernels).

Raw rows: `r3_capacity_mb16_vs_grid.jsonl`, `r3_capacity_canon_mb8.jsonl`
(this directory). Server log: `/home/lyh/veladraft-conv/logs/vconv_capacity_probe.log`.
