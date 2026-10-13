# VelaDraft plan - run log

One row per executed job. GPU-hours are charged as (devices x wall time),
including regen/capture/train/eval in the same job. The two entries below ran
before this log existed (setup-session smoke runs); they are still charged
against the plan's first-round budget.

| Run ID | Date | Job | Code | Inputs | Budget | Result | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-20261014-01 | 2026-10-14 | `upstream_two_stage` recipe smoke: overfit sanity + 16 training steps + gradient coverage | worktree = WP1 WIP (`66b595a`-equivalent) | features: `vlm-dflash/runs/qwen25vl3b_jetspec_2k/features` (1050 samples, store_final_hidden); target Qwen2.5-VL-3B; micro-batch 8; `cold_start_fraction 0.5` | 0.044 GPU-h (158.5 s, 1 GPU) | PASS for the stage contract: stage 0 = LK-only steps 1-8, stage 1 = CE + e2e TV steps 9-16; loss 11.0 -> 8.375; `nan_seen=false`; gradient coverage 42/42. Caveat: overfit sanity `passed=false` (slot0 acc 0.59 < 0.9 threshold; CE drop 0.815 > 0.6) - the sanity ran inside stage 0 (LK-only, no CE pressure on slot 0); peak GPU 46.5 GB | `a6000xuyang:~/vlm-recipe-smoke/runs/qwen25vl3b_dfly_2k_recipe_smoke` (`train/train_log.jsonl`, `train_summary.json`, `provenance.json`) |
| R-20261014-02 | 2026-10-14 | controlled_ce memory probe (1 step) | same | same feature store; micro-batch 8 config (name says mb16) | 0.002 GPU-h (8.7 s) | peak GPU 31.4 GB; sanity disabled by step budget | `a6000xuyang:~/vlm-recipe-smoke/runs/mem_ce_mb16` |

## Reuse sources (not charged; pre-existing plan assets)

- Feature store `~/vlm-dflash/runs/qwen25vl3b_jetspec_2k/features` (9.4 GB,
  1050 samples, capture v2 / final hidden) - the recipe teacher signal.
- Target models `~/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct` (7.1 GB),
  `Qwen3-VL-8B-Instruct` (17 GB).
- Canonical comparison runs from the previous program: `qwen25vl3b_dfly_2k`
  (AL 1.682, controlled_ce), `qwen25vl3b_dflare_2k` (AL 1.637), JetSpec store
  owner run - see `BENCHMARKS.md`.

## Planned (registered before launch)

| Run ID | Job | Success criterion | Budget |
| --- | --- | --- | --- |
| R-20261014-03 | `train_recipe: controlled_ce` 4 steps, tiny | loss trace equals no-recipe 4-step run (same seed) within bf16 reproducibility | 0.05 GPU-h |
| R-20261014-04 | recipe 4-step + resume 8-step, sanity disabled, micro-batch 8 | resume continues with matching stage log; `recipe_stage` consistent | 0.1 GPU-h |
| R-20261014-05 | export `ckpt_final` -> reload -> tiny eval + metadata check | token-exact smoke; `objective_profile` = recipe name in export | 0.1 GPU-h |
