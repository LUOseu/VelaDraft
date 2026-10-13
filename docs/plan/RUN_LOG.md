# VelaDraft plan - run log

One row per executed job. GPU-hours are charged as (devices x wall time),
including regen/capture/train/eval in the same job. The two entries below ran
before this log existed (setup-session smoke runs); they are still charged
against the plan's first-round budget.

| Run ID | Date | Job | Code | Inputs | Budget | Result | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-20261014-01 | 2026-10-14 | `upstream_two_stage` recipe smoke: overfit sanity + 16 training steps + gradient coverage | worktree = WP1 WIP (`66b595a`-equivalent) | features: `vlm-dflash/runs/qwen25vl3b_jetspec_2k/features` (1050 samples, store_final_hidden); target Qwen2.5-VL-3B; micro-batch 8; `cold_start_fraction 0.5` | 0.044 GPU-h (158.5 s, 1 GPU) | PASS for the stage contract: stage 0 = LK-only steps 1-8, stage 1 = CE + e2e TV steps 9-16; loss 11.0 -> 8.375; `nan_seen=false`; gradient coverage 42/42. Caveat: overfit sanity `passed=false` (slot0 acc 0.59 < 0.9 threshold; CE drop 0.815 > 0.6) - the sanity ran inside stage 0 (LK-only, no CE pressure on slot 0); peak GPU 46.5 GB | `a6000xuyang:~/vlm-recipe-smoke/runs/qwen25vl3b_dfly_2k_recipe_smoke` (`train/train_log.jsonl`, `train_summary.json`, `provenance.json`) |
| R-20261014-02 | 2026-10-14 | controlled_ce memory probe (1 step) | same | same feature store; micro-batch 8 config (name says mb16) | 0.002 GPU-h (8.7 s) | peak GPU 31.4 GB; sanity disabled by step budget | `a6000xuyang:~/vlm-recipe-smoke/runs/mem_ce_mb16` |
| R-20261014-03 | 2026-10-14 | E1a select-parity: `train_recipe: controlled_ce` 4 steps vs no-recipe 4 steps (same seed 42, same data stage) | `66b595a` worktree | same store; mb 8; sanity off; `save_every_steps 4` | 0.073 GPU-h (2 jobs, 143 s + 119 s) | PASS: loss/accuracy/valid_tokens/lr/grad_norm/fusion entropy all identical, steps 1-4 (bit-identical traces) | `runs/e1/e1_ce_select`, `runs/e1/e1_ce_plain` (`train_log.jsonl`) |
| R-20261014-04 | 2026-10-14 | E1b resume: straight 8-step recipe run vs kill-at-step-6 + `--resume` 8-step (stage boundary 4; kill left `ckpt_last` at step 4, `recipe_stage=1`) | same | same store; mb 8; sanity off | 0.101 GPU-h (straight 173 s; phase1 83 s killed; phase2 108 s) | PASS: all 8 steps bit-identical between straight and resumed runs; resume log reconciled 2 uncommitted rows to `train_log.uncommitted.jsonl`; resume preflight refused `--steps 4` with "Resume plan mismatch: checkpoint planned 8 total steps" (CPU preflight) | `runs/e1/e1_recipe`, `runs/e1/e1_recipe_resume` |
| R-20261014-05 | 2026-10-14 | E1c export + reload: export `ckpt_final` of both runs; metadata chain; reload eval via `demo --drafter` | same | CPU export + 1 GPU demo job (4 prompts, max 32 tokens) | 0.114 GPU-h (demo 410 s) | PASS: `ckpt.objective_profile=upstream_two_stage` with full `train_recipe` dict; export `config.objective_profile` + model card row + runtime note all stamped; demo reload token-exact 4/4 vs target greedy (AL 1.000 on the 8-step draft, expected) | `runs/e1/export_recipe`, `runs/e1/export_ce_plain`, `runs/demo_dfly_export_recipe` |
| R-20261014-06 | 2026-10-14 | E1d memory re-measure (inside R-04 runs) | same | same | 0 (shared) | recipe 8-step train peak **28.6 GB** with sanity off (vs 46.5 GB for the R-01 smoke whose overfit sanity dominated the peak); controlled_ce 1-step probe 31.4 GB | `kernels: train_summary.json` |
| R-20261014-07 | 2026-10-14 | duplicate E1 set from the other concurrent session: select-parity, kill@4 + resume, export + reload + 8-prompt eval | `6cf8141` | same store; slice 64; mb 8; sanity off | 0.26 GPU-h (train 118.7 + 85.1 + 163.8 + 125 + ~125 s; eval/export ~5 min) | PASS, agrees with R-20261014-03..05: select-parity traces bit-exact (`loss [12.3125, 22.0, 20.125, 21.0]`), 8-step kill+resume traces bit-exact (stage sequence `[0,0,0,0,1,1,1,1]`), export reload OK with `objective_profile=upstream_two_stage`, eval token-exact 8/8 (AL 1.133 vs init 1.000) | `~/vlm-recipe-smoke/runs/e1_ce_off`, `e1_ce_on`, `e1_recipe_straight`, `e1_recipe_kill`; `recipe-smoke-config/export_recipe_smoke`, `eval_recipe.log` |
| R-20261014-08 | 2026-10-14 | E2: WP2 multi-image/multi-turn smoke - synthetic 2-image/2-turn JSONL through prepare -> regen -> capture -> train (32 steps) -> eval | `6fd243c` worktree | `configs/smoke_multiturn.yaml`; 16 train + 4 val synthetic rows (`scripts/make_wp2_smoke_data.py`); target Qwen2.5-VL-3B; seq 1024; mb 4 | cap 0.5 GPU-h | PENDING (launched 2026-10-14 ~01:55, GPU0) | `~/vlm-recipe-smoke/runs/e2/smoke_multiturn`, `recipe-smoke-config/e2_train.log` |

## Reuse sources (not charged; pre-existing plan assets)

- Feature store `~/vlm-dflash/runs/qwen25vl3b_jetspec_2k/features` (9.4 GB,
  1050 samples, capture v2 / final hidden) - the recipe teacher signal.
- Target models `~/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct` (7.1 GB),
  `Qwen3-VL-8B-Instruct` (17 GB).
- Canonical comparison runs from the previous program: `qwen25vl3b_dfly_2k`
  (AL 1.682, controlled_ce), `qwen25vl3b_dflare_2k` (AL 1.637), JetSpec store
  owner run - see `BENCHMARKS.md`.

## Planned (registered before launch)

None outstanding. The three registered E1 runs executed on 2026-10-14 — once
per concurrent session (R-20261014-03..06 and the duplicate set
R-20261014-07; the two sets agree bit-exactly). E4 (canonical 2K recipe run)
is the next registered GPU job and stays on hold until the user rules on the
concurrent-session question.
