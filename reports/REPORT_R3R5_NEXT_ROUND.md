# VelaDraft R3/R5 round — 15K–20K effective samples, 3-layer vs 5-layer DFly

Status: **training + evaluation complete** (2026-10-16 12:2x); §9 is the recommendation, and the
serving/runtime follow-up that §7.3 opened is written up in `docs/parallel_verification_results.md`,
`docs/benchmark_metric_alignment.md` and `docs/vispec_runtime_comparison.md`.

## 0. Scope as executed

- Two formal runs only: **R3** (3 drafter layers) and **R5** (5 drafter layers), identical data,
  seed, recipe, optimizer, schedule and evaluation protocol; only `drafter.num_hidden_layers` differs.
- No 5K control, no extra sweeps (no other algorithm / LR / block size / data source).
- Both train from scratch; R5 does not inherit R3 weights and no layer-expansion initialisation was
  developed.
- Host: 4x RTX A6000 48G, `/home` filesystem effectively full -> all shared media is hardlinked and
  resumable training state lives on `/dev/shm`.

## 1. Provenance

- Repo: `/home/lyh/VelaDraft-scale` (clone of `github.com/LUOseu/VelaDraft`), start commit
  `5750c23` ("feat(eval): record per-prompt acceptance accounting in eval_samples.jsonl").
- Local uncommitted changes (no commit/push/PR was made): `reports/r3r5_local_diff.patch` - the
  round-scoped `git diff` over the tracked files this round actually touches, **8 files, +417/-33**
  (`vlm_dflash/data/prepare.py` +170, `vlm_dflash/target/regen.py` +72, `vlm_dflash/dflash/draft_model.py`
  +69, `vlm_dflash/dflash/train.py` +61, `vlm_dflash/eval/accept.py` +44, `scripts/eval_dflash.py` +18,
  `vlm_dflash/eval/run.py` +11, `tests/test_units.py` +5; the first four numbers are the training
  round, the rest include the serving/runtime work of §7.3) - plus untracked `scripts/scale/*`,
  `configs/scale/*`, `docs/*`, `reports/*`, `tests/__init__.py`. Summary of the changes:
  - `vlm_dflash/data/prepare.py` — image-content-grouped train/val/test split (union-find on image
    bytes), seeded group-level shuffle, `test_size`, `image_link_mode`, `num_groups`/`num_images`/
    `image_overlap` manifest stats.
  - `vlm_dflash/target/regen.py` — explicit greedy protocol (`do_sample=False`, `num_beams=1`,
    `repetition_penalty=1.0`) and the recorded `generation_protocol` block in `regen_stats.json`.
  - `vlm_dflash/eval/accept.py`, `vlm_dflash/eval/run.py`, `scripts/eval_dflash.py` — separate
    `speculative_rounds` vs actual `target_forward_calls`, `tokens_per_target_forward`,
    `target_forwards_per_round`, per-prompt rows, `--prompt-source-split`.
  - `vlm_dflash/dflash/train.py`, `dflash/draft_model.py` — `save_every_fraction_of_epoch`,
    eval-format `milestone_snapshots`, `resume_state_dir`, stage-boundary snapshot.
  - `tests/__init__.py` — makes `from tests.test_online_mode import ...` importable (a conda-env
    `tests` package shadowed the repo namespace dir; two pre-existing test failures fixed).
- **Concurrent-session contamination, excluded on purpose.** The working tree is shared with
  another session that is editing CVPR-benchmark code in the same checkout
  (`benchmarks/manifests/0{2,3,4}_*.json`, `benchmarks/prompts/*_v1.jsonl`,
  `scripts/prepare_benchmark.py`, `vlm_dflash/benchmark/adapters.py`; `adapters.py` was modified at
  02:51, i.e. *during* this round). A whole-tree `git diff` would therefore have attributed someone
  else's work to this round, so the round patch is deliberately scoped to the files listed above.
  The unfiltered tree diff is kept for transparency as `reports/r3r5_local_diff.full_tree.patch`
  and **nothing in this round depends on those benchmark files**.
- Environment: python 3.12 (`/home/lyh/miniforge3/envs/c1vllm/bin/python`), torch 2.11+cu130,
  transformers 5.11, no flash_attn; target `Qwen2.5-VL-3B-Instruct` (local, frozen).
- Full test suite: `pytest tests/ -q` (remote) — see §8 for the result; targeted suites
  (`test_prepare_media`, `test_resume`, `test_eval_checkpoints`, `test_strict_parallel`,
  `test_objectives`, `test_dfly`) pass.

## 2. Data

- Source: LLaVA-Instruct-150K (`/home/dhz/Data/llava_instruct_150k.json`) + COCO train2017 images.
- Config `scripts/scale/r3r5_data_config.yaml`: `slice=24000` (yields 24002 selected train rows),
  `val_size=256`, `test_size=256`, `seed=42`, `min_answer_chars=0`, `image_link_mode=hardlink`,
  `regen.max_new_tokens=256`, `min_response_tokens=8`, greedy.
- Split: rows are grouped by *image content* (sha256) and the groups are shuffled at group level
  (seed 42) into train/val/test, so an image never straddles splits. No index-order truncation.
- Regen ran as two contiguous chunks in parallel processes (rows 0:12000 and 12000:24002), merged
  in row order afterwards (`scripts/scale/split_input_rows.py`, `merge_regen_chunks.py`) — this is a
  throughput/restartability measure, not a data change.
- Datasets statistics: see `shared/data_stats.json` and §3 below.
- Coverage limitation: LLaVA-Instruct is image-QA/caption-style single-image data; this round does
  **not** cover multi-image, video, OCR-heavy or PDF/UI workloads.

## 3. Data statistics (filled after freeze)

Corpus `shared/regen/target_data.jsonl` frozen 2026-10-16T00:11:45+0800 at **N = 20000**.
Machine-readable: `shared/data_stats.json`, `shared/regen/freeze_record.json`.

- Input rows 24002 (from the 157712-row LLaVA-Instruct pool, group-level split seed 42); regen kept
  **23942 / 24002 (99.75%)**; 60 dropped, **all `too_short` (< 8 response tokens)**, no other reason.
- Frozen train corpus: **20000 rows / 11702 distinct images** (3942 kept rows dropped by the seeded
  freeze shuffle; frozen/dropped id hashes recorded, membership reproducible).
- Supervision: **1,701,868 unique supervised answer tokens** (mean 85.1 / row) + 7,822,274 prompt
  tokens. Repeating epochs re-presents the same tokens - not new independent supervision.
- Answer length: min 8 / p10 11 / median 89 / p90 161 / max 256; **730 / 20000 (3.65%) at the 256
  cap** (3.70% over all kept rows). Prompt length: min 286 / median 382 / max 575, so
  **0 rows exceed seq_length 2048 (0% truncation)**.
- Image band [200704, 1003520] px: 2765 rows upscaled from below, 0 downscaled from above, 17235 in band.
- Isolation: rows grouped by image content (sha256), group-level seed-42 shuffle; **train-n-val image
  overlap = 0, train-n-test = 0, val-n-test = 0** (12449 / 128 / 132 images). No index-order truncation.
- Dev/test: `val.jsonl` 257 rows, `test.jsonl` 257 rows (whole groups); the round uses
  **dev = first 256 val rows**, periodic subset = first 64 (prefix selection, so periodic ⊂ full dev);
  the test split is used for reporting only and never selects a checkpoint.
- Hashes: rows `fdffa4c2...`, val `6229aa8b...`, test `9a12e74f...`, `data.jsonl` `3ac88aa5...`,
  frozen corpus `44e357c7a1ab...`, full kept list `9ac59c3e...`.
- Generation protocol: greedy - `do_sample=false, num_beams=1, repetition_penalty=1.0`,
  max_new_tokens 256, EOS 151643/151645, `min_response_tokens=8`, `min_answer_chars=0`,
  multi-turn -> last turn only, batch 64.
- Coverage limitation: LLaVA-Instruct single-image QA/caption-style data only. This round does not
  cover multi-image, video, OCR/PDF/UI workloads; repeated images are not counted as independent data.

## 4. Training configuration (both runs)

target frozen Qwen2.5-VL-3B; `block_size=8`; DFly `context_layer_ids=[1,17,33]` (R5 keeps the same
target feature layers); `num_anchors=256`; BF16; flex_attention; spatial RoPE replace [16,24,24];
`feature_mode=online`; `online_batch_size=8`; host feature cache 32 GiB (reduced from the previous
round's 96 GiB to keep the shared host out of swap); micro-batch 8, grad-accum 1 (effective batch 8);
AdamW lr 3e-4, betas [0.9,0.95], wd 0, max_grad_norm 1; cosine, warmup 4%, min_lr_ratio 0.1;
`train_recipe=upstream_two_stage`, `cold_start_fraction=0.25`; <=3 epochs; seed 42;
checkpoints every 0.5 epoch + stage boundary + final, resumable state on `/dev/shm`.
The 3B target + live extraction peaks at ~36.3 GiB on a 48 GiB card (measured), so each run needs a
nearly-free GPU.

Resolved facts of the two launched runs (configs `configs/scale/r{3,5}_dfly_n20000.yaml`, archived
as `<run>/config.json` + `config.train.json`):

- Depth: `drafter.num_hidden_layers` 3 (R3) vs 5 (R5) - the only recipe difference; the target
  feature layers, capture set, recipe, schedule, data and seed are identical.
- Drafter parameters (gradient-coverage report, both runs): **R3 264.8 M / R5 418.9 M** trainable
  (fc 12.58 M both; decoder stack 231.2 M vs 385.4 M; hidden-correction 21.0 M both; fusion
  weights 9 vs 15). Fresh init for both: default HF init for the decoder stack + fc, DFly's
  `fusion_init=distance`, zero-init hidden-correction `down_proj` (upstream construction), and
  **no inheritance of R3 weights into R5** (no layer-expansion initialisation was developed).
- Plan: 20000 rows -> 2500 optimizer steps/epoch -> **7500 steps for 3 epochs**; two-stage boundary
  at **step 1875** (0.25 x 7500); milestones every 0.5 epoch (1250, 2500, ...), stage boundary and
  final snapshot all saved; eval-format snapshots in `<run>/train/`, resumable state in
  `/dev/shm/r3r5_r{3,5}/train_state`.
- **R5 geometry fallback (sanctioned by the round brief)**: at micro-batch 8 the 5-layer run needs
  more than the 48 GiB card (smoke OOM at 46.8 GiB in use; the first R5 attempt OOM'ed the same way
  at step 1). R5 therefore runs **micro_batch_size=4 x grad_accumulation_steps=2**, i.e. the same
  effective batch of 8, same 2500 steps/epoch and the same `loss/accumulation` normalisation
  (`vlm_dflash/dflash/train.py`); 5000 micro-batches/epoch divide evenly by 2, so no partial
  accumulation window. R3 keeps micro-batch 8 x accum 1. The aborted R5 attempt is preserved under
  `runs/r3r5_n20000/r5_attempt0_mbs8_aborted/` (its 1.68 GB `ckpt_init.pt` was moved to
  `/dev/shm/r3r5_n20000_aborted/` because `/home` had 6.7 GB free).

## 5. Training runs (in progress; interim numbers marked as such)

Live facts (all timestamps CST, host clock):

| | R3 (3 layers) | R5 (5 layers) |
|---|---|---|
| first launch | 00:15:33, gpu 0 (eval gpu 3) | 00:27:44, gpu 2 (eval gpu 1) |
| geometry | micro 8 x accum 1 | micro 4 x accum 2 (5L at micro 8 needs > 48 GiB) |
| total steps | 7500 (3 epochs x 2500) | 7500 |
| stage boundary | step 1875 | step 1875 |
| first dev point | step 1250 (0.5 ep): AL 2.054, slot0 0.520, tok/fwd 0.989, 64p, token_exact | - |
| restart | - | **restarted from scratch at 01:44:13 after ENOSPC at step 1008** (see §8) |
| 2nd interruption | - | **host OOM kill at 03:51:44 (step 2277); resumed 04:24:05 from step 1875** (see §8) |

Startup/compile cost is logged separately from the steady-state step rate (`driver/progress.jsonl`
ticks). Boot (import + target load + visual-path check + 4-sample overfit sanity, 200 steps) took
~4 min on R3; the in-trainer sanity runs before the main loop and is *not* part of the 7500 steps.
Measured steady-state (before the host filled up, on a quiet host): 2.36-2.68 s/optimizer-step for
R3 (micro 8) and 2.31-2.54 s for R5 (micro 4 x accum 2) -- i.e. the two extra draft layers cost
essentially nothing per optimizer step next to the frozen 3B target forward. Under host contention
(three trainers + a neighbour benchmark + tmpfs swap pressure, 01:10-01:45) both runs degraded to
3.4-4.5 s/step; ETA is recomputed from a 600 s window every tick.

Checkpoints: eval-format milestone snapshots every 0.5 epoch (1250 steps), at the stage boundary
(1875) and `ckpt_final.pt`; resumable states (model + fp32 master + Adam + scheduler) every 0.5
epoch under `/dev/shm/r3r5_r{3,5}/train_state/ckpt_last.pt`.

Loss / grad-norm / LR curves: `<run>/train/train_log.jsonl` (per optimizer step, R3 kept as
`/dev/shm/r3r5_cp/r3/train_log.jsonl`); dev curves: `<run>/driver/periodic_summary.jsonl`.

### 5.1 Matched-progress dev curve (64-prompt fixed subset, real decoding)

Machine-readable: `analysis/dev_points.json` (+ `.md`), rebuild with
`scripts/scale/collect_r3r5_results.py`; paired statistics in `analysis/paired_periodic_p*.json`.
Both arms are evaluated on the **same 64 held-out dev prompts at the same optimizer steps**, so
these rows are directly paired and matched by construction.

| step | epoch | R3 AL | R5 AL | R3 acc | R5 acc |
|---|---|---|---|---|---|
| 1250 | 0.50 | 2.0543 | 2.0362 | 0.1511 | 0.1482 |
| 1875 | 0.75 (stage boundary) | **2.2775** | 2.2109 | **0.1838** | 0.1741 |
| 2500 | 1.00 | 2.3118 | 2.3072 | 0.1880 | 0.1870 |
| 3750 | 1.50 | 2.5495 | 2.5257 | 0.2229 | 0.2194 |
| 5000 | 2.00 | 2.7452 | 2.7387 | 0.2509 | 0.2496 |
| 6250 | 2.50 | 2.8682 | **2.8843** | 0.2686 | **0.2702** |
| 7500 | 3.00 (`ckpt_final.pt`) | 2.9227 | 2.9171 | 0.2761 | 0.2752 |

Paired per-prompt bootstrap over the same 64 prompts (10k resamples); `al_micro` is the headline
metric, `al_macro` the mean of per-prompt ratios:

| step | al_micro diff (R5-R3) | al_macro diff | al_macro 95% CI | accept_rate diff | accept 95% CI | wins/losses (R5) |
|---|---|---|---|---|---|---|
| 1250 | -0.0181 | +0.0745 | [-0.0813, +0.2686] | +0.0086 | [-0.0116, +0.0325] | 23/27 |
| 1875 | -0.0666 | **-0.1555** | **[-0.2985, -0.0349]** | **-0.0212** | **[-0.0397, -0.0049]** | 15/32 |
| 2500 | -0.0046 | -0.0770 | [-0.2943, +0.1018] | -0.0096 | [-0.0353, +0.0120] | 23/27 |
| 3750 | -0.0237 | -0.0126 | [-0.1208, +0.0900] | -0.0020 | [-0.0175, +0.0126] | 21/20 |
| 5000 | -0.0065 | +0.0345 | [-0.0802, +0.1519] | +0.0057 | [-0.0093, +0.0217] | 24/22 |
| 6250 | +0.0161 | +0.0766 | [-0.0458, +0.2093] | +0.0102 | [-0.0072, +0.0292] | 24/19 |
| 7500 | -0.0055 | -0.0175 | [-0.1134, +0.0800] | -0.0015 | [-0.0156, +0.0125] | 18/24 |

**Reading (one seed; 64-prompt subset - the full-256 confirmation is §6).**

1. **On the 64-prompt subset exactly one point is resolvable, and it is R3's** - at the stage
   boundary (step 1875, 0.75 epoch) the 5-layer arm is behind: `al_macro` and `acceptance_rate`
   both have CIs excluding zero and R5 loses 32 of 64 prompts. **The full-256 dev confirmation in
   §6.3 dissolves even that point** (diff -0.0100 macro, CI [-0.0882, +0.0681], and -0.0131 micro),
   so the honest summary is "no resolvable difference anywhere". Everywhere else on the subset the
   CIs straddle zero, and at 3 epochs the two arms land within **0.0055 AL** of each other
   (2.9227 vs 2.9171), with 18/24 wins - a coin flip on 64 prompts. This is also the round's
   cleanest demonstration of why checkpoint *comparisons* must use the full dev: the subset both
   flatters absolute AL (+0.067) and manufactures a significant-looking gap that does not exist.
2. **So depth did not help at this budget; it produced a slower start that converged to parity.**
   The deficit decays -0.156 -> -0.077 -> -0.013 -> +0.035 -> +0.077 -> -0.018 - i.e. after the
   boundary it oscillates around zero rather than trending, which is what "no effect" looks like at
   this sample size, not what "5 layers need longer" would look like.
3. **Where the extra depth does show up is the tail, and it nets out.** Final per-slot acceptance,
   R3 vs R5: `0.691 0.469 0.293 0.193 0.132 0.087 0.059` vs `0.679 0.457 0.300 0.202 0.136 0.087
   0.058` - R5 is better on slots 2-4 (0.300/0.202/0.136 vs 0.293/0.193/0.132) and worse on slots 0-1.
   The deeper drafter does buy a slightly longer acceptance horizon, but it gives back at least as
   much at the first positions, so the aggregate AL is a wash.
4. At step 1250 `al_micro` and `al_macro` disagree in sign, so neither flavour is quoted alone, and
   the 64-prompt subset is optimistic relative to the full dev (§6.1: 2.9227 vs 2.8557), so these
   numbers order checkpoints and never decide the round.

## 6. Evaluation (filled after the runs)

Protocol (identical for both arms):

- **Metric definitions.** **Two** acceptance numbers are reported, because the literature uses both
  and they differ by the guaranteed-progress token (`docs/benchmark_metric_alignment.md`):
  `al_progress_tokens_per_round` = `new_tokens / speculative_rounds` (what this report calls `mean_al`,
  the legacy alias in the artifacts) and `al_draft_tokens_per_round` = `accepted_tokens /
  speculative_rounds`, which is the definition ViSpec's official README/speed.py reports as `tau`
  (their Medusa rows are < 1, which only a draft-only metric can be). **Aligned against ViSpec, R3 is
  1.851 (dev) / 1.798 (test), not 2.856 / 2.802**; the older headline numbers are progress-inclusive
  and must not be quoted next to a published `tau`.
  `acceptance_rate` = accepted_tokens / drafted_tokens (per-position rate
  averaged over all blocks). `per_slot_acceptance[i]` = fraction of verified blocks whose i-th draft
  position was accepted. `target_forward_calls` = *actual* target forward passes (1 prefill + 1 per
  speculation round + 1 per fallback), kept separate from `speculative_rounds`;
  `tokens_per_target_forward` and `target_forwards_per_round` are reported separately so rounds are
  never confused with target savings.
- **Real decoding, not teacher forcing.** Every dev point is a real greedy generation with the
  frozen target verifying sequentially (`--verification sequential`), `max_new_tokens=128`,
  temperature 0, on the fixed dev prompts. Teacher-forced / train-side acceptance is reported
  separately (train_log `accuracy`/`accept_len`) and never mixed with these numbers.
- **Sequential verification makes `tokens_per_target_forward` degenerate - say so instead of
  quoting it as a saving.** In the sequential path the target issues one forward per newly accepted
  token, so `target_forward_calls ~ 1 + rounds + accepted_tokens` and
  `tokens_per_target_forward` sits at ~0.985-0.989 *whatever the drafter quality is* (see the
  interim table in §5.1: it barely moves between step 1250 and step 3750 while AL rises 2.05 -> 2.55).
  The code says the same thing (`vlm_dflash/dflash/draft_model.py:1408-1416`: "under sequential
  verification one round issues several target forwards ... rounds must never be reported as
  target-call savings"). So under sequential verification **AL is a drafter-quality metric only**;
  the target-call saving story belongs to the parallel path and is reported separately in §7.
- **Dev numbering**: dev = first 256 rows of `val.jsonl` (image-group disjoint from train/test);
  every 0.5 epoch a fixed 64-row prefix subset is evaluated; checkpoint *selection* uses the full
  256-row dev; the test split is touched once at the end for reporting only.
- Checkpoints evaluated: every milestone (0.5 epoch), the stage boundary (1875) and `ckpt_final.pt`.
- Paired R3-vs-R5: per-prompt differences at matched steps with 10k-sample bootstrap 95% CIs
  (`scripts/scale/analyze_paired.py`), single seed explicitly treated as a limitation.
- Paired analysis keys prompts by `(id, occurrence_index)`, so duplicate ids inside the dev split
  are paired by position rather than silently dropped (an earlier `id`-only keying dropped 8 of 64
  prompts and is why the intermediate numbers are labelled interim). It reports `al_micro`
  (= `sum(new_tokens)/sum(verify_steps)`, identical to the headline `mean_al` of an eval point) and
  `al_macro` (= mean over prompts of `new_tokens/verify_steps`) **separately**; they are different
  numbers and are never mixed. At step 1250 they even disagree in sign (micro: R3 ahead by 0.018;
  macro: R5 ahead by 0.075), both with CIs straddling zero - a concrete reason the round reports
  both and does not select a winner from one.
- Every dev point is also available as per-prompt rows (`eval_samples_<tag>.jsonl`), which is what
  the paired bootstrap consumes.

### 6.1 R3 (3-layer) - FINAL

Training finished 06:58:59 at step 7500 (3.0 epochs) after 403.5 min of wall clock; the driver then
ran the confirm/test phase and logged `run_complete` at 08:11:07. Artifact:
`<run>/driver/selection.json`, checkpoints kept: `ckpt_final.pt`, `ckpt_step0006250.pt`,
`ckpt_step0001875.pt` (boundary).

| eval | checkpoint | prompts | AL | acceptance_rate | token_exact |
|---|---|---|---|---|---|
| periodic best (64) | `ckpt_final.pt` | 64 | 2.9227 | 0.2761 | True |
| **full dev (selection)** | `ckpt_final.pt` | 256 | **2.8557** | 0.2678 | True |
| full dev (selection) | `ckpt_step0006250.pt` | 256 | 2.7796 | 0.2573 | True |
| **test (report only)** | `ckpt_final.pt` | 256 | **2.8023** | 0.2610 | True |

- Selection used the **full 256-row dev**; the ranking above it came from the fixed 64-row subset,
  and the selection protocol never reads the test split (`selection.json` records both).
- Test per-slot acceptance: `0.670 0.432 0.279 0.185 0.121 0.079 0.049`.
- **Answer to the round's question 1: yes.** 20 000 effective training samples and 3 epochs give a
  held-out acceptance length of **2.86 on dev and 2.80 on a test split that was never used for any
  decision** - a usable, non-degenerate speculative drafter, with the dev/test gap small (2.86 ->
  2.80), so the number is not a dev-specific artefact. The `mean_al` ceiling for `block_size=8` is
  8.0, so 2.80 means a verified block yields ~2.8 tokens per round.

### 6.2 R5 (5-layer)

Training finished 09:09:39 at step 7500 (3.0 epochs) after two host-OOM kills and one disk-full kill
(§8); the driver's confirm/test phase logged `run_complete` at 10:30:51.

| eval | checkpoint | prompts | AL | acceptance_rate | token_exact |
|---|---|---|---|---|---|
| periodic best (64) | `ckpt_final.pt` | 64 | 2.9171 | 0.2752 | True |
| full dev (selection) | `ckpt_final.pt` | 256 | **2.8332** | 0.2649 | True |
| full dev (selection) | `ckpt_step0006250.pt` | 256 | 2.7602 | 0.2542 | True |
| test (report only) | `ckpt_final.pt` | 256 | 2.7940 | 0.2599 | True |

Per-slot acceptance on the full 256-row dev, `ckpt_final.pt`: `0.674 0.443 0.288 0.185 0.123 0.079
0.053`.

### 6.3 R3 vs R5 head-to-head at matched progress (the round's question 2)

Both arms kept the same checkpoints, so the comparison is exactly matched: identical data, seed,
recipe, schedule, tokenizer, target, feature layers and eval protocol, differing only in
`drafter.num_hidden_layers`. Full-dev numbers, 256 held-out prompts, real greedy decoding with
sequential verification, `token_exact=True` for all four evals. Statistics:
`analysis/paired_full_dev_{mdev_step1875,dev_step6250,dev_final}.json` (macro/accept figures are
paired per-prompt means; the 0.75-epoch row's micro numbers are from `eval_result_mdev_step1875.json`).

| step | R3 micro AL | R5 micro AL | micro diff | R3 al_macro | R5 al_macro | macro diff | macro 95% CI | R3 acc | R5 acc | acc diff | acc 95% CI | R5 W/L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1875 (0.75 ep) | 2.1595 | 2.1464 | **-0.0131** | 2.5887 | 2.5787 | -0.0100 | [-0.0882, +0.0681] | 0.2265 | 0.2247 | -0.0018 | [-0.0123, +0.0086] | 93/110 |
| 6250 (2.5 ep) | 2.7796 | 2.7602 | **-0.0194** | 3.5212 | 3.5391 | +0.0179 | [-0.0408, +0.0828] | 0.3579 | 0.3592 | +0.0014 | [-0.0067, +0.0100] | 86/95 |
| 7500 (3.0 ep) | 2.8557 | 2.8332 | **-0.0225** | 3.6044 | 3.5716 | -0.0328 | [-0.0918, +0.0271] | 0.3686 | 0.3643 | -0.0043 | [-0.0127, +0.0043] | 77/101 |

Per-slot acceptance at step 7500 (R3 / R5):
`0.684 0.448 0.290 0.187 0.124 0.080 0.053` / `0.674 0.443 0.288 0.185 0.123 0.079 0.053`; at step
6250: `0.672 0.434 0.276 0.173 0.113 0.074 0.051` / `0.664 0.428 0.274 0.172 0.111 0.073 0.049`.

**Answer to question 2: the 5-layer arm did not beat the 3-layer arm, so at this budget the extra
depth is not worth its cost.**

- **No resolvable difference.** On the full 256-row dev, no metric has a bootstrap CI excluding zero
  at either matched step, and the two arms land within **0.02 micro AL (~0.8% relative)** of each
  other. A paired sign test agrees: at 7500, R5 loses 101 of 178 non-tied prompts (z=-1.72,
  p=0.085); at 6250 it is 95 of 181 (z=-0.59, p=0.55). Nothing reaches significance, and where the
  evidence leans at all it leans toward the shallower model.
- **The one difference the 64-prompt subset found does not survive confirmation.** At the 0.75-epoch
  stage boundary the subset put R5 significantly behind (§5.1); on the **full 256-row dev** the same
  checkpoint pair is indistinguishable: `al_macro` -0.0100 (CI [-0.0882, +0.0681]), `accept_rate`
  -0.0018 (CI [-0.0123, +0.0086]), micro AL_progress 2.1595 vs 2.1464 (**-0.0131**), `token_exact`
  True for both (`analysis/paired_full_dev_mdev_step1875.json`, `eval_result_mdev_step1875.json`).
  So the honest summary at matched progress is "**no resolvable difference at any step, including
  the stage boundary**", not "5 layers start slower and converge to a tie".
- **Where depth does show up, it nets out.** Per-slot acceptance differences are ≤0.01 absolute at
  every position. On the small 64-prompt subset R5 briefly led slots 2-4; on the full dev that does
  not survive - R3 is ahead on every slot at both matched steps. So there is no acceptance-horizon
  win to trade against the cost.
- **And the cost is real.** R5 carries **418.9 M** trainable parameters against R3's **264.8 M**
  (+58%): the decoder stack is 385.4 M vs 231.2 M and there are 15 fusion weights vs 9, while the
  shared fc (12.58 M) and hidden-correction (21.0 M) are identical (§4). It also needs micro-batch
  4 x accum 2 to fit a 48 GiB card where R3 fits micro-batch 8 x accum 1 (§4), i.e. half the batch
  geometry headroom. The per-sample drafter latency is reported separately in §7 - an AL tie cannot
  be turned into a "not better" claim without it, and it is measured on an exclusive GPU there.

**Caveats, stated rather than buried.** One training seed per arm. The bootstrap quantifies
prompt-sampling uncertainty only, not seed-to-seed variance, so "no resolvable difference" means
"this round cannot resolve a difference of this size", not "the two models are provably identical";
with ~0.02 AL between them and 256 prompts, a difference of that size is simply below the resolution
of a single-seed 256-prompt comparison. The R5 arm also absorbed 442 steps of replay from the host
OOM kill (§8), but those steps were re-run under the identical recipe and schedule, so they affect
R5's wall clock, not its optimisation path.

## 7. Speed / quality

Protocol (identical for both arms, run back-to-back on the **same exclusive GPU** by
`r3r5_postrun.sh`, which waits for both drivers so nothing else is on the card): 3 warmup prompts
(1 target-only, 1 speculative) excluded from the measured window, then the same 32 fixed dev prompts
decoded twice in one process - target-only greedy, then speculative - both `max_new_tokens=128`,
temperature 0, bf16. Reported: paired wall ratios, output-token equivalence, speculative rounds,
**actual** target forward calls, fallback rounds, and the draft / target-verify / target-prefill
split. Raw artifacts: `<run>/eval/speed_paired_r{3,5}_speed_final.json`, logs `logs/32_speed_r{3,5}.log`.

### 7.1 Sequential verification (the lossless reference) - FINAL: no speedup, and that is structural

| | R3 | R5 |
|---|---|---|
| target-only wall / sample | 2.2547 s | 2.1595 s |
| speculative wall / sample | 2.6248 s | 2.5728 s |
| **speedup (ratio of means)** | **0.8590x** | **0.8394x** |
| speedup (paired mean of ratios / median) | 0.8799x / 0.8614x | 0.8616x / 0.8404x |
| `token_exact` | **32/32** | **32/32** |
| new tokens | 2280 | 2280 |
| speculative rounds | 813 | 821 |
| **actual target forward calls** | 2317 | 2309 |
| target forwards per round | 2.850 | 2.812 |
| tokens per target forward | 0.9840 | 0.9874 |
| fallback rounds | 0 | 0 |
| draft / verify / prefill seconds | 10.62 / 68.28 / 3.92 | 12.83 / 64.35 / 4.03 |
| mean AL (this 32-prompt set) | 2.8044 | 2.7771 |
| peak GPU memory | 8.63 GB | 8.95 GB |

**The measured slowdown is the theoretical value, not a bug.** Sequential verification issues one
target forward per newly accepted token, so its forward count is identical to target-only's
(`2317` for 2280 generated tokens + 32 prefills). Decomposing the wall clock confirms it exactly:

| | target work inside spec (verify + prefill) | target-only wall | drafter cost | predicted speedup | measured |
|---|---|---|---|---|---|
| R3 | 68.28 + 3.92 = **72.20 s** | 72.15 s | 10.62 s | 72.15/(72.20+10.62) = **0.872x** | **0.859x** |
| R5 | 64.35 + 4.03 = **68.38 s** | 69.10 s | 12.83 s | 69.10/(68.38+12.83) = **0.851x** | **0.839x** |

The speculative path spends *the same* target time as target-only and then adds the drafter on top,
so the ceiling for this path is 1.0x (attained only with a free drafter) and the observed 0.86x/0.84x
is where theory says it should be. **Consequence: the sequential path can never be the speed story.
It is the correctness reference only** - and it is a good one, since it is 32/32 token-exact. This is
also why `tokens_per_target_forward` must not be read as a saving: under sequential verification it
is ~1 by construction (0.984/0.987 here) and is invariant to drafter quality.

### 7.2 R5-vs-R3 drafter cost

Held to the same prompts, same decoding, same GPU:

- Drafter time per round: R3 **13.06 ms**, R5 **15.62 ms** -> the two extra layers cost **+19.6% per
  round**, i.e. **+2.21 s over the 32-prompt set (+20.8%)**.
- Parsed against the AL tie from §6.3, R5 therefore pays more and gains nothing: the same
  32-prompt set gives R5 **2.7771** mean AL against R3's **2.8044**, so on this block the deeper
  drafter is behind on both axes at once.
- Note the drafter is ~15% of the target work (10.62 s vs 72.20 s for R3), so even a perfect drafter
  would not move the sequential number much - the cost of depth shows up as inference latency, and it
  is a real cost, not a rounding error.

### 7.3 Parallel verification - the only path that can beat target-only

**Official 32-prompt measurement (both arms, `r3r5_matchdev.sh`, exclusive GPU, same protocol as
§7.1): `speed_paired_r{3,5}_speed_final_parallel.json`.**

| | R3 | R5 |
|---|---|---|
| wall speedup (ratio of means) | **1.902x** | **1.775x** |
| paired mean / median of per-sample ratios | 2.117x / 2.043x | 1.958x / 1.865x |
| `token_exact` | **18/32** | **18/32** |
| tokens per target forward | 2.763 | 2.747 |
| target forwards per round | 1.040 | 1.040 |
| AL_progress / AL_draft | 2.874 / 1.868 | 2.856 / 1.848 |
| draft / verify / prefill share of wall | 28.0% / 60.9% / 9.6% | 32.2% / 57.9% / 8.5% |

So the parallel path **does** convert acceptance into wall-clock speed - ~1.9x on R3 against 0.86x
for the lossless sequential path - but **it is not token-exact on 14 of 32 prompts**, so it cannot be
quoted as a lossless serving speedup. Root cause, per-prompt evidence and the full decomposition are
in `docs/parallel_verification_results.md`; the short version is a last-ULP bf16 near-tie at the
deciding position (same top-5 candidate set, top-2 swapped by 0.0625 at magnitude ~15), reproduced
under both SDPA backends and eliminated by neither the parity backend nor the margin-2.0 guard (which
costs more than the sequential path: 0.667x, 19/32). **End-to-end faithful speedup: still not
verified.**

Parallel verification is where the speedup has to come from, because one target forward verifies the
whole block instead of one token, so `tokens_per_target_forward` approaches AL instead of being
pinned at ~1. The round's rule applies: **a parallel result that is not `token_exact` cannot be
presented as the faithful speedup**, so the 1.902x above is reported as an *unverified-fidelity*
number next to its 18/32 exactness, not as a serving result.

**Cost decomposition, measured directly** (`scripts/scale/profile_parallel.py`, same 32 prompts,
same GPU, target-only reference probe of ~29 ms per single-token decode):

| component | ms per round | share of speculative wall |
|---|---|---|
| target block verify (1 forward) | 29.77 | 62.5% |
| drafter (1 block forward) | 13.00 | 27.3% |
| target prefill (amortised over 25 rounds) | 4.19 | 8.8% |
| cache commit / verify prep / round prep | 0.35 combined | 0.8% |
| misc (python glue, sampling, stop scans) | 0.35 | 0.7% |

which gives `r = 1.000` (a block verify costs exactly one single-token decode - the floor for a
one-forward-per-round design), `d = 0.447`, `A = 2.874`, so `S_model = A/(r+d) = 1.99` against
`S_wall = 1.86` (the ~7% residual is prefill). Consequences for the next round: **`r` is out of
levers; the speedup is `A/(1+d)`**, so a cheaper drafter (`d`) and a higher `A` are worth roughly the
same - `d = 0` would give 2.87x and `A = 4` at the current `d` would give 2.76x. An earlier draft of
this report quoted `speedup ~= (AL/r)/(1+drafter_share)` with an inferred `r ~ 1.63` from the 4-prompt
dry run; the direct measurement supersedes it, and the 4-prompt dry run (1.462x, 3/4 exact) is kept
in `docs/parallel_verification_results.md` only as harness-validation history.

**Strict-margin sweep (R3, same 32 prompts, exclusive GPU, 12:27-12:41; full table in
`docs/parallel_verification_results.md` §3.6).** The near-tie guard leaves no usable setting:

| margin | token-exact | wall speedup | `r` | fallback rounds |
|---|---|---|---|---|
| 0.25 | 16/32 | 1.208x | 1.815 | 25.9% |
| 1.0 | 19/32 | 0.746x | 3.243 | 77.5% |
| 2.0 | 19/32 | 0.678x | 3.625 | 92.2% |
| 1e9 (all-fallback control) | **32/32** | 0.655x | 3.828 | 100% |

Exactness is reachable only at the all-fallback control, and that point is dominated by the plain
sequential path (0.655x vs 0.853x, because every round also pays a discarded block verify); the fast
end is not exact; and `m=0.25` is *worse* than no guard (16/32 vs 18/32) at 65% of the speed, because
a partially-firing guard mixes block-shaped and token-shaped KV along the stream. Practical reading:
the round's parallel path neither needs nor benefits from the strict guard - keep it off.

**Same-machine ViSpec control (task doc §12, P2; official checkpoint, SQA, T=0).** The official
`gen_{baseline,spec}_answer_sqa` entry points were run with the repo's own `exp.sh` flags on 100 SQA
test questions (seed 42), same A6000, bf16, SDPA fallback (no flash-attn): baseline 3.13 s/question
vs spec 1.41 s/question, official-style speedup **2.038x**, `tau` **1.9810** (their README SQA row:
1.99 / 1.87x). So an official ViSpec deployment on this card is a ~2x system - the same league our
parallel path measures (1.86-1.90x) - and *it too is not token-exact against its own baseline*
(32/100 identical, divergences mid-generation). Full numbers and caveats:
`docs/vispec_runtime_comparison.md` §7.

### 7.4 Serving/runtime tooling shipped in this round (code, no training)

The §7.3 diagnosis required instruments that did not exist, so they were added (all additive, no
behaviour change when disabled):

- `vlm_dflash/eval/accept.py` - dual acceptance reporting: `al_draft_tokens_per_round` (ViSpec's
  `tau` definition) alongside `al_progress_tokens_per_round` / legacy `mean_al`
  (`docs/benchmark_metric_alignment.md`).
- `vlm_dflash/dflash/draft_model.py` - CUDA-event phase brackets (`round_prep`, `draft`,
  `verify_prep`, `target_verify`, `target_prefill`, `commit`) and an opt-in
  `capture_decision_trace` that records position / forward shape / KV length / argmax / top-5 logits
  per verification forward.
- `scripts/scale/profile_parallel.py` (new) - target-only decomposition probe + speculative run,
  reporting `A`, `r`, `d`, `S_model`, `S_wall` and the full phase breakdown.
- `scripts/scale/diag_parallel_first_divergence.py` (new) - four-path first-divergence diagnosis with
  per-path decision traces and a target-only bit-stability check.
- `scripts/scale/speed_paired.py` - extended with the dual AL metrics and the phase breakdown.
- `tests/test_units.py` - asserts the dual-metric maths (`al_progress - al_draft == 1.0` in the
  uncapped case).
- Outside the repo (remote-only harness): the same-machine ViSpec control - venv + official
  checkpoint under `/home/lyh/ViSpec-reference/`, driver `/tmp/run_vispec_sqa.sh`, result collector
  `/tmp/vispec_collect.py`, raw answer files under
  `/home/lyh/ViSpec-reference/eval_runs/sqa_{baseline,spec}/` (copies in
  `reports/artifacts/r3r5_20k/vispec_sqa/`); documented in `docs/vispec_runtime_comparison.md` §7.

## 8. Failures / anomalies / open items

Infrastructure (all recovered; none changed the recipe for either arm):

1. **R5 ENOSPC crash + clean restart (01:27:30).** `/home` is a 15 TB volume at 100% occupancy
   shared with other users (2.6-6.7 GB free during the round, once 1.2 GB). R5's
   `train/train_log.jsonl` append hit `OSError: [Errno 28] No space left on device` at step 1008;
   the resumable-state cadence is 0.5 epoch (1250 steps), so no train-state existed and the driver
   reported `attempt_failed rc=1 oom=false` and stopped. Recovery: both runs' `train/` *and*
   `driver/` directories now live on `/dev/shm` via symlinks, the aborted attempt's 1008-step log is
   preserved (`/dev/shm/r3r5_cp/r5/train_log.attempt1_enospc.jsonl`, one truncated final line
   dropped) and R5 was relaunched from scratch at 01:44:13 with identical seed/recipe/geometry.
   Cost: ~50 min of R5 wall clock; no dev point had been produced yet, so no result is affected.
   Restart determinism is checked against the preserved attempt-1 log.
2. **Host memory pressure / tmpfs thrash (01:00-01:15).** 251 GB host, swap full, 1 GB free
   (`wait_on_page_bit_common` stalls; NVRM allocation failures in dmesg). Cause: 34 GB of my smoke
   temporaries on `/dev/shm` + three trainer processes + a neighbour's 69 GB tmpfs. Fixed by
   deleting the smoke artifacts (small evidence JSONs archived to `logs/smoke_artifacts/`); the hung
   5L resume check was killed and re-run successfully (rc=0, exact step 60).
   **Follow-up (05:50):** a further **15 GB** of my own smoke scratch was still on `/dev/shm`
   (`/dev/shm/r3r5_smoke`, 12736 files: the 5L resume-gate state and the R5 smoke run). tmpfs pages
   count against host RAM, so this was reclaimed after first writing
   `logs/smoke_artifacts/smoke_shm_manifest.tsv` (every file with its size) and after confirming the
   small evidence JSONs were already archived in `logs/smoke_artifacts/`. Result: `/dev/shm` free
   18 -> 33 GB, host `available` 80 -> 90 GB. `/home` itself stayed at ~0.9 GB free because that is
   other tenants' occupancy, not ours (our `/home` footprint is ~10 GB against 14 TB used).
3. **First R5 attempt OOM at micro-batch 8** (46.8 of 47.4 GiB in use) - expected; R5 runs
   micro 4 x accum 2, the round brief's sanctioned fallback.
4. **Driver crash on first launch (00:13).** `Driver.__init__` planned the step count from
   `<run>/regen/target_data.jsonl` before `link_shared()` staged it -> `FileNotFoundError`. Fixed by
   planning after the link step; verified by the 00:15:33 relaunch.
5. `ckpt_init.pt` (fresh-start snapshot; 0.53 GB R3 / 1.68 GB R5) was dropped by the driver's
   disk-pressure rule (< 4 GB free) in both runs; sha256s recorded in `driver/progress.jsonl`.
6. `r3_driver/train_stdout.log` was unlinked during the /dev/shm migration while the trainer kept
   the inode open; the live file is snapshotted every minute from `/proc/<pid>/fd/1` to
   `/dev/shm/r3r5_cp/r3_driver/train_stdout.live.log` (the per-step stdout record is therefore
   complete).
7. **Host OOM killer SIGKILLed the R5 trainer at 03:51:44 (`attempt_failed rc=-9`, step 2277).**
   dmesg: `Out of memory: Killed process 1838489 (python) ... anon-rss:40249972kB`. The host has
   251 GB RAM, 7 GB swap (fully used) and ~104 GB of tmpfs held by other tenants; the two trainers
   (R3 37.7 GB + R5 40.2 GB anon RSS) are the largest RSS objects on the host, so the kernel picks
   them. The whole process tree (driver included) died, and the driver *cannot* tell a host SIGKILL
   from an ordinary crash - the log contains no OOM text - so it stopped by design and no fallback
   geometry was attempted (which was lucky: the ladder's next entry starts from scratch).
   Recovery, all local and confined to `scripts/scale/r3r5_train_run.py`:
   - `--start-attempt N` - the ladder always resumed at attempt 1 (fresh start), which would have
     discarded 1875 committed steps; R5 was relaunched with `--start-attempt 2` (`spec-resume`).
   - `committed_state_step()` guard - any later ladder entry that would start *from scratch* is now
     refused and logged as `fresh_restart_refused` whenever a committed resumable state exists, so a
     misdiagnosed crash can never silently throw away hours of optimisation.
   - `load_previous_evals()` - a relaunched driver re-adopts the milestone evals already recorded in
     `periodic_summary.jsonl`, so it does not re-pay for them and does not forget which snapshot is
     the best (otherwise the next `prune()` would delete it).
   The trainer reconciled its own log on resume (`reconciled train log to committed step 1875
   (402 uncommitted rows -> train_log.uncommitted.jsonl)`), so `train_log.jsonl` has no duplicated
   or out-of-order steps. R5 relaunched 04:24:05 from step 1875; **cost 402 steps ~ 24 min**.
   Residual risk: R3's driver still runs the pre-fix code (restarting it would have cost ~950 steps
   of committed progress), and the host memory pressure is not ours to remove. A read-only
   `host_mem_watch.sh` now logs `free` + top-5 RSS every 60 s to `logs/60_mem_watch.log` so any
   repeat can be attributed rather than guessed.
8. **R5 killed a second time, 06:04:35 (step 4192), same cause.** dmesg shows our trainer
   (`anon-rss:45962776kB`) plus *three other tenants'* processes of 78 GB / 94 GB / 94 GB RSS being
   killed at 06:07, 06:14 and 06:21 - the host is in a sustained over-commit, not a one-off spike.
   R5's driver correctly refused to fall back (attempt 2, `oom=false` -> stop), so nothing was
   destroyed; the run was resumed from its committed state at **step 3750**, losing 442 steps ~ 20
   min. Response, kept deliberately minimal:
   - `scripts/scale/r3r5_restart_r5.sh` - a resume-only relaunch path (always `--start-attempt 2`,
     same micro 4 x accum 2 geometry, same config/data/seed/protocol).
   - `scripts/scale/r3r5_watchdog.sh` - auto-resumes whichever arm loses its whole process tree, only
     when both its driver and trainer are gone and `run_complete` has not been logged; every action
     lands in `logs/70_restarts.log`.
   - Deliberately **not** done: changing `save_every_fraction_of_epoch`. That knob also drives the
     milestone snapshots (`train.py:1301-1345` writes the resumable payload and the eval-format
     snapshot in the same callback), so lowering it for one arm would move that arm's periodic dev
     points off the matched steps and break the paired comparison. The cadence stayed at 0.5 epoch
     for both arms and the ~20-25 min worst-case replay cost per kill was accepted instead.

- In-trainer overfit sanity at `num_samples=16` needs ~47 GiB and OOMs on a 48 GiB card even alone;
  reduced to **4 samples** (identical for both arms) as a plumbing gate. The scientific recipe is
  unaffected.
- GPU contention: a parallel audit/benchmark session of the same account holds GPU1/GPU3
  intermittently (8–11 GiB), which caused three smoke OOMs at the spec geometry; the run driver now
  ships a recovery ladder (resume at spec geometry → micro-batch 4 / accum 2 → +online_batch 4).
- **Operational finding: the 32 GiB host feature cache is nearly useless at this data scale and is
  a direct contributor to the OOM war.** R3's finished `train_summary.json` reports
  `capacity_bytes 34359738368`, `peak_bytes 34359736680` (i.e. pinned at the cap),
  **`hits 1861` vs `misses 58939` = a 3.1% hit rate**, `stores 58939`, `evictions 54542`
  (92.5% of everything stored was evicted before reuse), `skipped_too_large 0`. With DFly features
  at ~25 MB/row (3 capture layers x 2048 hidden x bf16 x 2048 tokens), the ~20000-row corpus needs
  roughly 0.5 TB to cache, so a 32 GiB cap covers ~1400 rows and LRU evicts them before the next
  epoch reaches them. The knob therefore buys ~3% of target forwards while holding ~34 GB of host
  RAM - the single largest lever we have on the memory pressure that has killed R5 twice. It was
  **left unchanged for both arms in this round** because it is part of the frozen shared recipe; see
  §9 for the recommendation.
- Startup sanity (pre-run gates): 3L/5L overfit sanity passed (ce drop 0.937/0.932, slot0 1.0),
  gradient coverage 42/42 and 64/64 tensors, 5L checkpoint eval-load `token_exact=True`, and both
  kill/resume gates reached exactly step 60 (3L at micro 8 x accum 1, 5L at micro 4 x accum 2).
- Full-suite `pytest tests/ -q` on the run environment: see §1/`logs/90_pytest_full.log` (passes,
  6 skipped, two pre-existing failures fixed by `tests/__init__.py`).
- Open item: 5L resume is validated on the pilot corpus only, not on the full 20K run; no
  per-atom re-verification of the single-process (DDP-free) setup was done (out of scope).
- **Checkpoint durability:** every training checkpoint, including both `ckpt_final.pt` files, lives
  on tmpfs (`/dev/shm/r3r5_cp/{r3,r5}/`, adopted during the `/home` ENOSPC episode) and is therefore
  lost on a host reboot. Durable copies of both final checkpoints plus their train logs/summaries
  were placed in `runs/r3r5_n20000/durable_ckpts/` with an `MD5SUMS` manifest
  (`r3_ckpt_final.pt` `6ad3389b2ad5b2796111a32363193274`, `r5_ckpt_final.pt`
  `5043368fe7f76a90456fbfc38e2eb581`, verified against the shm originals); the shm copies stay in
  place so the documented eval/resume paths keep working.

## 9. Next round recommendation

**Bottom line: do not borrow A100 time for "more of the same", and close the depth question here.**
The round's constraints turned out to be neither GPU capacity nor data volume: the frozen target at
`block_size=8` yields **AL ~2.8**, and the wall-clock story is gated by (a) a parallel-verification
path that is not yet token-exact and (b) an acceptance ceiling that two extra drafter layers did not
move. (a) costs engineering rather than GPU money - the same-machine ViSpec control (§7.3) shows a
2x-class system on this exact card, so 2x is a proven target, not a hardware question - and it is the
only thing standing between this work and a real end-to-end speedup, so it is the next thing to
spend on, not a bigger machine.

### 9.1 What the round now rules out (and what it cannot answer)

- **Depth: closed.** 5 layers does not beat 3 layers on any held-out metric at matched progress
  (§6.3), costs +58% drafter parameters, +19.6% drafter latency per round (§7.2), and needs half the
  batch geometry to fit a 48 GiB card (micro 4 x accum 2 against micro 8 x accum 1). No evidence in
  this round supports spending another hour on drafter depth.
- **"The drafter is undertrained": partially, and with a clear shape.** AL was still rising when the
  run stopped, but the marginal return per unit of wall clock is decaying fast (R3, 64-row periodic
  set, AL per half-epoch): `2.054 -> 2.278 -> 2.312 -> 2.549 -> 2.745 -> 2.868 -> 2.923`, i.e. gains
  of `+0.237, +0.196, +0.123, +0.055` per half-epoch from epoch 1 onwards. The 2.5 -> 3.0 epoch
  half bought **+0.055 AL (~2% relative) for ~67 minutes of one A6000**.
- **"More data would fix it": still unknown, and this round's design cannot say.** The 5K control was
  cancelled by instruction, so nothing here separates "more unique samples" from "more passes over
  the same 20 000". That is the one open question worth GPU time (§9.4).
- **Data quality / supervision is not the limiter at this scale.** 1 701 868 unique supervised
  tokens, 0% truncation, image-disjoint splits, dev->test gap of only -0.05 AL (§3, §6.1): the
  pipeline is not losing signal between train and eval.

### 9.2 Three levers, not one - and why +depth cannot pull any of them

**Correction to an earlier draft of this section.** It previously called AL "the number that decides
everything" and framed the next round as "raise AL". That framing is wrong and is retracted here.
With `A` = tokens committed per verification round, `r` = block-verify time / single-token decode
time and `d` = draft-round time / single-token decode time,

```
S ~= A / (r + d)          (S = decode speedup vs target-only)
```

so **A, r and d are three independent levers**, and a cheaper target verification runtime raises `S`
at *constant* AL. This round only had evidence about `A` (and about depth); it does not license the
claim that AL is the only lever. The instrumented profile now measures `r` and `d` directly
(`scripts/scale/profile_parallel.py`; full numbers in `docs/parallel_verification_results.md`). On
the official 32-prompt parallel run: **`r = 1.000`, `d = 0.447`, `A = 2.874`, `S_model = 1.99`,
`S_wall = 1.86` (R3) / 1.78 (R5)**. Since `r` is at its floor (one verify forward = one decode step),
the speedup reduces to `A / (1 + d)`, and **`d` is worth about as much as `A`**: driving the drafter
to free would give 2.87x, and raising `A` from 2.87 to 4 at the current `d` would give 2.76x. The
remaining AL argument below is kept only because it rules out one specific plan ("train longer, same
recipe"):

`AL = 1 + sum_i p_i` over the per-slot acceptance rates. On the full 256-row dev at `ckpt_final`
(R3): `p = 0.684 0.448 0.290 0.187 0.124 0.080 0.053` -> `sum p_i = 1.87` -> AL 2.86.

- The four tail slots (positions 4-7) contribute **0.44 tokens combined**; the first two contribute
  1.13. Nothing about the deeper arm changed this shape (R5 differs by <=0.01 on every slot).
- Even **perfect** acceptance on the first two draft positions would only give
  `1 + 1 + 1 + 0.29 + 0.19 + 0.12 + 0.08 + 0.05 = 3.73`. **AL >= 4 is not reachable by incremental
  training of this objective** - it needs near-perfect acceptance on the first three positions, i.e.
  a qualitatively better drafter.

- Parallel-decode math: with the provisional `r ~= 1.63`, `drafter_share ~= 0.2` from the 4-prompt dry
  run, **AL 2.8 -> ~1.4x, AL 3.5 -> ~1.8x, AL 4.0 -> ~2.0x, AL 8.0 -> ~4.0x**; with the directly
  measured `r ~= 0.99, d ~= 0.44` the same A gives `S_model ~= 2.2` (the two estimates differ because
  the dry-run `r` was inferred from call savings rather than measured; the direct measurement
  supersedes it). Buying +1.15 A by more training at the *current* marginal
  rate (0.055 AL per half-epoch) would take ~21 half-epochs ~ **10.5 more epochs ~ 23 A6000-hours on
  the same 20 000 rows** - and that rate is itself decaying, so that is a floor, not an estimate.
  This is the concrete evidence that "train the same recipe longer" is not the route to a real
  speedup.

### 9.3 Do first, before any rental (no GPU required, ~1-2 days of engineering)

1. **Make the parallel path token-exact.** It is the only path that can exceed 1.0x, and with
   sequential verification the ceiling is 1.0x by construction (§7.1), so **until this is fixed there
   is no end-to-end speed claim at all** and a larger AL would change nothing. The diagnosis is now
   done rather than hypothetical (§7.3, `docs/parallel_verification_results.md`): the flips are
   **last-ULP bf16 near-ties at the deciding position** (same top-5 candidate set as the sequential
   path, top-2 swapped by 0.0625 at magnitude ~15), they reproduce under both SDPA backends, and
  they are **not** fixed by the strict-margin guard at any threshold: the measured sweep
  (`docs/parallel_verification_results.md` §3.6) runs from 18/32 @ 1.86x (no guard) to the
  all-fallback point 32/32 @ 0.655x - the only exact setting is slower than just running the
  sequential path (0.853x), and `m=0.25` is even *worse* than no guard (16/32 @ 1.21x).
   The work item is therefore specifically "make the *decision* reproduce, not the logits": e.g. an
   fp32 (or fp32-attention) verify head, a per-round token-level re-check restricted to near-ties, or
   accepting and *documenting* an epsilon-exactness contract instead of claiming losslessness.
2. **Disable the 32 GiB host feature cache for both arms** (`train.online_feature_cache_gb: 0`).
   Measured: **3.1% hit rate** (1861 hits / 58 939 misses), 92.5% of stored rows evicted before
   reuse, ~34 GB of host RAM pinned at the cap, `skipped_too_large = 0` - and the corpus would need
   ~0.5 TB to become resident. It contributed to both OOM kills (§8). Removing it is free accuracy-wise
   (3% of target forwards) and is the largest single robustness win available.
3. **Keep the operational guards** that this round added (`--start-attempt`, the
   `committed_state_step()` refusal to restart from zero, `load_previous_evals()`, the watchdog).
   They are what turned 3 hard kills into a finished run, and they are re-usable for any future arm.

### 9.4 What A100 time should buy, if it is bought at all

The only next experiment this round's evidence justifies is a **data-scale test at matched cost**,
because that is the one open variable (§9.1):

The second candidate, now that `d` is measured at 0.447 single-token decodes per round, is a
**drafter-cost experiment at constant AL** - it is worth roughly the same as the whole AL question
(`d -> 0` gives 2.87x at the current A, i.e. +55%). It does not need A100 time either: the drafter is
264.8 M parameters and 13 ms per round on an A6000, so a smaller/cheaper draft forward can be
measured without renting anything. A 2-layer variant already exists as a config knob
(`drafter.num_hidden_layers`), and the training recipe is unchanged - but note this round's evidence
(§6.3) says depth changes moved AL by <=0.02, so the *only* reason to run it is the `d` term, and it
would have to be judged on `A/(r+d)` and wall time, not on AL.

| arm | unique rows | epochs | rows seen | cost |
|---|---|---|---|---|
| A (already on disk, 0 GPU) | 20 000 | 3 | 60 000 | 6.7 h A6000 (403.5 min, measured) |
| B (new) | 60 000 | 1 | 60 000 | ~6.7 h A6000 |

- Same recipe, seed, protocol, block size, feature layers and eval harness; one variable changes.
  If **B >= A**, the drafter is data-limited and scaling data is the lever. If **B << A**, it is
  optimisation/objective-limited and the lever is the drafting objective, not the corpus.
- Budget: 2 arms x ~6.7 h of A6000 compute + ~21 min per full-dev eval + one test eval ~ **15 h of
  one A6000-class GPU, or ~8-10 A100-hours** (assuming roughly 2x for a 80 GB SXM A100 - this round
  did not measure that ratio). **This fits the existing 2 x A6000 budget; renting buys wall clock,
  not capability.**
- Only if B wins does a rented A100 become the right instrument, and then the follow-up is
  "same recipe, 5-10x unique data, 1-2 epochs": 100 000 unique rows ~ 6-12 h per arm on one A100.
  Beyond that, the AL model above says extra data still has to produce `AL ~ 4` to be worth
  anything, so a 100K-row arm should be judged on AL per GPU-hour, not on loss.

Two narrower things A100 VRAM would genuinely buy today, independent of scaling: R3's training peak
was **47.6 GB of a 48 GiB card** (R5: 34.9 GB at micro 4 x accum 2), i.e. the shared recipe has no
VRAM headroom and was competing with a neighbouring agent's 8-11 GiB during this round. 80 GB would
remove that entire class of failure, and would let the feature-cache question be revisited in VRAM
instead of host RAM if it is ever wanted again.

### 9.5 What would make the spend pointless

- Parallel verification still not token-exact -> speedup stays < 1.0x no matter how good the drafter
  becomes, because the only *demonstrated* lossless path is the sequential one, and its ceiling is
  1.0x by construction (§7.1).
- `A` still ~2.9 after an honest data-scale test -> a rented day buys a drafter that commits 2.9
  tokens per round, i.e. `S ~= A/(1+d) ~= 2.0x` (measured 1.86x wall) at best in the parallel path,
  which is a real but modest serving win for a full training round of GPU time.
- Do not spend the next round on: drafter depth (closed here), another hyper-parameter sweep of
  block size / learning rate / anchors (not the binding constraint, and it would confound the
  baseline that this round froze), or a temporarily-added unvalidated parallel scheme (§8).

### 9.6 Reusable assets for the next round

- Frozen data: `shared/regen/freeze_record.json`, `frozen_file_sha256=44e357c7a1ab...`, 20 000 rows /
  11 702 images, image-disjoint train/dev/test, plus the fixed 256-row dev and 64-row periodic subset.
- Trained arms and the selection record: `<run>/driver/selection.json` for both R3 and R5, with
  `ckpt_final.pt`, `ckpt_step0006250.pt` and the stage-boundary `ckpt_step0001875.pt` on disk.
- The eval/analysis harness (`scripts/scale/{speed_paired.py,analyze_paired.py,plot_r3r5_curves.py}`)
  and the report's metric definitions, including the two traps this round documented: the sequential
  path's degenerate `tokens_per_target_forward ~ 1` and the 64-row subset's ~+0.067 AL optimism
  against the full 256-row dev.
