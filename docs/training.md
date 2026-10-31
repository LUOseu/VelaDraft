# Training runs: launch, monitor, resume

For a first small run see [getting-started.md](getting-started.md); for data
preparation rules see [data.md](data.md). This page covers real training runs -
including long ones on a server.

## What a run does

`vlm-dflash train` runs the staged pipeline in one process on **one GPU**:

```
prepare -> regen (target answers) -> capture (offline features) -> train
```

`--mode online` replaces the capture stage with live extraction inside the
trainer (see [online-training.md](online-training.md)); `--features <store>`
skips regen+capture and trains from an existing store. There is no
distributed/DDP launch path; the device table in
[online-training.md#devices-what-is-supported-today](online-training.md)
covers the one supported multi-GPU layout (target and drafter co-located on two
cards in the same process).

## Launch a full recipe

Point the config at your local target and data, then start the reference 2K
recipe (adjust `--run-dir` per method):

```bash
export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_LLAVA_JSON=/path/to/llava_instruct_150k.json
export VLM_DFLASH_COCO_IMAGES=/path/to/coco/train2017

vlm-dflash train \
  --config configs/dfly/qwen2_5_vl_3b_2k.yaml \
  --run-dir runs/dfly_2k
```

For a long run on a server, start it detached and keep a log file:

```bash
mkdir -p logs
nohup vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml \
  --run-dir runs/dfly_2k > logs/dfly_2k.log 2>&1 &
tail -f logs/dfly_2k.log
```

## How long does it run? (stop conditions)

There is **no wall-clock timeout**: a run stops when one of these happens.

1. **The step budget is reached.** The budget is either an explicit `--steps`
   (`train.steps` equivalent) or computed from the data and batch geometry:
   `num_epochs x ceil(ceil(len(train_rows) / micro_batch_size) / grad_accumulation_steps)`.
   `--steps` counts **optimizer steps**; the overfit-sanity phase, capture and
   eval are extra. Do not fake "infinite training" with a huge step count -
   the LR schedule derives from the planned total.
2. **The stall guard trips**: three consecutive epochs in which every batch
   raised `NoValidAnchorsError` (no row has enough supervised tokens) stop the
   run with a message.
3. **A failure or your own kill.** The last checkpoint and logs survive; see
   *Resume*.

Reference scale (Qwen2.5-VL-3B, 2K corpus, 792 optimizer steps = 6 epochs x
132 steps at micro-batch 8 x accumulation 1): **main training loop ~10-24 min**
on a shared A6000, full pipeline including regen/capture/eval ~65-80 min. This
is the recorded example scale - not a validated 50K long run. Numbers and
caveats: [training-costs.md](training-costs.md).

## What you see while it runs

- **stdout/stderr**: stage logs (`prepare_data: ...`, `regen: kept ...`,
  `train: wrote resumable checkpoint (step N)`).
- `<run_dir>/train/train_log.jsonl`: one JSON row per logged optimizer step
  (`step`, `loss`, `accuracy`, `accept_len`, `lr`, `grad_norm`, ...).
- `<run_dir>/train/train_summary.json`: final summary (steps, losses, wall
  time, peak allocated GPU memory, sanity result, dtypes, feature source).
- `<run_dir>/provenance.json` and `<run_dir>/<stage>/stage_manifest.json`:
  digests of every stage input/output (same inputs -> stage skipped on rerun;
  `--force` rebuilds).

## Checkpoints

Written under `<run_dir>/train/`:

| File | Meaning |
| --- | --- |
| `ckpt_init.pt` | random-init baseline (kept for the parity eval; never regenerated) |
| `ckpt_last.pt` | resumable state (model + optimizer + scheduler + RNG), updated every `train.save_every_steps` (default: total_steps / 4) and at completion |
| `ckpt_final.pt` | final trained weights |

The overfit-sanity gate writes `<run_dir>/train/overfit_sanity.json`; it is
recorded, not enforced - a `"passed": false` there is a signal, not a crash.

## Resume

```bash
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml \
  --run-dir runs/dfly_2k --resume
```

`--resume` is an **exact** resume: re-issue the original command (same config
file, same `--limit` / `--features` / `--mode`, same `--set` overrides) and add
`--resume`. Before writing anything, the preflight validates, read-only:

- the target identity and the feature-source digest (data/regen/capture or the
  online source digest),
- the resolved training plan (seed, batch geometry, epochs/steps, optimizer,
  schedule, dtype, drafter and algorithm knobs), and
- the stage dependency chain for the mode actually used.

Changing the seed, step budget, recipe, batch geometry or feature source after
a checkpoint exists is **refused** (loudly, with the mismatch named) - those
change the planned trajectory, so exact resume cannot honor them. The refusal
leaves the run directory byte-identical; keep the original command around (or
your launch script) so a retry is trivial. Checkpoints written before the
plan-digest scheme exist are not resumable at all - start a new run directory
instead of re-stamping old files.

Interruptions are safe: a `SIGKILL` mid-epoch resumes from the committed
`ckpt_last.pt`; log rows past the checkpoint are moved to
`train_log.uncommitted.jsonl` and replayed, so `train_log.jsonl` keeps unique,
ordered steps.

## After training

```bash
vlm-dflash eval runs/dfly_2k --num-prompts 128 --max-new-tokens 128   # sequential (token-exact) by default
vlm-dflash export runs/dfly_2k                                        # self-describing portable checkpoint
```

`eval` needs the target model (it verifies against target greedy decoding) and
writes `<run_dir>/eval/eval_result.json`; `export` writes
`<run_dir>/export/` with weights, configs, contract, algorithm record,
`model_card.md` and checksums. Reading the numbers: [../BENCHMARKS.md](../BENCHMARKS.md)
and [verification-modes.md](verification-modes.md).

## Scoring memory and repeated teacher work

For opt-in chunked CE, frozen-head gradient precomputation, and bounded teacher
projection reuse, see [memory-bounded scoring](training-scoring.md). These switches
have different objective support and memory/time tradeoffs; defaults are unchanged.
