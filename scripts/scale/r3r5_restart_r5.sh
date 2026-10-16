#!/bin/bash
# Re-launch the R5 run after its process tree was killed (the host OOM killer
# SIGKILLs the trainer; the driver reads that as a plain crash and stops by
# design, and refuses to fall back to a from-scratch geometry).
#
# Resume-only: always attempt 2 of the ladder (`spec-resume`), i.e. the exact
# same micro 4 x accum 2 geometry, continued from the last committed resumable
# state. Nothing about the recipe, data, seed or eval protocol changes.
set -u
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
BASE=/home/lyh/vlm-scale/runs/r3r5_20k
R5=/home/lyh/vlm-scale/runs/r3r5_n20000/r5
STAMP=$(date '+%Y%m%d-%H%M%S')
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %T')] restart_r5: $*" | tee -a "$BASE/logs/70_restarts.log"; }

if pgrep -f "r3r5_train_run.py --config configs/scale/r5_dfly_n20000" > /dev/null 2>&1; then
  say "an R5 driver is already running; refusing to start a second one"
  exit 1
fi
if pgrep -f "train_dflash.py --config configs/scale/r5_dfly_n20000" > /dev/null 2>&1; then
  say "an R5 trainer is already running without a driver; refusing"
  exit 1
fi

STEP=$(tail -1 "$R5/train/train_log.jsonl" 2>/dev/null | "$PY" -c "import json,sys;print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo 0)
say "trainer dead; last logged step=$STEP; relaunching (resume-only)"
nohup setsid "$PY" scripts/scale/r3r5_train_run.py \
  --config configs/scale/r5_dfly_n20000.yaml --run-dir "$R5" \
  --shared-dir "$BASE/shared" --train-gpu 2 --eval-gpu 1 --eval-prompts 64 \
  --oom-fallback \
  --initial-sets train.micro_batch_size=4,train.grad_accumulation_steps=2 \
  --start-attempt 2 \
  > "$BASE/logs/08_restart_r5_$STAMP.log" 2>&1 < /dev/null &
say "launched; log $BASE/logs/08_restart_r5_$STAMP.log"
