#!/bin/bash
# Unattended round starter: wait for the frozen corpus, then run the startup
# smoke and launch both training drivers with the planned GPU split:
#   smoke -> GPU_SMOKE (default 2)  |  R3 train -> GPU 0 (eval GPU 2)
#   R5 train -> GPU_SMOKE once the smoke has released the card (eval GPU 1)
# Everything is nohup'd; the script itself can be killed without killing the
# drivers (they are re-parented via setsid).
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
BASE=$ROOT/runs/r3r5_20k
SHARED=$BASE/shared
RUNS=$ROOT/runs
LOGS=$BASE/logs
GPU_R3=${GPU_R3:-0}
GPU_R5=${GPU_R5:-2}
GPU_EVAL_R3=${GPU_EVAL_R3:-2}
GPU_EVAL_R5=${GPU_EVAL_R5:-1}
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %H:%M:%S')] start_round: $*" | tee -a "$LOGS/08_start_round.log"; }

for _ in $(seq 1 1080); do
  [ -f "$SHARED/regen/freeze_record.json" ] && break
  sleep 10
done
if [ ! -f "$SHARED/regen/freeze_record.json" ]; then say "TIMEOUT waiting for freeze"; exit 1; fi
N=$($PY -c "import json;print(json.load(open('$SHARED/regen/freeze_record.json'))['num_frozen'])")
say "frozen corpus N=$N; launching smoke on gpu $GPU_R5 then R3/R5 drivers"

SMOKE_GPU=$GPU_R5 nohup bash scripts/scale/r3r5_smoke.sh all > "$LOGS/09_smoke_nohup.log" 2>&1 &
SMOKE_PID=$!
say "smoke pid=$SMOKE_PID on gpu $GPU_R5"

sleep 120   # let the smoke take the card first; the R3 guard waits for gpu 0 anyway
setsid nohup $PY scripts/scale/r3r5_train_run.py \
  --config "configs/scale/r3_dfly_n$N.yaml" \
  --run-dir "$RUNS/r3r5_n$N/r3" --shared-dir "$SHARED" \
  --train-gpu "$GPU_R3" --eval-gpu "$GPU_EVAL_R3" --eval-prompts 64 --oom-fallback \
  > "$LOGS/20_r3_driver.log" 2>&1 &
say "R3 driver pid=$! train_gpu=$GPU_R3 eval_gpu=$GPU_EVAL_R3"

wait $SMOKE_PID
say "smoke exited rc=$?"
sleep 20
setsid nohup $PY scripts/scale/r3r5_train_run.py \
  --config "configs/scale/r5_dfly_n$N.yaml" \
  --run-dir "$RUNS/r3r5_n$N/r5" --shared-dir "$SHARED" \
  --train-gpu "$GPU_R5" --eval-gpu "$GPU_EVAL_R5" --eval-prompts 64 --oom-fallback \
  > "$LOGS/21_r5_driver.log" 2>&1 &
say "R5 driver pid=$! train_gpu=$GPU_R5 eval_gpu=$GPU_EVAL_R5"
say "ROUND LAUNCHED N=$N"
