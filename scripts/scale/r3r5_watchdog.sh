#!/bin/bash
# The host is in an OOM war with other tenants whose processes routinely reach
# 78-94 GB RSS; our two trainers (38-46 GB each) get SIGKILLed every so often and
# the driver then stops by design. Waiting for a human (or an agent) to notice
# costs the steps since the last 0.5-epoch save. This watchdog resumes whichever
# arm lost its process tree, using the resume-only path, and never starts a
# duplicate.
#
# Safety rules:
#   * only acts when BOTH the driver and the trainer of an arm are gone,
#   * never acts once the arm logged run_complete (post-training work is the
#     driver's job, and a finished run must not be restarted),
#   * never touches a run whose process tree is alive,
#   * every action is appended to logs/70_restarts.log.
set -u
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
BASE=/home/lyh/vlm-scale/runs/r3r5_20k
LOGS=$BASE/logs
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %T')] watchdog: $*" >> "$LOGS/70_restarts.log"; }

restart_arm() {  # $1 = r3|r5
  local arm=$1 run=$BASE/../r3r5_n20000/$1
  local sets=""
  if [ "$arm" = "r5" ]; then
    sets="--initial-sets train.micro_batch_size=4,train.grad_accumulation_steps=2"
  fi
  local stamp; stamp=$(date '+%Y%m%d-%H%M%S')
  local step; step=$(tail -1 "$run/train/train_log.jsonl" 2>/dev/null \
      | "$PY" -c "import json,sys;print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo 0)
  say "$arm died (last logged step $step); resuming from the last committed state"
  # shellcheck disable=SC2086
  nohup setsid "$PY" scripts/scale/r3r5_train_run.py \
    --config "configs/scale/${arm}_dfly_n20000.yaml" --run-dir "$run" \
    --shared-dir "$BASE/shared" --train-gpu "$([ "$arm" = r3 ] && echo 0 || echo 2)" \
    --eval-gpu "$([ "$arm" = r3 ] && echo 3 || echo 1)" --eval-prompts 64 \
    --oom-fallback $sets --start-attempt 2 \
    > "$LOGS/08_restart_${arm}_$stamp.log" 2>&1 < /dev/null &
  say "$arm relaunched; log $LOGS/08_restart_${arm}_$stamp.log"
}

say "watchdog up (interval 120s)"
while true; do
  for arm in r3 r5; do
    run=$BASE/../r3r5_n20000/$arm
    [ -d "$run" ] || continue
    cfgname="configs/scale/${arm}_dfly_n20000"
    if pgrep -f "r3r5_train_run.py --config $cfgname" > /dev/null 2>&1; then continue; fi
    if pgrep -f "train_dflash.py --config $cfgname" > /dev/null 2>&1; then continue; fi
    if grep -q '"event": "run_complete"' "$run/driver/progress.jsonl" 2>/dev/null; then continue; fi
    if grep -q '"event": "train_failed_giving_up"' "$run/driver/progress.jsonl" 2>/dev/null; then
      say "$arm gave up on a non-OOM failure earlier; NOT auto-restarting"
      continue
    fi
    restart_arm "$arm"
    sleep 180   # let the new driver come up before the next evaluation
  done
  sleep 120
done
