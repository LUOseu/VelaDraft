#!/bin/bash
# One-shot status digest for the R3/R5 round (safe to run any time).
BASE=/home/lyh/vlm-scale/runs/r3r5_20k
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
echo "== $(date '+%F %H:%M:%S') =="
echo "-- gpus:"; nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
echo "-- disk /home $(df -h /home | tail -1 | awk '{print $4}')  shm $(df -h /dev/shm | tail -1 | awk '{print $4}')"
echo "-- data stage:"; tail -3 "$BASE/logs/06_finalize.log" 2>/dev/null
grep -E "processed" "$BASE/logs/03a_regen_a.log" 2>/dev/null | tail -1
grep -E "processed" "$BASE/logs/03b_regen_b.log" 2>/dev/null | tail -1
for tag in r3 r5; do
  RUN=$BASE/../r3r5_n*/$tag
  RUN=$(ls -d $RUN 2>/dev/null | head -1)
  [ -z "$RUN" ] && continue
  echo "-- $tag ($RUN):"
  tail -1 "$RUN/driver/progress.jsonl" 2>/dev/null | cut -c1-400
  grep -oE '"(step|loss|grad_norm|lr)": [0-9.e-]+' "$RUN/train/train_log.jsonl" 2>/dev/null | tail -4 | tr '\n' ' '; echo
  ls "$RUN/train" 2>/dev/null | grep -E "ckpt" | tr '\n' ' '; echo
  tail -2 "$RUN/driver/periodic_summary.jsonl" 2>/dev/null
done
echo "-- smoke:"; tail -3 "$BASE/logs/smoke.log" 2>/dev/null
