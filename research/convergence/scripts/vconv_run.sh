#!/bin/bash
# vconv launcher: guarded train/eval for one leg.
# usage: vconv_run.sh <LEG> <train|eval> [extra args...]
# env: VCONV_GPU (default 0), VCONV_STEPS (default 800), VCONV_REPO
set -u
LEG=$1; MODE=$2; shift 2
GPU=${VCONV_GPU:-0}
RUN=/home/lyh/veladraft-conv/runs/$LEG
LOG=/home/lyh/veladraft-conv/logs/vconv_${LEG}_${MODE}.log
REPO=${VCONV_REPO:-/home/lyh/veladraft-conv/repo}
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
cd "$REPO" || exit 1
ok=0
for attempt in 1 2 3 4 5 6; do
  util=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits)
  used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  dh=$(df -BG /home | tail -1 | awk '{print $4}' | tr -dc '0-9')
  echo "guard attempt=$attempt $(date '+%F %H:%M:%S') gpu=$GPU util=${util}% free=${free}MiB used=${used}MiB home=${dh}G" >> "$LOG"
  if [ "$util" -le 5 ] && [ "$free" -ge 40960 ] && [ "$dh" -ge 15 ]; then ok=1; break; fi
  if [ "$used" -le 200 ]; then echo "transient spike -> retry in 20s" >> "$LOG"; sleep 20; continue; fi
  echo "GUARD FAIL (tenant) util=$util free=$free used=$used" >> "$LOG"; exit 1
done
[ "$ok" -eq 1 ] || { echo "GUARD FAIL after retries $(date '+%F %H:%M:%S')" >> "$LOG"; exit 1; }
{
  echo "=== launch $(date '+%F %H:%M:%S') gpu=$GPU leg=$LEG mode=$MODE"
  echo "repo_head: $(git -C "$REPO" rev-parse HEAD)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
} >> "$LOG"
if [ "$MODE" = train ]; then
  "$PY" scripts/train_dflash.py --config /home/lyh/veladraft-conv/config/e2e_w3_conv.json --run-dir "$RUN" \
    --steps "${VCONV_STEPS:-800}" --skip-sanity \
    --set train.feature_mode=offline --set train.log_every_steps=1 --set train.num_epochs=20 "$@" >> "$LOG" 2>&1
else
  "$PY" scripts/eval_dflash.py --config /home/lyh/veladraft-conv/config/e2e_w3_conv.json --run-dir "$RUN" \
    --num-prompts 64 --set eval.max_new_tokens=128 "$@" >> "$LOG" 2>&1
fi
rc=$?
echo "=== exit=$rc $(date '+%F %H:%M:%S') ===" >> "$LOG"
exit $rc
