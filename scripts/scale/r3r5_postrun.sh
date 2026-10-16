#!/bin/bash
# Unattended post-training work: wait for both drivers, then run the
# matched-progress paired analyses (periodic / full-dev confirm / test) and the
# exclusive paired speed measurements on one free GPU.
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
BASE=$ROOT/runs/r3r5_20k
LOGS=$BASE/logs
export PYTHONPATH=$REPO VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %H:%M:%S')] postrun: $*" | tee -a "$LOGS/30_postrun.log"; }
say "watcher up; waiting for the frozen corpus"

for _ in $(seq 1 1440); do   # up to 12h
  [ -f "$BASE/shared/regen/freeze_record.json" ] && break
  sleep 30
done
N=$($PY -c "import json;print(json.load(open('$BASE/shared/regen/freeze_record.json'))['num_frozen'])")
R3=$ROOT/runs/r3r5_n$N/r3
R5=$ROOT/runs/r3r5_n$N/r5
mkdir -p "$BASE/analysis"

for _ in $(seq 1 1440); do
  d3=$(grep -c '"event": "run_complete"' "$R3/driver/progress.jsonl" 2>/dev/null || true)
  d5=$(grep -c '"event": "run_complete"' "$R5/driver/progress.jsonl" 2>/dev/null || true)
  [ "${d3:-0}" -ge 1 ] && [ "${d5:-0}" -ge 1 ] && break
  sleep 60
done
say "drivers finished (r3_complete=${d3:-0} r5_complete=${d5:-0}) N=$N"

# ---- matched-progress paired analyses over every milestone both runs evaluated
for tag in $(ls "$R3/eval" 2>/dev/null | sed -n 's/^eval_samples_\(p[0-9]*\)\.jsonl$/\1/p' | sort); do
  [ -f "$R5/eval/eval_samples_$tag.jsonl" ] || continue
  $PY scripts/scale/analyze_paired.py \
      --samples-a "$R3/eval/eval_samples_$tag.jsonl" --samples-b "$R5/eval/eval_samples_$tag.jsonl" \
      --label-a R3 --label-b R5 --out "$BASE/analysis/paired_periodic_$tag.json" \
      >> "$LOGS/31_analyze.log" 2>&1
done
for tag in $(ls "$R3/eval" 2>/dev/null | sed -n 's/^eval_samples_\(dev_step[0-9]*\|dev_final\)\.jsonl$/\1/p' | sort); do
  [ -f "$R5/eval/eval_samples_$tag.jsonl" ] || continue
  $PY scripts/scale/analyze_paired.py \
      --samples-a "$R3/eval/eval_samples_$tag.jsonl" --samples-b "$R5/eval/eval_samples_$tag.jsonl" \
      --label-a R3 --label-b R5 --out "$BASE/analysis/paired_dev_$tag.json" \
      >> "$LOGS/31_analyze.log" 2>&1
done
say "paired analyses written to $BASE/analysis ($(ls $BASE/analysis 2>/dev/null | wc -l) files)"

# ---- exclusive paired speed measurement on the freest GPU
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
say "speed: using gpu $GPU (free ${FREE}MiB)"
if [ "${FREE:-0}" -ge 40000 ]; then
  for arm in r3 r5; do
    cfg="configs/scale/${arm}_dfly_n$N.yaml"
    rundir="$ROOT/runs/r3r5_n$N/$arm"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/scale/speed_paired.py \
        --config "$cfg" --run-dir "$rundir" --ckpt ckpt_final.pt \
        --num-prompts 32 --warmup-prompts 3 --max-new-tokens 128 \
        --tag "${arm}_speed_final" > "$LOGS/32_speed_$arm.log" 2>&1
    say "speed $arm rc=$?"
  done
else
  say "speed SKIPPED: no GPU with >=40GiB free (best gpu $GPU free ${FREE}MiB)"
fi
say "POSTRUN COMPLETE"
