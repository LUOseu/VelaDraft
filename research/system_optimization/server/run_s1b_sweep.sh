#!/bin/bash
# S1b — teacher-forward group sweep on real LLaVA-corpus rows.
# Builds a 256-slice run (~130 kept rows, seq <=2048, one image each) with the
# canonical 2K recipe, then profiles groups 1/2/4/8/16 (render + batched
# forward, CUDA-synced) with scripts/profile_online_extraction.py.
set -u
LANE=/home/lyh/vlm-systems
REPO=/home/lyh/VelaDraft-systems
DEV="${1:?device index}"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_FLEX_FORCE_MAIN=1
export VLM_DFLASH_SKIP_RESUME_CKPT=1
export VLM_DFLASH_LLAVA_JSON=/home/lyh/vlm-systems/data/llava_v1safe.json
export VLM_DFLASH_COCO_IMAGES=/home/dhz/Data/COCO-caption/train2017
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
export CUDA_VISIBLE_DEVICES="$DEV"
RUNS=$LANE/runs/s1b
mkdir -p "$RUNS" "$LANE/logs"
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
R=$RUNS/real256

if [ ! -f "$R/regen/target_data.jsonl" ]; then
  FREE=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); FREE=${FREE:-0}
  if [ "$FREE" -lt 4 ]; then
    echo "$(ts) s1b build SKIPPED: only ${FREE}G free on /home" >> "$LANE/logs/s1b_progress.log"
    exit 1
  fi
  rm -rf "$R"
  echo "$(ts) s1b build slice start (free ${FREE}G)" >> "$LANE/logs/s1b_progress.log"
  $PY -m vlm_dflash.cli train --mode online --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$R" --limit 256 --steps 1 \
    --set train.overfit_sanity.enabled=false --set train.log_every_steps=1 \
    --set train.save_every_steps=100000 \
    > "$LANE/logs/s1b_build.log" 2>&1
  ec=$?
  echo "$(ts) s1b build exit=$ec" >> "$LANE/logs/s1b_progress.log"
  if [ "$ec" -ne 0 ] || [ ! -f "$R/regen/target_data.jsonl" ]; then
    echo "$(ts) s1b build FAILED; skipping profile" >> "$LANE/logs/s1b_progress.log"
    exit 1
  fi
fi
# prune the throwaway 1-step training artifacts, keep data+regen+contract
rm -rf "$R/train"
echo "$(ts) s1b profile start" >> "$LANE/logs/s1b_progress.log"
$PY scripts/profile_online_extraction.py --config configs/qwen2_5_vl_3b_2k.yaml \
  --run-dir "$R" --groups 1,2,4,8,16 --render-repeats 2 --forward-repeats 3 \
  > "$LANE/logs/s1b_profile.json" 2>&1
ec=$?
echo "$(ts) s1b profile exit=$ec" >> "$LANE/logs/s1b_progress.log"
echo "$(ts) S1b DONE" >> "$LANE/logs/s1b_progress.log"
