#!/bin/bash
# S2c — causal check for the H-C1 numeric drift behind the S2 gate failure.
# S2 result: OFF legs and ON legs are each bitwise reproducible, but ON drifts
# from OFF at bf16-rounding scale from the first cache-hit step. Mechanism
# candidate: online extraction is sensitive to batch composition (padding /
# kernel shapes; see probe_record_invariance2), and the shuffle changes the
# grouping every epoch, so the cache freezes epoch-1-era values while the OFF
# leg recomputes under each epoch's own grouping. Prediction: with a fixed
# epoch order (VLM_DFLASH_TRAIN_SHUFFLE=0) the ON leg reproduces OFF bitwise.
set -u
LANE=/home/lyh/vlm-systems
REPO=/home/lyh/VelaDraft-systems
DEV="${1:?device index}"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_FLEX_FORCE_MAIN=1
export VLM_DFLASH_SKIP_RESUME_CKPT=1
export VLM_DFLASH_TRAIN_SHUFFLE=0
export VLM_DFLASH_LLAVA_JSON=/home/lyh/vlm-systems/data/llava_v1safe.json
export VLM_DFLASH_COCO_IMAGES=/home/dhz/Data/COCO-caption/train2017
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
export CUDA_VISIBLE_DEVICES="$DEV"
RUNS=$LANE/runs/s2c
MASTER=$LANE/runs/s1b/real256
mkdir -p "$RUNS" "$LANE/logs"
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
plog() { echo "$(ts) $*" >> "$LANE/logs/s2c_progress.log"; }
free_gb() { local v; v=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); echo "${v:-0}"; }

if [ ! -f "$MASTER/regen/target_data.jsonl" ]; then
  plog "s2c slice missing (master $MASTER)"; exit 1
fi

prep_leg() { rm -rf "$1"; mkdir -p "$1"; cp -a "$MASTER/data" "$MASTER/regen" "$MASTER/contract.json" "$1/"; }

run_leg() {  # $1=dir $2=cache_gb $3=label
  echo "$(ts) $3 start (cache_gb=$2, GPU $DEV, free $(free_gb)G)" >> "$LANE/logs/s2c_progress.log"
  /usr/bin/time -v -o "$LANE/logs/$3.time" \
  $PY -m vlm_dflash.cli train --mode online --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$1" --limit 256 \
    --set train.overfit_sanity.enabled=false \
    --set train.log_every_steps=1 \
    --set train.save_every_steps=100000 \
    --set train.online_batch_size=8 \
    --set train.online_feature_cache_gb="$2" \
    > "$LANE/logs/$3.log" 2>&1
  local ec=$?
  echo "$(ts) $3 exit=$ec (log $LANE/logs/$3.log)" >> "$LANE/logs/s2c_progress.log"
}

if [ "$(free_gb)" -lt 5 ]; then plog "S2c SKIPPED: only $(free_gb)G free on /home"; exit 0; fi
: > "$LANE/logs/s2c_ckpt_hashes.txt"
prep_leg "$RUNS/a"
prep_leg "$RUNS/b"
run_leg "$RUNS/a" 0 "s2c_a_off"
run_leg "$RUNS/b" 8 "s2c_b_on"
ha=$(sha256sum "$RUNS/a/train/ckpt_final.pt" | awk '{print $1}')
hb=$(sha256sum "$RUNS/b/train/ckpt_final.pt" | awk '{print $1}')
echo "$ha  $RUNS/a/train/ckpt_final.pt" >> "$LANE/logs/s2c_ckpt_hashes.txt"
echo "$hb  $RUNS/b/train/ckpt_final.pt" >> "$LANE/logs/s2c_ckpt_hashes.txt"
if [ "$ha" = "$hb" ]; then
  plog "S2c: ckpt sha256 MATCH ($ha) — fixed-order ON reproduces OFF bitwise"
else
  plog "S2c: ckpt sha256 MISMATCH (a=$ha b=$hb) — drift persists under fixed order"
fi
plog "S2c DONE"
