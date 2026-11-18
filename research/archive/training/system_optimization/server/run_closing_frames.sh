#!/bin/bash
# Closing frames for H-C1 after S3 (REPORT 4.5; STATE 19:30 section).
#   Step 1 (~0.2-0.4 GPU-h, read-only on ckpts): acceptance eval of the S3
#     final ckpts runs/s3/{a_off,b_on} via `vlm_dflash.cli eval` with
#     --checkpoints ckpt_final.pt --tag closing (init was pruned after the
#     sha-match). Output: <run>/eval/eval_result_closing.json + logs.
#   Step 2 (~0.7 GPU-h, diagnostic): seed-change frame - cache OFF and ON at
#     train.seed=43 (staged inputs copied from a_off; the seed-43 init differs
#     from the seed-42 legs by design, identical between c and d). wdiff/ulp
#     of (a,c) = order+init variation, (c,d) = cache effect at seed 43.
# Idempotent per step; >=5 GiB free guard before each training leg.
set -u
LANE=/home/lyh/vlm-systems
REPO=/home/lyh/VelaDraft-systems
DEV="${1:?device index}"
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
LOG=$LANE/logs
RUNS=$LANE/runs/s3
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_FLEX_FORCE_MAIN=1
export VLM_DFLASH_SKIP_RESUME_CKPT=1
export VLM_DFLASH_LLAVA_JSON=/home/lyh/vlm-systems/data/llava_v1safe.json
export VLM_DFLASH_COCO_IMAGES=/home/dhz/Data/COCO-caption/train2017
export CUDA_VISIBLE_DEVICES="$DEV"
plog() { echo "$(date +'[%F %T]') $*" >> "$LOG/closing_progress.log"; }
free_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

plog "closing frames start on GPU $DEV"

# ---- step 1: acceptance eval of the S3 pair ----
for leg in a_off b_on; do
  d="$RUNS/$leg"
  if [ -f "$d/eval/eval_result_closing.json" ]; then plog "eval $leg already present; skip"; continue; fi
  plog "eval $leg start"
  $PY -m vlm_dflash.cli eval "$d" --checkpoints ckpt_final.pt --tag closing \
      > "$LOG/closing_eval_${leg}.log" 2>&1
  ec=$?
  plog "eval $leg exit=$ec (log closing_eval_${leg}.log)"
  tail -2 "$LOG/closing_eval_${leg}.log" | sed 's/^/    /' >> "$LOG/closing_progress.log"
done

# ---- step 2: seed-change frame (off/on at train.seed=43) ----
C=$RUNS/c_off_seed43
D=$RUNS/d_on_seed43
run_leg() { # $1 dir $2 cache_gb $3 label
  local dir="$1" cache="$2" label="$3"
  if [ -f "$dir/train/train_summary.json" ]; then plog "$label already done; skip"; return 0; fi
  if [ "$(free_gb)" -lt 5 ]; then plog "$label skipped: <5G free"; return 1; fi
  mkdir -p "$dir"
  if [ ! -f "$dir/contract.json" ]; then
    cp -a "$RUNS/a_off/data" "$RUNS/a_off/regen" "$RUNS/a_off/contract.json" "$dir/" 2>>"$LOG/closing_progress.log"
  fi
  rm -rf "$dir/train"
  plog "$label start (cache_gb=$cache, free $(free_gb)G)"
  /usr/bin/time -v -o "$LOG/$label.time" \
  $PY -m vlm_dflash.cli train --mode online --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$dir" \
    --set train.seed=43 \
    --set train.overfit_sanity.enabled=false \
    --set train.log_every_steps=10 \
    --set train.save_every_steps=100000 \
    --set train.online_batch_size=8 \
    --set train.online_feature_cache_gb="$cache" \
    > "$LOG/$label.log" 2>&1
  plog "$label exit=$?"
}
run_leg "$C" 0 closing_c_seed43_off
run_leg "$D" 16 closing_d_seed43_on

for leg in c_off_seed43 d_on_seed43; do
  d="$RUNS/$leg"
  [ -f "$d/train/train_summary.json" ] && cp "$d/train/train_summary.json" "$LOG/s3_${leg}_summary.json"
  [ -f "$d/train/train_log.jsonl" ] && cp "$d/train/train_log.jsonl" "$LOG/s3_${leg}_train_log.jsonl"
  [ -f "$d/train/ckpt_init.pt" ] && sha256sum "$d/train/ckpt_init.pt" >> "$LOG/s3_seed43_ckpt_hashes.txt"
  [ -f "$d/train/ckpt_final.pt" ] && sha256sum "$d/train/ckpt_final.pt" >> "$LOG/s3_seed43_ckpt_hashes.txt"
done

if [ -f "$C/train/ckpt_final.pt" ] && [ -f "$D/train/ckpt_final.pt" ]; then
  for pair in "a_off:$RUNS/a_off/train/ckpt_final.pt:$C/train/ckpt_final.pt:ac" \
              "cd:$C/train/ckpt_final.pt:$D/train/ckpt_final.pt:cd"; do
    name="${pair%%:*}"; rest="${pair#*:}"; pa="${rest%%:*}"; rest="${rest#*:}"; pb="${rest%%:*}"; tag="${rest#*:}"
    $PY "$LANE/scripts/wdiff_ckpt.py" "$pa" "$pb" > "$LOG/wdiff_seed43_${tag}.txt" 2>&1
    $PY "$LANE/scripts/ulp_drift.py" "$pa" "$pb" --json "$LOG/ulp_seed43_${tag}.json" > "$LOG/ulp_seed43_${tag}.txt" 2>&1
  done
  plog "seed43 diffs done: c-vs-d (cache effect @43), a-vs-c (order+init variation)"
fi
plog "closing frames DONE on GPU $DEV"
