#!/bin/bash
# S2 — H-C1 (host-RAM record cache) paired A/B on a real LLaVA slice.
# Six-epoch online runs, cache OFF vs ON, interleaved (off/on x3), same
# data+regen+seed. Numeric gate: sha256(ckpt_final.pt) equality between each
# paired off/on leg + full loss sequence from train_log.jsonl. Host RSS per
# leg via /usr/bin/time -v. Disk-frugal: after each pair's hashes are
# recorded, matched leg checkpoints (incl. 975 MB ckpt_init) are pruned;
# mismatched pairs are kept for debugging. Disk-guarded at 5 GiB (transient ckpt pairs ~3 GiB with
# resumable-checkpoint writes disabled).
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
RUNS=$LANE/runs/s2
MASTER=$LANE/runs/s1b/real256
mkdir -p "$RUNS" "$LANE/logs"
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
plog() { echo "$(ts) $*" >> "$LANE/logs/s2_progress.log"; }
free_gb() { local v; v=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); echo "${v:-0}"; }

if [ ! -f "$MASTER/regen/target_data.jsonl" ]; then
  plog "s2 slice build start (master $MASTER)"
  rm -rf "$MASTER"
  $PY -m vlm_dflash.cli train --mode online --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$MASTER" --limit 256 --steps 1 \
    --set train.overfit_sanity.enabled=false --set train.log_every_steps=1 \
    --set train.save_every_steps=100000 > "$LANE/logs/s2_build.log" 2>&1
  ec=$?
  plog "s2 slice build exit=$ec"
  rm -rf "$MASTER/train"
  if [ "$ec" -ne 0 ] || [ ! -f "$MASTER/regen/target_data.jsonl" ]; then
    plog "s2 slice build FAILED; aborting pairs"
    exit 1
  fi
fi

prep_leg() { rm -rf "$1"; mkdir -p "$1"; cp -a "$MASTER/data" "$MASTER/regen" "$MASTER/contract.json" "$1/"; }

run_leg() {  # $1=dir $2=cache_gb $3=label
  echo "$(ts) $3 start (cache_gb=$2, GPU $DEV, free $(free_gb)G)" >> "$LANE/logs/s2_progress.log"
  /usr/bin/time -v -o "$LANE/logs/$3.time" \
  $PY -m vlm_dflash.cli train --mode online --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$1" --limit 256 \
    --set train.overfit_sanity.enabled=false \
    --set train.log_every_steps=8 \
    --set train.save_every_steps=100000 \
    --set train.online_batch_size=8 \
    --set train.online_feature_cache_gb="$2" \
    > "$LANE/logs/$3.log" 2>&1
  local ec=$?
  echo "$(ts) $3 exit=$ec (log $LANE/logs/$3.log)" >> "$LANE/logs/s2_progress.log"
}

harvest_pair() {  # $1=off tag $2=on tag ; records evidence and prunes matched ckpts
  local off="$1" on="$2" a="$RUNS/$1" b="$RUNS/$2"
  for leg in "$1" "$2"; do
    d="$RUNS/$leg"
    [ -f "$d/train/train_summary.json" ] && cp "$d/train/train_summary.json" "$LANE/logs/s2_${leg}_summary.json"
    [ -f "$d/train/train_log.jsonl" ] && cp "$d/train/train_log.jsonl" "$LANE/logs/s2_${leg}_train_log.jsonl"
  done
  local match=1
  for ck in ckpt_final.pt ckpt_init.pt; do
    ha=$(sha256sum "$a/train/$ck" 2>/dev/null | awk '{print $1}')
    hb=$(sha256sum "$b/train/$ck" 2>/dev/null | awk '{print $1}')
    [ -n "$ha" ] && echo "$ha  $a/train/$ck" >> "$LANE/logs/s2_ckpt_hashes.txt"
    [ -n "$hb" ] && echo "$hb  $b/train/$ck" >> "$LANE/logs/s2_ckpt_hashes.txt"
    if [ -z "$ha" ] || [ "$ha" != "$hb" ]; then match=0; fi
  done
  rm -f "$a/train/ckpt_last.pt" "$b/train/ckpt_last.pt"
  if [ "$match" -eq 1 ]; then
    rm -f "$a/train/ckpt_final.pt" "$b/train/ckpt_final.pt" "$a/train/ckpt_init.pt" "$b/train/ckpt_init.pt"
    plog "pair $off/$on: ckpt sha256 matched (final+init); pruned all four"
  else
    plog "pair $off/$on: ckpt sha256 MISMATCH; files kept for debugging"
  fi
}

run_pair() {  # $1=index
  local i="$1"
  if [ "$(free_gb)" -lt 5 ]; then plog "pair $i SKIPPED: only $(free_gb)G free"; return 0; fi
  prep_leg "$RUNS/a$i"; prep_leg "$RUNS/b$i"
  run_leg "$RUNS/a$i" 0 "s2_a${i}_off"
  run_leg "$RUNS/b$i" 8 "s2_b${i}_on"
  harvest_pair "a$i" "b$i"
}

if [ "$(free_gb)" -lt 5 ]; then plog "S2 SKIPPED: only $(free_gb)G free on /home"; exit 0; fi
: > "$LANE/logs/s2_ckpt_hashes.txt"
run_pair 1
run_pair 2
run_pair 3
plog "S2 DONE"
