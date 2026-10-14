#!/bin/bash
# S1a — unified-boundary paired re-measurement on the pinned WP2/WP4 smoke
# slice (16 synthetic 2-image/2-turn rows, seq 1024, Qwen2.5-VL-3B target).
# Purpose: reproduce WP4's A/B (per-sample vs capture group 4) on post-R2 code
# (prepare_seconds / forward_seconds split, CUDA-synced), plus a resume leg.
# Legs use save_every_steps=100000 so resumable-checkpoint writes (28-49 s on
# this disk) do not dominate the wall; both legs identical.
set -u
LANE=/home/lyh/vlm-systems
REPO=/home/lyh/VelaDraft-systems
DEV="${1:?device index}"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_FLEX_FORCE_MAIN=1
export VLM_DFLASH_SKIP_RESUME_CKPT=1
export VLM_DFLASH_E2_SOURCE=$LANE/data/e2_smoke/source.jsonl
export VLM_DFLASH_E2_IMAGES=$LANE/data/e2_smoke/images
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
export CUDA_VISIBLE_DEVICES="$DEV"
RUNS=$LANE/runs/s1a
mkdir -p "$RUNS" "$LANE/logs" "$LANE/data"
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }

if [ ! -f "$VLM_DFLASH_E2_SOURCE" ]; then
  $PY scripts/make_wp2_smoke_data.py --out "$LANE/data/e2_smoke" > "$LANE/logs/s1a_gen.log" 2>&1 \
    || { echo "$(ts) data gen FAILED" >> "$LANE/logs/s1a_progress.log"; exit 1; }
fi

run_leg() {  # $1=run dir  $2=extra args  $3=label; caller stages/cleans the dir
  local dir="$1"; shift; local extra="$1"; shift; local label="$1"; shift
  local free; free=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); free=${free:-0}
  if [ "$free" -lt 4 ]; then
    echo "$(ts) $label SKIPPED: only ${free}G free on /home" >> "$LANE/logs/s1a_progress.log"
    return 1
  fi
  echo "$(ts) $label start (GPU $DEV, free ${free}G)" >> "$LANE/logs/s1a_progress.log"
  $PY -m vlm_dflash.cli train --mode online --config configs/smoke_multiturn.yaml \
    --run-dir "$dir" --steps 4 \
    --set train.log_every_steps=1 --set train.save_every_steps=100000 $extra \
    > "$LANE/logs/${label}.log" 2>&1
  local ec=$?
  echo "$(ts) $label exit=$ec (log $LANE/logs/${label}.log)" >> "$LANE/logs/s1a_progress.log"
}

# Leg A: per-sample reference (online_batch_size=1, the default)
rm -rf "$RUNS/ab1"
run_leg "$RUNS/ab1" "" "s1a_ab1"
# Leg B: capture group 4; reuse A's staged data+regen
rm -rf "$RUNS/ab4"; mkdir -p "$RUNS/ab4"
cp -a "$RUNS/ab1/data" "$RUNS/ab1/regen" "$RUNS/ab1/contract.json" "$RUNS/ab4/" 2>/dev/null
run_leg "$RUNS/ab4" "--set train.online_batch_size=4" "s1a_ab4"
# Harvest summaries; the R2 re-measurement needs only the summaries, so prune
# all per-leg checkpoints (~1.5 GB/leg) to protect the disk for S2/S3 guards.
for d in ab1 ab4; do
  [ -f "$RUNS/$d/train/train_summary.json" ] && cp "$RUNS/$d/train/train_summary.json" "$LANE/logs/s1a_${d}_summary.json"
  rm -f "$RUNS/$d/train/ckpt_last.pt" "$RUNS/$d/train/ckpt_init.pt" "$RUNS/$d/train/ckpt_final.pt"
done
echo "$(ts) S1a DONE" >> "$LANE/logs/s1a_progress.log"
