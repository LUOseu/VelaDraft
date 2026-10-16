#!/bin/bash
# R3/R5 round, data stage: prepare -> pilot regen (throughput) -> full regen -> freeze.
# usage: bash r3r5_prepare.sh [prepare|pilot|regen|freeze|stats|all]
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
CFG=$REPO/scripts/scale/r3r5_data_config.yaml
BASE=$ROOT/runs/r3r5_20k
SHARED=$BASE/shared
PILOT=$BASE/regen_pilot
LOGS=$BASE/logs
GPU=${R3R5_GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
mkdir -p "$LOGS"

say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOGS/data_stage.log"; }

guard() {
  local attempt util free dh
  for attempt in $(seq 1 20); do
    util=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
    free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
    dh=$(df -BG /home | tail -1 | awk '{print $4}' | tr -dc 0-9)
    if [ "${util:-100}" -le 5 ] && [ "${free:-0}" -ge 35000 ] && [ "${dh:-0}" -ge 6 ]; then
      say "guard OK gpu=$GPU util=${util}% free=${free}MiB home=${dh}G"
      return 0
    fi
    say "guard wait $attempt/20 gpu=$GPU util=${util}% free=${free}MiB home=${dh}G"
    sleep 30
  done
  say "GUARD FAIL gpu=$GPU util=${util}% free=${free}MiB home=${dh}G"
  return 1
}

STAGE="${1:-all}"
cd "$REPO" || exit 1

if [ "$STAGE" = prepare ] || [ "$STAGE" = all ]; then
  if [ -f "$SHARED/data/data.jsonl" ]; then
    say "prepare already present, skip"
  else
    guard || exit 1
    say "STAGE prepare: 24000 input rows + 256 val + 256 test, image-grouped seed 42, hardlink media"
    rc=0
    $PY scripts/prepare_data.py --config "$CFG" --run-dir "$SHARED" > "$LOGS/01_prepare.log" 2>&1 || rc=$?
    say "prepare rc=$rc (log: logs/01_prepare.log)"
    [ $rc -ne 0 ] && exit 1
    tail -3 "$LOGS/01_prepare.log" | tee -a "$LOGS/data_stage.log"
  fi
fi

if [ "$STAGE" = pilot ] || [ "$STAGE" = all ]; then
  if [ -f "$PILOT/regen/target_data.jsonl" ] && [ "$STAGE" = all ]; then
    say "pilot already present, skip"
  else
    guard || exit 1
    mkdir -p "$PILOT"
    [ -d "$PILOT/data" ] || cp -al "$SHARED/data" "$PILOT/data"
    say "STAGE pilot: first 1200 rows, batch 16, greedy, max_new 256 (throughput + keep-rate probe)"
    t0=$(date +%s)
    rc=0
    $PY scripts/generate_target_data.py --config "$CFG" --run-dir "$PILOT" --limit 1200 > "$LOGS/02_pilot.log" 2>&1 || rc=$?
    t1=$(date +%s)
    kept=$(wc -l < "$PILOT/regen/target_data.jsonl" 2>/dev/null || echo 0)
    say "pilot rc=$rc kept=$kept wall=$((t1-t0))s ($(( (t1-t0)*1000/1200 )) ms/row) (log: logs/02_pilot.log)"
    [ $rc -ne 0 ] && exit 1
    $PY - "$PILOT/regen/regen_stats.json" >> "$LOGS/data_stage.log" <<'PYEOF'
import json, sys
stats = json.load(open(sys.argv[1]))
hist = stats["answer_token_histogram"]
print(f"  pilot stats: kept={stats['num_kept']}/{stats['num_input']} "
      f"({stats['num_kept']/max(1,stats['num_input'])*100:.1f}%) "
      f"answer len min={hist['min']} p10={hist['p10']} median={hist['median']} p90={hist['p90']} "
      f"max={hist['max']} at_cap={hist.get('at_cap')} ({hist.get('at_cap_fraction',0)*100:.1f}%)")
PYEOF
  fi
fi

if [ "$STAGE" = regen ] || [ "$STAGE" = all ]; then
  if [ -f "$SHARED/regen/target_data.jsonl" ]; then
    say "regen already present, skip"
  else
    guard || exit 1
    say "STAGE regen: 24000 rows, batch 16, greedy, max_new 256, min_response 8"
    rc=0
    $PY scripts/generate_target_data.py --config "$CFG" --run-dir "$SHARED" > "$LOGS/03_regen.log" 2>&1 || rc=$?
    kept=$(wc -l < "$SHARED/regen/target_data.jsonl" 2>/dev/null || echo 0)
    say "regen rc=$rc kept=$kept (log: logs/03_regen.log)"
    [ $rc -ne 0 ] && exit 1
  fi
fi

if [ "$STAGE" = freeze ] || [ "$STAGE" = all ]; then
  if [ -f "$SHARED/regen/freeze_record.json" ]; then
    say "freeze already present, skip"
  else
    say "STAGE freeze: N = min(20000, kept), deterministic subset (seed 42)"
    rc=0
    $PY scripts/scale/freeze_train_rows.py --run-dir "$SHARED" --target 20000 --floor 15000 \
        > "$LOGS/04_freeze.log" 2>&1 || rc=$?
    say "freeze rc=$rc (log: logs/04_freeze.log)"
    [ $rc -ne 0 ] && exit 1
    tail -5 "$LOGS/04_freeze.log" | tee -a "$LOGS/data_stage.log"
  fi
fi

if [ "$STAGE" = stats ] || [ "$STAGE" = all ]; then
  say "STAGE stats: dataset statistics -> $SHARED/data_stats.json"
  rc=0
  $PY scripts/scale/dataset_stats.py --run-dir "$SHARED" --config "$CFG" \
      > "$LOGS/05_stats.log" 2>&1 || rc=$?
  say "stats rc=$rc (log: logs/05_stats.log)"
  [ $rc -ne 0 ] && exit 1
  tail -20 "$LOGS/05_stats.log" | tee -a "$LOGS/data_stage.log"
fi

say "DATA STAGE COMPLETE (stage=$STAGE)"
