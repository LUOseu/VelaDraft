#!/bin/bash
# Wait for both regen chunks, then merge -> freeze -> emit run configs -> stats.
# Runs unattended so the data stage is ready the moment the chunks finish.
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
CFG=$REPO/scripts/scale/r3r5_data_config.yaml
BASE=$ROOT/runs/r3r5_20k
SHARED=$BASE/shared
CHUNK_B=$BASE/chunks/b
LOGS=$BASE/logs
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %H:%M:%S')] finalize: $*" | tee -a "$LOGS/06_finalize.log"; }

wait_for() {  # $1 = path, $2 = label
  for _ in $(seq 1 2880); do   # up to 8h
    if [ -f "$1" ] && ! pgrep -f "generate_target_data.py .*--run-dir $(dirname $(dirname $1))" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  say "TIMEOUT waiting for $2"; return 1
}

wait_for "$SHARED/regen/target_data.jsonl" "chunk A" || exit 1
wait_for "$CHUNK_B/regen/target_data.jsonl" "chunk B" || exit 1
sleep 20
say "both chunks present; merging"
$PY scripts/scale/merge_regen_chunks.py --shared-dir "$SHARED" --config "$CFG" \
    --chunk "$SHARED" --chunk "$CHUNK_B" > "$LOGS/06_merge.log" 2>&1 || { say "MERGE FAILED"; exit 1; }
tail -2 "$LOGS/06_merge.log" | tee -a "$LOGS/06_finalize.log"

say "freezing corpus (target 20000, floor 15000)"
$PY scripts/scale/freeze_train_rows.py --run-dir "$SHARED" --target 20000 --floor 15000 \
    > "$LOGS/04_freeze.log" 2>&1 || { say "FREEZE FAILED"; exit 1; }
tail -2 "$LOGS/04_freeze.log" | tee -a "$LOGS/06_finalize.log"
N=$($PY -c "import json;print(json.load(open('$SHARED/regen/freeze_record.json'))['num_frozen'])")

say "emitting run configs for N=$N"
$PY scripts/scale/emit_run_configs.py --n "$N" > "$LOGS/07_emit.log" 2>&1 || { say "EMIT FAILED"; exit 1; }
tail -1 "$LOGS/07_emit.log" | tee -a "$LOGS/06_finalize.log"

say "compiling dataset stats"
$PY scripts/scale/dataset_stats.py --run-dir "$SHARED" --config "$CFG" > "$LOGS/05_stats.log" 2>&1 \
    || say "STATS FAILED (non-blocking)"
say "DATA STAGE FINALIZED N=$N"
