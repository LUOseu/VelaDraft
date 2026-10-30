#!/bin/bash
# One lane worker: claims queued methods (atomic mv) and runs each with run_native_lane.sh.
#   lane_queue.sh <queue dir> <gpus e.g. 0,1> <source jsonl> <num_prompts> <repeats> <run prefix>
# Queue layout: <queue>/todo/<NN_method>, claimed -> <queue>/running/<NN_method>.<lane>,
# finished -> done/ or failed/. A todo entry whose method has no export yet (DSpark
# still training) is put back and retried after the other entries.
set -uo pipefail
Q=$1 GPUS=$2 SOURCE=$3 NUM=$4 REPEATS=$5 PREFIX=$6
HERE=$(cd "$(dirname "$0")" && pwd)
: "${Q36_SITE:?set Q36_SITE}"
source "$Q36_SITE"
LANE="gpu${GPUS//,/_}"
mkdir -p "$Q"/{todo,running,done,failed}
say() { echo "[queue $LANE] $(date '+%F %T') $*"; }
while true; do
  claimed=""
  for entry in $(ls "$Q/todo" 2>/dev/null | sort); do
    method=${entry#*_}
    if [ ! -f "$TASK/exports/${method}_runtime.json" ]; then continue; fi
    if mv "$Q/todo/$entry" "$Q/running/$entry.$LANE" 2>/dev/null; then claimed=$entry; break; fi
  done
  if [ -z "$claimed" ]; then
    if [ -z "$(ls "$Q/todo" 2>/dev/null)" ]; then say "queue empty, exiting"; break; fi
    sleep 120; continue
  fi
  method=${claimed#*_}
  name="${PREFIX}_${method}"
  say "start $method on $GPUS -> $name"
  bash "$HERE/run_native_lane.sh" "$method" "$GPUS" "$SOURCE" "$NUM" "$REPEATS" "$name"
  rc=$?
  if [ $rc -eq 0 ]; then mv "$Q/running/$claimed.$LANE" "$Q/done/$claimed"; else mv "$Q/running/$claimed.$LANE" "$Q/failed/$claimed"; fi
  say "end $method rc=$rc"
done
