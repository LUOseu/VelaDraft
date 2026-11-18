#!/bin/bash
# Retry-epoch driver around vconv_confirm_series.sh: keeps re-queuing unfinished
# legs (chain is idempotent) until all are done or epochs are exhausted.
# usage: vconv_confirm_driver.sh <L> <gpus> [pilot] [epochs]
set -u
L=$1; GPUS=$2; PILOT=${3:-}; EPOCHS=${4:-8}
SCRIPTS=/home/lyh/veladraft-conv/scripts
RUNS=/home/lyh/veladraft-conv/runs
STATUS=/home/lyh/veladraft-conv/logs/vconv_status.log

legs=(conv_canon_s42 conv_canon_s43 conv_canon_s44 conv_canon_s45
      conv_grid2_s42 conv_grid2_s43 conv_grid2_s44 conv_grid2_s45)
if [ "$PILOT" = "pilot" ]; then
  legs+=(conv_token2_s42 conv_token2_s43 conv_split2_s42 conv_split2_s43)
fi

remaining () {
  local n=0
  for leg in "${legs[@]}"; do
    if [ ! -f "$RUNS/$leg/train/train_summary.json" ] || [ ! -f "$RUNS/$leg/eval/eval_result.json" ]; then
      n=$((n+1))
    fi
  done
  echo $n
}

for e in $(seq 1 "$EPOCHS"); do
  rem=$(remaining)
  echo "driver epoch $e remaining=$rem $(date '+%F %H:%M:%S')" >> "$STATUS"
  [ "$rem" -eq 0 ] && { echo "driver: all legs done" >> "$STATUS"; exit 0; }
  rm -f /home/lyh/veladraft-conv/logs/queue_gpu*.txt
  bash "$SCRIPTS/vconv_confirm_series.sh" "$L" "$GPUS" "$PILOT" >> "$STATUS" 2>&1
  # wait for this epoch's workers (they are nohup'd by the series script)
  while pgrep -f ".vconv_worker_gpu" >/dev/null; do sleep 60; done
  for leg in "${legs[@]}"; do
    if [ ! -f "$RUNS/$leg/eval/eval_result.json" ]; then
      echo "driver: still pending $leg after epoch $e" >> "$STATUS"
    fi
  done
done
echo "driver: epochs exhausted remaining=$(remaining) $(date '+%F %H:%M:%S')" >> "$STATUS"
