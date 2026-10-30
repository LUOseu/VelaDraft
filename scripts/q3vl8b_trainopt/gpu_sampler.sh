#!/bin/bash
# Low-frequency GPU record for the campaign: one JSON line per GPU every INTERVAL seconds into
# <campaign>/raw/gpu_samples.jsonl (exported gzipped). Stops when <campaign>/UPLOAD_FINAL exists.
set -u
CAMPAIGN=$1
INTERVAL=${INTERVAL:-60}
OUT=$CAMPAIGN/raw/gpu_samples.jsonl
while [ ! -e "$CAMPAIGN/UPLOAD_FINAL" ]; do
  t=$(date +%s)
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' -v t="$t" '{printf "{\"time\": %s, \"gpu\": %s, \"util_pct\": %s, \"mem_used_mib\": %s, \"mem_total_mib\": %s, \"power_w\": %s}\n", t, $1, $2, $3, $4, $5}' >> "$OUT"
  sleep "$INTERVAL"
done
