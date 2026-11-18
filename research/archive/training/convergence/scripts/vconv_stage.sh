#!/bin/bash
# Stage one leg dir: hardlink the frozen W3 store (data/features/regen) and
# overlay the val-64 held-out split + its images (real files, no write-through).
# usage: vconv_stage.sh <leg_name>
set -eu
LEG=$1
PROBE=/home/lyh/rsi-d1/runs/d1_w3_probe
VAL64=/home/lyh/rsi-d1/runs/d1q_val64
RUN=/home/lyh/veladraft-conv/runs/$LEG
if [ -e "$RUN" ]; then echo "refusing: $RUN already exists"; exit 1; fi
mkdir -p "$RUN/data" "$RUN/features" "$RUN/regen"
cp -al "$PROBE/features/." "$RUN/features/"
cp -al "$PROBE/regen/." "$RUN/regen/"
cp -al "$PROBE/data/images" "$RUN/data/images"
for f in data.jsonl manifest.json meta.json stage_manifest.json; do [ -e "$PROBE/data/$f" ] && cp -al "$PROBE/data/$f" "$RUN/data/$f"; done
cp "$VAL64/val.jsonl" "$RUN/data/val.jsonl"
cp -al "$VAL64/images/." "$RUN/data/images/"
echo "staged $RUN: val=$(wc -l < "$RUN/data/val.jsonl") features=$(wc -l < "$RUN/features/index.jsonl")"
