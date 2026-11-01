#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
gpu=${1:?GPU required}
out="$TASK/confirm_v032_gpu${gpu}_a1"
mkdir "$out"
if [ "$gpu" = 0 ]; then arms=(ar sd); else arms=(sd ar); fi
for arm in "${arms[@]}"; do
  export Q36_V032_REUSE_CACHE="$TASK/v032_${arm}_a2"
  bash "$TASK/repo/research/qwen36_v032/run.sh" --gpu "$gpu" --arm "$arm" --full --out "$out/$arm"
done
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export PYTHONPATH="$TASK/repo"
cd "$TASK/repo"
exec python -u -m research.qwen36_v032.summarize "$out"
