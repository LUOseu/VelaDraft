#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
gpu=${1:?GPU required}
out="$TASK/confirm_v032_gdn_triton_gpu${gpu}_a1"
mkdir "$out"
export VLLM_GDN_DECODE_KERNEL=triton
if [ "$gpu" = 0 ]; then arms=(sd ar); else arms=(ar sd); fi
for arm in "${arms[@]}"; do
  export Q36_V032_REUSE_CACHE="$TASK/v032_gdn_triton_cache_${arm}"
  bash "$TASK/repo/research/qwen36_v032/run.sh" --gpu "$gpu" --arm "$arm" --full --out "$out/$arm"
done
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export PYTHONPATH="$TASK/repo"
cd "$TASK/repo"
exec python -u -m research.qwen36_v032.summarize "$out"
