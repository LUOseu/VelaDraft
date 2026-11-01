#!/bin/bash
# Reproduce the confirmed natural-completion preset in a NEW evidence directory.
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
out=${1:?provide a NEW absolute output directory}
gpu=${2:-0}
mkdir "$out"
export VLLM_GDN_DECODE_KERNEL=triton
for arm in ar sd; do
  export Q36_V032_REUSE_CACHE="$TASK/v032_gdn_triton_cache_${arm}"
  bash "$TASK/repo/research/qwen36_v032/run.sh" --gpu "$gpu" --arm "$arm" --full --out "$out/$arm"
done
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export PYTHONPATH="$TASK/repo"
cd "$TASK/repo"
exec python -u -m research.qwen36_v032.summarize "$out"
