#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export PYTHONPATH="$TASK/repo"
export TRITON_CACHE_DIR="$TASK/triton"
export TORCHINDUCTOR_CACHE_DIR="$TASK/inductor"
export VLLM_CACHE_ROOT="$TASK/vllm_cache"
cd "$TASK/repo"
case "$1" in
  moe) exec python -u -m research.qwen36_backend.moe_fixed_search --gpu 0 --out "$TASK/moe_search_a1" --baseline-folder /root/autodl-tmp/qwen36_rootcause_20261101/moe_tune_k2_a1/tuned_configs ;;
  gate) exec python -u -m research.qwen36_backend.shared_gate_probe --gpu 1 --out "$TASK/gate_probe_a1" ;;
  *) exit 2 ;;
esac
