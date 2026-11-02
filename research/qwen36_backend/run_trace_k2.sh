#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export Q36_ROOTCAUSE_TASK="$TASK"
export Q36_EXISTING_PLAN=/root/autodl-tmp/qwen36_rootcause_20261101/k2_setup/plan.evidence/plan.json
export Q36_ROOTCAUSE_GPUS=0
export PYTHONPATH="$TASK/repo"
export VLLM_TUNED_CONFIG_FOLDER=/root/autodl-tmp/qwen36_rootcause_20261101/moe_tune_k2_a1/tuned_configs
cd "$TASK/repo"
exec python -u -m research.qwen36_backend.trace_k2
