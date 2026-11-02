#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export Q36_ROOTCAUSE_TASK="$TASK"
export Q36_EXISTING_PLAN=/root/autodl-tmp/qwen36_rootcause_20261101/k2_setup/plan.evidence/plan.json
export PYTHONPATH="$TASK/repo"
export TRITON_CACHE_DIR="$TASK/triton"
cd "$TASK/repo"
exec python -u -m research.qwen36_backend.e2e_compare "$@"
