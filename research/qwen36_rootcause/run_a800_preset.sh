#!/bin/bash
# Reproduce the validated diagnostic protocol into a new output directory.
# Usage: bash run_a800_preset.sh /root/autodl-tmp/qwen36_rootcause_20261101/new_confirmation [gpu]
set -euo pipefail
out=${1:?provide a NEW output directory};gpu=${2:-0}
source /root/autodl-tmp/qwen36_rootcause_20261101/repo/research/qwen36_rootcause/a800_site.sh
cd "$Q36_ROOTCAUSE_TASK/repo"
exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/confirmation.py \
 --gpu "$gpu" --plan "$Q36_ROOTCAUSE_TASK/k2_setup/plan.evidence/plan.json" \
 --out "$out" --tuned-folder "$Q36_ROOTCAUSE_TASK/moe_tune_k2_a1/tuned_configs" --order ar,sd
