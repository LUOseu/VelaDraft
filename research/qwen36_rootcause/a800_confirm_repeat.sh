#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/qwen36_rootcause_20261101/repo/research/qwen36_rootcause/a800_site.sh
cd "$Q36_ROOTCAUSE_TASK/repo"
exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/confirmation.py \
 --gpu 1 --plan "$Q36_ROOTCAUSE_TASK/k2_setup/plan.evidence/plan.json" \
 --out "$Q36_ROOTCAUSE_TASK/confirm_tuned_k2_gpu1_repeat" \
 --tuned-folder "$Q36_ROOTCAUSE_TASK/moe_tune_k2_a1/tuned_configs" \
 --after "$Q36_ROOTCAUSE_TASK/confirm_default_k7_gpu1/SUMMARY.json" --order sd,ar
