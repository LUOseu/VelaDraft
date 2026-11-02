#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/qwen36_rootcause_20261101/repo/research/qwen36_rootcause/a800_site.sh
cd "$Q36_ROOTCAUSE_TASK/repo"
case "$1" in
 default)
  exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/confirmation.py --gpu 1 --plan "$Q36_EXISTING_PLAN" --out "$Q36_ROOTCAUSE_TASK/confirm_default_k7_gpu1" --after "$Q36_ROOTCAUSE_TASK/moe_e2e_ab_a1/SUMMARY.json";;
 candidate)
  exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/confirmation.py --gpu 0 --plan "$Q36_ROOTCAUSE_TASK/k2_setup/plan.evidence/plan.json" --out "$Q36_ROOTCAUSE_TASK/confirm_tuned_k2_gpu0" --tuned-folder "$Q36_ROOTCAUSE_TASK/moe_tune_k2_a1/tuned_configs" --after "$Q36_ROOTCAUSE_TASK/ab_k2/moe_e2e_ab_a1/SUMMARY.json";;
 *) exit 2;;
esac
