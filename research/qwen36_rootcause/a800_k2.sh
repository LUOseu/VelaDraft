#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/qwen36_rootcause_20261101/repo/research/qwen36_rootcause/a800_site.sh
cause_base=$Q36_ROOTCAUSE_TASK
cd "$cause_base/repo"
"$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/moe_tune.py --gpu 0 --tp-size 1 --batch-sizes 1 3 --out "$cause_base/moe_tune_k2_a1"
mkdir -p "$cause_base/ab_k2"
ln -s ../repo "$cause_base/ab_k2/repo"
ln -s ../moe_tune_k2_a1 "$cause_base/ab_k2/moe_tune_a1"
export Q36_ROOTCAUSE_TASK="$cause_base/ab_k2" Q36_ROOTCAUSE_GPUS=0
export Q36_EXISTING_PLAN="$cause_base/k2_setup/plan.evidence/plan.json"
exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/e2e_compare.py
