#!/bin/bash
set -euo pipefail
source /root/autodl-tmp/qwen36_rootcause_20261101/repo/research/qwen36_rootcause/a800_site.sh
cd "$Q36_ROOTCAUSE_TASK/repo"
case "$1" in
 trace) exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/trace_target.py;;
 tune) exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/moe_tune.py --gpu 1 --tp-size 1 --batch-sizes 1 8 --out "$Q36_ROOTCAUSE_TASK/moe_tune_a1";;
 ab) exec "$Q36_EXISTING_TASK/runtime/bin/python" -u research/qwen36_rootcause/e2e_compare.py;;
 *) exit 2;;
esac
