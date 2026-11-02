#!/bin/bash
set -u
task=/home/lyh/qwen36_k_sweep_20261101
export TMPDIR="$task/tmp" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH="$task/repo" PYTHONDONTWRITEBYTECODE=1
"$task/runtime/bin/python" "$task/repo/research/qwen36_k_sweep/prepare.py"
rc=$?
echo "$rc" > "$task/PREPARE_EXIT"
exit "$rc"
