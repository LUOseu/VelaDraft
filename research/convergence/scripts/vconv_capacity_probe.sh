#!/bin/bash
# Capacity probe at logical batch 16 on the W3 store:
#   A) canonical with ref-micro-batch 16 (shipped layout, single 16-row chunk) -> expect OOM
#   B) canonical with ref-micro-batch 8  (shipped 2x8 chunking)                -> fits, peak ref
#   C) split sub2 (8 groups of 2, positional) and grid sub2 (8 groups of 2,   -> fit, peak/padding
#      C x A sorted) inside the same run as A.
# usage: VCONV_GPU=<g> vconv_capacity_probe.sh
set -u
REPO=/home/lyh/veladraft-conv/repo
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
GPU=${VCONV_GPU:-0}
EV=/home/lyh/veladraft-conv/evidence
STATUS=/home/lyh/veladraft-conv/logs/vconv_capacity_probe.log
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export REPO=$REPO
export D1_GIT_SHA=$(git -C "$REPO" rev-parse HEAD)
W3=/home/lyh/rsi-d1/runs/d1_w3_probe/features
CFG=/home/lyh/veladraft-conv/config/e2e_w3_conv.json
RUN=/home/lyh/veladraft-conv/runs/harness_run_dir
mkdir -p "$EV"
cd "$REPO" || exit 1
probe () {  # name arms refmicro sub batches
  name=$1; arms=$2; refmb=$3; sub=$4; batches=$5
  echo "start $name arms=$arms ref_micro=$refmb sub=$sub batches=$batches $(date '+%F %H:%M:%S')" >> "$STATUS"
  "$PY" research/convergence/harness/r3_ab_step.py --store "$W3" --run-dir "$RUN" --config "$CFG" \
    --device cuda:0 --batches "$batches" --warmup 1 --batch-size 16 --sub-batch "$sub" \
    --ref-micro-batch "$refmb" --arms "$arms" --parity 0 \
    --out "$EV/r3_capacity_${name}.jsonl" --tag "capacity_$name" >> "$STATUS" 2>&1
  echo "done $name rc=$? $(date '+%F %H:%M:%S')" >> "$STATUS"
}
probe mb16_vs_grid canonical,split,grid 16 2 4
probe canon_mb8     canonical          8  2 4
echo "capacity probe done $(date '+%F %H:%M:%S')" >> "$STATUS"
