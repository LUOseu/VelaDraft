#!/bin/bash
# Series of paired A/B harness runs: per-step wall/peak/padding + loss/grad parity
# vs canonical and vs the ref (machinery-identity) arm.
# usage: vconv_harness_series.sh [series-name]
set -u
NAME=${1:-series1}
REPO=/home/lyh/veladraft-conv/repo
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
export CUDA_VISIBLE_DEVICES=${VCONV_GPU:-1}
export PYTHONPATH=$REPO
export REPO=$REPO
export D1_GIT_SHA=$(git -C $REPO rev-parse HEAD)
CFG=/home/lyh/veladraft-conv/config/e2e_w3_conv.json
EV=/home/lyh/veladraft-conv/evidence
mkdir -p "$EV" /home/lyh/veladraft-conv/runs/harness_run_dir
STATUS=/home/lyh/veladraft-conv/logs/vconv_harness_${NAME}.log
cd "$REPO" || exit 1
run () {  # run <tag> <store> <sub> <batches> <arms> [extra...]
  tag=$1; store=$2; sub=$3; batches=$4; arms=$5; shift 5
  echo "start $tag $(date '+%F %H:%M:%S')" >> "$STATUS"
  "$PY" research/convergence/harness/r3_ab_step.py --store "$store" \
    --run-dir /home/lyh/veladraft-conv/runs/harness_run_dir --config "$CFG" \
    --device cuda:0 --batches "$batches" --warmup 1 --batch-size 8 --sub-batch "$sub" \
    --ref-micro-batch 8 --arms "$arms" --parity 3 \
    --out "$EV/r3_${tag}.jsonl" --tag "$tag" "$@" >> "$STATUS" 2>&1
  echo "done $tag rc=$? $(date '+%F %H:%M:%S')" >> "$STATUS"
}
W3=/home/lyh/rsi-d1/runs/d1_w3_probe/features
MIX=/home/lyh/rsi-d1/runs/d1_mix_w1w2/features
W1=/home/lyh/rsi-d1/runs/d1_w1_probe/features
ARMS=canonical,cp,ref,split,token,dual,cost,grid
for r in 1 2 3; do run "w3_k4_r$r" "$W3" 2 100 "$ARMS"; done
for r in 1 2 3; do run "w3_k2_r$r" "$W3" 4 100 "$ARMS"; done
for r in 1 2; do run "mix_k4_r$r" "$MIX" 2 100 "$ARMS"; done
run "w1_k4_r1" "$W1" 2 60 "$ARMS"
run "seam_w3_k4" "$W3" 2 3 "canonical,cp,ref,split"
echo "series $NAME done $(date '+%F %H:%M:%S')" >> "$STATUS"
