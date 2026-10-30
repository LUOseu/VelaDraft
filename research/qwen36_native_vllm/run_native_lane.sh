#!/bin/bash
# One native-eval lane (TP=2 by default) for one Qwen3.6 method on this node.
#   run_native_lane.sh <method> <gpus e.g. 0,1> <source jsonl> <num_prompts> <repeats> <run name>
# Frozen settings live in this file; the evidence directory is node-local (tmpfs,
# per-stage compile caches are many small files) and is archived to cephfs at the end
# without the caches.
set -euo pipefail
METHOD=$1 GPUS=$2 SOURCE=$3 NUM=$4 REPEATS=$5 NAME=$6

# Machine paths come from a site file kept outside the repository, defining
# REPO PY TASK SHM TARGET IMAGES (and optionally HF_HOME and Q36_PRELAUNCH).
: "${Q36_SITE:?set Q36_SITE to the site env file}"
source "$Q36_SITE"
DRAFT=$TASK/exports/$METHOD
MANIFEST=$TASK/exports/${METHOD}_runtime.json
OUT=$SHM/runs/$NAME/result.json
ARCHIVE=$TASK/runs/$NAME

# Frozen protocol knobs (see FROZEN_CONFIG.md).
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM=${GPU_MEM:-0.85}
MIN_PIXELS=65536
MAX_PIXELS=1048576
TEMPLATE_KWARGS='{"enable_thinking": false}'
SD_GRAPH=FULL_AND_PIECEWISE
# MMSpec reference construction (evaluation/utils.py build_prompt).
SYSTEM_PROMPT="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

mkdir -p "$SHM/runs" "$SHM/gpu_locks" "$SHM/triton" "$SHM/xdg" "$TASK/logs"
export PYTHONPATH=$REPO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$SHM/triton XDG_CACHE_HOME=$SHM/xdg
export QWEN8B_DIGEST_CACHE=$SHM/digests.sqlite
export VELADRAFT_GPU_LOCK_DIR=$SHM/gpu_locks
export VELADRAFT_GPU_OWNERSHIP=count VELADRAFT_GPU_MAX_OWN_APPS=${VELADRAFT_GPU_MAX_OWN_APPS:-2}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

ARGS=(--target "$TARGET" --draft "$DRAFT" --source "$SOURCE" --images-dir "$IMAGES"
      --method "$METHOD" --gpus "$GPUS" --num-prompts "$NUM" --repeats "$REPEATS"
      --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$MAX_MODEL_LEN"
      --speculative-tokens 7 --gpu-memory-utilization "$GPU_MEM"
      --min-pixels $MIN_PIXELS --max-pixels $MAX_PIXELS
      --chat-template-kwargs "$TEMPLATE_KWARGS" --sd-cudagraph-mode "$SD_GRAPH"
      --system-prompt "$SYSTEM_PROMPT" --runtime-manifest "$MANIFEST" --out "$OUT")

echo "[lane] $(date '+%F %T') $(hostname -I | cut -d' ' -f1) method=$METHOD gpus=$GPUS source=$SOURCE n=$NUM repeats=$REPEATS out=$OUT"

# An optional site hook (Q36_PRELAUNCH, given the lane's GPU list) may release site
# processes on this lane's cards before the lease; the protocol itself never stops a
# foreign process and refuses a busy card. Extra apps seen while an engine loads are
# recorded, and only extra apps inside timed windows invalidate a run.
if [ -n "${Q36_PRELAUNCH:-}" ]; then
  "$Q36_PRELAUNCH" "$GPUS" || true
fi
for _ in $(seq 60); do
  busy=0
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    apps=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader)
    { [ "$used" -ge 500 ] || [ -n "$apps" ]; } && busy=1
  done
  [ $busy -eq 0 ] && break
  sleep 2
done
cd "$REPO"
set +e
"$PY" -m vlm_dflash.cli native-eval "${ARGS[@]}"
RC=$?
set -e
mkdir -p "$ARCHIVE"
cp "$OUT" "$ARCHIVE/" 2>/dev/null || true
EV=${OUT%.json}.evidence
if [ -d "$EV" ]; then
  # A handful of JSON/log files; the per-stage compile caches stay node-local.
  tar -C "$(dirname "$EV")" --exclude='*/cache' --exclude='*/inductor' -cf - "$(basename "$EV")" | tar -C "$ARCHIVE" -xf -
fi
echo "LANE_EXIT=$RC $(date '+%F %T')"
exit $RC
