#!/bin/bash
# One run of the Qwen3-VL-8B training-systems campaign.
#
#   launch_run.sh <config.json> <run_dir> <gpu list, e.g. 3 or 6,7> [extra `vlm-dflash train` args]
#
# Machine paths come from the node's environment file (TRAINOPT_ENV, kept on the node, not in
# the repository): VELA_REPO (this checkout), VELA_CORPUS (the frozen 200K corpus dir with data/
# and regen/, linked read-only into the run dir), VELA_TARGET, VELA_LOCAL (node-local scratch),
# HF_CACHE_HOME and TP (training python). Compile/temp caches are node-local and per run, so
# every run compiles cold. Appends to <run_dir>/stdout.log and ends it with EXIT_CODE=<rc>.
set -uo pipefail
CFG=$1
RUN=$2
GPUS=$3
shift 3
if [ -n "${TRAINOPT_ENV:-}" ]; then source "$TRAINOPT_ENV"; fi
: "${VELA_REPO:?}" "${VELA_CORPUS:?}" "${VELA_TARGET:?}" "${VELA_LOCAL:?}" "${HF_CACHE_HOME:?}" "${TP:?}"
PY=$TP
TAG=$(basename "$RUN")
LOCAL=$VELA_LOCAL/$TAG
mkdir -p "$RUN" "$LOCAL/tmp" "$LOCAL/triton" "$LOCAL/inductor"
[ -e "$RUN/data" ] || ln -s "$VELA_CORPUS/data" "$RUN/data"
[ -e "$RUN/regen" ] || ln -s "$VELA_CORPUS/regen" "$RUN/regen"
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$VELA_REPO
export HF_HOME=$HF_CACHE_HOME HF_HUB_OFFLINE=1 VLM_DFLASH_TARGET_QWEN3VL=$VELA_TARGET
export TMPDIR=$LOCAL/tmp TRITON_CACHE_DIR=$LOCAL/triton TORCHINDUCTOR_CACHE_DIR=$LOCAL/inductor
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8} TOKENIZERS_PARALLELISM=false
PIN=()
[ -n "${CPUS:-}" ] && PIN=(taskset -c "$CPUS")
{
  echo "[$(date '+%F %T')] START $TAG gpus=$GPUS cpus=${CPUS:-all} host=$(hostname)"
  echo "commit=$(git -C "$VELA_REPO" rev-parse HEAD) dirty=$(git -C "$VELA_REPO" status --porcelain | wc -l)"
  echo "config=$CFG args=$*"
} >> "$RUN/stdout.log"
"${PIN[@]}" "$PY" -m vlm_dflash train --mode online --config "$CFG" --run-dir "$RUN" "$@" >> "$RUN/stdout.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] EXIT_CODE=$rc" >> "$RUN/stdout.log"
exit $rc
