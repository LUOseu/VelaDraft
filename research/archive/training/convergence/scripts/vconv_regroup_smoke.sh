#!/bin/bash
# Generic per-drafter regroup smoke: 3 legs (off / canonical-bypass / grid) on a
# staged feature store, then the bitwise identity gate (off vs bypass).
# usage: VCONV_GPU=<g> vconv_regroup_smoke.sh <algorithm> <source_run_dir> [steps]
set -u
ALG=$1; SRC=$2; STEPS=${3:-3}
GPU=${VCONV_GPU:-0}
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
REPO=/home/lyh/veladraft-conv/repo
RUNS=/home/lyh/veladraft-conv/runs
STATUS=/home/lyh/veladraft-conv/logs/vconv_smoke_${ALG}.log
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
mkdir -p "$RUNS"
cd "$REPO" || exit 1

stage () {  # stage <leg>
  local run=$RUNS/smoke_${ALG}_$1
  if [ -e "$run" ]; then echo "refusing existing $run" >&2; exit 1; fi
  mkdir -p "$run/data" "$run/features" "$run/regen"
  cp -al "$SRC/features/." "$run/features/"
  cp -al "$SRC/regen/." "$run/regen/" 2>/dev/null || true
  cp -al "$SRC/data/." "$run/data/" 2>/dev/null || true
}
leg () {  # leg <suffix> [regroup sets...]
  local suffix=$1; shift
  local run=$RUNS/smoke_${ALG}_${suffix}
  stage "${suffix}"
  echo "start $ALG/$suffix $(date '+%F %H:%M:%S')" >> "$STATUS"
  "$PY" scripts/train_dflash.py --config "$SRC/config.json" --run-dir "$run" \
    --steps "$STEPS" --skip-sanity \
    --set train.seed=43 --set train.feature_mode=offline \
    --set train.log_every_steps=1 --set train.save_every_steps=100000 "$@" >> "$STATUS" 2>&1
  echo "done $ALG/$suffix rc=$? $(date '+%F %H:%M:%S')" >> "$STATUS"
}
leg off
leg bypass --set train.regroup.mode=canonical --set train.regroup.sub_batch=8 \
           --set train.regroup.fp32_weights=true --set train.regroup.identity_bypass=true
leg grid --set train.regroup.mode=grid --set train.regroup.sub_batch=2 \
         --set train.regroup.fp32_weights=true --set train.regroup.identity_bypass=true

"$PY" - "$RUNS/smoke_${ALG}_off/train/train_log.jsonl" "$RUNS/smoke_${ALG}_bypass/train/train_log.jsonl" <<'PYEOF' | tee -a "$STATUS"
import json, sys
a=[json.loads(l) for l in open(sys.argv[1])]
b=[json.loads(l) for l in open(sys.argv[2])]
ok = len(a)==len(b) and all(x["loss"]==y["loss"] and x["grad_norm"]==y["grad_norm"] for x,y in zip(a,b))
print("GATE1_BITWISE_IDENTITY:", "PASS" if ok else "FAIL")
if not ok:
    for x,y in zip(a,b):
        if x["loss"]!=y["loss"] or x["grad_norm"]!=y["grad_norm"]:
            print("  divergence at step", x.get("step"), x["loss"], x["grad_norm"], "vs", y["loss"], y["grad_norm"]); break
raise SystemExit(0 if ok else 1)
PYEOF
echo "GATE2_GRID: rc evidence in $STATUS (train_summary.regroup counters / traceback)" | tee -a "$STATUS"
