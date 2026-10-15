#!/bin/bash
# Per-drafter regroup adaptation smoke (run before enabling grid2 legs for a
# drafter other than dflash; see RUNBOOK.md section 4).
#
# usage: regroup_adaptation_smoke.sh <name> <gpu> <config> <template_run_dir> <steps>
#   template_run_dir: a staged run dir (features/ regen/ data/) to hardlink from
# gate 1: canonical/8 + identity_bypass must be BITWISE identical to regroup-off
# gate 2: grid/2 runs and eval token_exact=true
set -u
NAME=$1; GPU=$2; CFG=$3; TPL=$4; STEPS=${5:-3}
PY=${VELADRAFT_PYTHON:-python3}
REPO=${VELADRAFT_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
ROOT=${VELADRAFT_SMOKE_ROOT:-/tmp/veladraft_regroup_smoke}
STATUS=$ROOT/smoke.log
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO
mkdir -p "$ROOT"
stage () {  # stage <leg>
  local run=$ROOT/$1
  [ -e "$run" ] && { echo "refusing existing $run" >&2; exit 1; }
  mkdir -p "$run/data" "$run/features" "$run/regen"
  cp -al "$TPL/features/." "$run/features/"
  cp -al "$TPL/regen/." "$run/regen/"
  cp -al "$TPL/data/." "$run/data/" 2>/dev/null || true
}
leg () {  # leg <suffix> [regroup sets...]
  local suffix=$1; shift
  local run=$ROOT/${NAME}_${suffix}
  stage "${NAME}_${suffix}"
  echo "start $NAME/$suffix $(date '+%F %H:%M:%S')" >> "$STATUS"
  ( cd "$REPO" && "$PY" scripts/train_dflash.py --config "$CFG" --run-dir "$run" \
      --steps "$STEPS" --skip-sanity --set train.seed=43 --set train.feature_mode=offline \
      --set train.log_every_steps=1 --set algorithm="$NAME" "$@" ) >> "$STATUS" 2>&1
  echo "done $NAME/$suffix rc=$? $(date '+%F %H:%M:%S')" >> "$STATUS"
}
leg off
leg bypass --set train.regroup.mode=canonical --set train.regroup.sub_batch=8 \
           --set train.regroup.fp32_weights=true --set train.regroup.identity_bypass=true
leg grid --set train.regroup.mode=grid --set train.regroup.sub_batch=2 \
         --set train.regroup.fp32_weights=true --set train.regroup.identity_bypass=true
"$PY" - "$ROOT/${NAME}_off/train/train_log.jsonl" "$ROOT/${NAME}_bypass/train/train_log.jsonl" <<'PYEOF'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1])]
b = [json.loads(l) for l in open(sys.argv[2])]
ok = len(a) == len(b) and all(
    x["loss"] == y["loss"] and x["grad_norm"] == y["grad_norm"] for x, y in zip(a, b))
print("GATE1_BITWISE_IDENTITY:", "PASS" if ok else "FAIL")
if not ok:
    for x, y in zip(a, b):
        if x["loss"] != y["loss"] or x["grad_norm"] != y["grad_norm"]:
            print(" first divergence:", x.get("step"), x["loss"], x["grad_norm"], "vs", y["loss"], y["grad_norm"])
            break
raise SystemExit(0 if ok else 1)
PYEOF
echo "GATE2_GRID_RAN: check $ROOT/${NAME}_grid/train/train_summary.json loss_decreased/nan_seen and run eval for token_exact" | tee -a "$STATUS"
