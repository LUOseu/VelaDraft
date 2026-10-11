#!/usr/bin/env bash
# Phase-2 scaling suite: train the same DFlash recipe at increasing dataset
# slices and record one eval headline per point.
#
# Jobs (all reuse the pinned recipe; only the slice changes):
#   refresh3b  512  rows, configs/qwen2_5_vl_3b.yaml     -> runs/qwen25vl3b_refresh
#   3b_2k     2000 rows, configs/qwen2_5_vl_3b_2k.yaml   -> runs/qwen25vl3b_2k
#   3b_5k     5000 rows, configs/qwen2_5_vl_3b_5k.yaml   -> runs/qwen25vl3b_5k
#   8b_2k     2000 rows, configs/qwen3_vl_8b_2k.yaml     -> runs/qwen3vl8b_2k
#
# Usage (GPU box, from the repo root):
#   PYTHON=.venv/bin/python CUDA_VISIBLE_DEVICES=0 bash scripts/run_scaling_suite.sh
#
# Each job runs the full quickstart (prepare -> regen -> capture -> train ->
# eval), then an HF export. `prune=1` jobs delete features/ after a successful
# run: captured features are deterministic to recreate and are by far the
# largest artifact; checkpoints, logs and eval results are kept.
# The script skips (does not fail) a job when the disk preflight fails.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
LOGDIR="${LOGDIR:-logs}"
STATUS="$LOGDIR/scaling_suite_status.txt"
mkdir -p "$LOGDIR"

free_mb() { df -Pm . | awk 'NR==2 {print $4}'; }

summarize() {
  "$PYTHON" - "$1" <<'PY' >> "$STATUS"
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
try:
    res = json.loads((run / "eval" / "eval_result.json").read_text())
    parts = [f"run={run.name}", "prompts=" + str(res["settings"]["num_prompts"]),
             "token_exact=" + str(res["equivalence_check"]["token_exact"])]
    for name, r in res["runs"].items():
        parts.append(f"{name}: AL={r['mean_al']:.4f} slots={['%.3f' % s for s in r['per_slot_acceptance']]}")
    print(" ".join(parts))
except Exception as exc:  # keep the suite alive; the log has details
    print(f"run={run.name} summary FAILED: {exc}")
PY
}

run_job() {  # name config limit need_mb prune [run_dir]
  local name="$1" config="$2" limit="$3" need_mb="$4" prune="$5" run_dir="${6:-}"
  local log="$LOGDIR/scale_${name}.log"
  if [[ -z "$run_dir" ]]; then
    run_dir=$("$PYTHON" - "$config" <<'PY'
import sys
sys.path.insert(0, ".")
from vlm_dflash.config import load_config
print(load_config(sys.argv[1])["run"]["dir"])
PY
)
  fi
  local free; free=$(free_mb)
  echo "[suite] job=$name config=$config limit=$limit run=$run_dir free=${free}MB need=${need_mb}MB $(date -Is)" | tee -a "$STATUS"
  if (( free < need_mb )); then
    echo "[suite] SKIP $name (not enough disk)" | tee -a "$STATUS"
    return 0
  fi
  local args=(--config "$config" --limit "$limit")
  [[ -n "$run_dir" ]] && args+=(--run-dir "$run_dir")
  if PYTHON="$PYTHON" bash scripts/quickstart.sh "${args[@]}" >>"$log" 2>&1; then
    echo "[suite] quickstart OK $name $(date -Is)" >>"$STATUS"
    if "$PYTHON" -m vlm_dflash export "$run_dir" --format hf >>"$log" 2>&1; then
      echo "[suite] export OK $name" >>"$STATUS"
    else
      echo "[suite] export FAILED $name" >>"$STATUS"
    fi
    summarize "$run_dir"
    if [[ "$prune" == "1" && -d "$run_dir/features" ]]; then
      du -sh "$run_dir/features" >>"$STATUS" 2>/dev/null || true
      rm -rf "$run_dir/features"
      echo "[suite] pruned $run_dir/features (deterministic to recapture; ckpts kept)" >>"$STATUS"
    fi
  else
    echo "[suite] quickstart FAILED $name (see $log)" >>"$STATUS"
  fi
}

run_job refresh3b configs/qwen2_5_vl_3b.yaml    512  8000  0 runs/qwen25vl3b_refresh
run_job 3b_2k     configs/qwen2_5_vl_3b_2k.yaml 2000 22000 1
run_job 3b_5k     configs/qwen2_5_vl_3b_5k.yaml 5000 45000 1
run_job 8b_2k     configs/qwen3_vl_8b_2k.yaml   2000 55000 0
echo "[suite] all jobs done $(date -Is)" | tee -a "$STATUS"
