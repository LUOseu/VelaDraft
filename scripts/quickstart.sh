#!/usr/bin/env bash
# End-to-end MVP run: data -> regen -> capture -> train -> eval.
#
# Usage (defaults to the COCO smoke config on a CUDA box):
#   bash scripts/quickstart.sh                       # smoke pipeline
#   bash scripts/quickstart.sh --config configs/qwen2_5_vl_3b.yaml --limit 512 --steps 1500
#
# Extra key=value overrides are forwarded to every stage, e.g.:
#   bash scripts/quickstart.sh --set target.model=/path/to/Qwen2.5-VL-3B-Instruct
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/smoke_coco.yaml"
RUN_DIR=""
LIMIT="64"
STEPS=""
EVAL_PROMPTS=""
SETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --eval-prompts) EVAL_PROMPTS="$2"; shift 2 ;;
    --set) SETS+=("--set" "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHON="${PYTHON:-python3}"
COMMON=(--config "$CONFIG")
if [[ -n "$RUN_DIR" ]]; then
  COMMON+=(--run-dir "$RUN_DIR")
fi
if [[ ${#SETS[@]} -gt 0 ]]; then
  COMMON+=("${SETS[@]}")
fi

echo "-- stage 1/5: prepare_data"
"$PYTHON" scripts/prepare_data.py "${COMMON[@]}" --limit "$LIMIT"

echo "-- stage 2/5: generate_target_data"
"$PYTHON" scripts/generate_target_data.py "${COMMON[@]}" --limit "$LIMIT"

echo "-- stage 3/5: capture_features"
"$PYTHON" scripts/capture_features.py "${COMMON[@]}" --limit "$LIMIT"

echo "-- stage 4/5: train_dflash"
TRAIN_ARGS=()
if [[ -n "$STEPS" ]]; then
  TRAIN_ARGS+=(--steps "$STEPS")
fi
"$PYTHON" scripts/train_dflash.py "${COMMON[@]}" "${TRAIN_ARGS[@]}"

echo "-- stage 5/5: eval_dflash"
EVAL_ARGS=()
if [[ -n "$EVAL_PROMPTS" ]]; then
  EVAL_ARGS+=(--num-prompts "$EVAL_PROMPTS")
fi
"$PYTHON" scripts/eval_dflash.py "${COMMON[@]}" "${EVAL_ARGS[@]}"

echo "quickstart complete."
