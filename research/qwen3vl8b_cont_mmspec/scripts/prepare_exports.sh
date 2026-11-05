#!/usr/bin/env bash
# CPU-only native export + runtime manifest for the seven Qwen3-VL-8B CONT finals.
#   PYTHON=<vLLM 0.22.1 env python> bash prepare_exports.sh <cont runs dir> <exports dir>
# <cont runs dir>/<method>/train/ckpt_final.pt must be the step-125000 CONT final. Existing
# exports/manifests are kept (an export directory must be empty to be written again).
set -euo pipefail
RUNS=$1 OUT=$2
: "${PYTHON:?set PYTHON to the evaluation environment python}"
NATIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../native" && pwd)"
export CUDA_VISIBLE_DEVICES=
mkdir -p "$OUT"
for method in dflash dspark domino jetspec dflash2 dflare dfly; do
  ckpt="$RUNS/$method/train/ckpt_final.pt"
  [[ -f "$ckpt" ]] || { echo "missing $ckpt" >&2; exit 1; }
  if [[ ! -f "$OUT/$method/export_receipt.json" ]]; then
    "$PYTHON" "$NATIVE/export_checkpoint.py" --method "$method" "$ckpt" "$OUT/$method" > "$OUT/$method.export.log"
  fi
  if [[ ! -f "$OUT/${method}_runtime.json" ]]; then
    "$PYTHON" "$NATIVE/make_runtime_manifest.py" --method "$method" --checkpoint "$ckpt" \
      --out "$OUT/${method}_runtime.json" > /dev/null
  fi
  echo "EXPORT_OK $method"
done
echo PREPARE_DONE
