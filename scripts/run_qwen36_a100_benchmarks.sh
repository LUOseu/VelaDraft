#!/usr/bin/env bash
# One-command entry for the Qwen3.6-35B-A3B seven-method MMSpec benchmark (research/qwen36_a100_seven_eval).
#   bash scripts/run_qwen36_a100_benchmarks.sh --preflight --config configs/eval/qwen36_a100_seven.yaml
#   bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-full --methods dflash,dspark,domino,jetspec,dflash2,dflare,dfly --draft-length 3 --publish
# Uses the python on PATH (or $Q36_PYTHON); it must be the vLLM 0.32.0 environment from the README.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec "${Q36_PYTHON:-python}" -u -m research.qwen36_a100_seven_eval.cli "$@"
