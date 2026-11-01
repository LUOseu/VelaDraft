#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export PYTHONPATH="$TASK/repo"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export FLASHINFER_WORKSPACE_BASE="$TASK/flashinfer_v032_cuda130"
export PATH="$CUDA_HOME/bin:$TASK/runtime_v032/bin:$PATH"
cd "$TASK/repo"
exec "$TASK/runtime_v032/bin/python" -u -m research.qwen36_v032.prewarm
