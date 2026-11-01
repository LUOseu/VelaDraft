#!/bin/bash
set -euo pipefail
TASK=/root/autodl-tmp/qwen36_backend_20261101
source "$TASK/repo/research/qwen36_rootcause/a800_site.sh"
export PYTHONPATH="$TASK/repo"
export TMPDIR="$TASK/v032_tmp"
mkdir -p "$TMPDIR"
cd "$TASK/repo"
exec "$Q36_EXISTING_TASK/runtime/bin/python" -u -m research.qwen36_v032.queue "$@"
