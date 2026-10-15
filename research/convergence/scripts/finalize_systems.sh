#!/bin/bash
# Pull the full harness-series raw rows and build the systems table.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CONV=$(cd "$HERE/.." && pwd)
PY=${CONV_PY:-/Users/Admin/grounding-parallel/vlm-dflash-dflare/.venv/bin/python}
HOST=${VCONV_HOST:-oab-remote}
mkdir -p "$CONV/evidence/harness"
scp -q "$HOST:/home/lyh/veladraft-conv/evidence/r3_*.jsonl" "$CONV/evidence/harness/" || true
cd "$CONV"
"$PY" analyze_systems.py --dir evidence/harness --out evidence/systems_table
echo "table -> $CONV/evidence/systems_table.md"
