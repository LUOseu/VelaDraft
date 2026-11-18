#!/bin/bash
# Fetch all confirm+pilot leg artifacts and run the frozen quality gates.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CONV=$(cd "$HERE/.." && pwd)
PY=${CONV_PY:-/Users/Admin/grounding-parallel/vlm-dflash-dflare/.venv/bin/python}
bash "$HERE/fetch_evidence.sh" \
  conv_canon_s42 conv_canon_s43 conv_canon_s44 conv_canon_s45 \
  conv_grid2_s42 conv_grid2_s43 conv_grid2_s44 conv_grid2_s45 \
  conv_token2_s42 conv_token2_s43 conv_split2_s42 conv_split2_s43
cd "$CONV"
"$PY" build_quality_data.py --legs-dir evidence/legs --include-pilot --out evidence/quality_data.json
"$PY" analyze_quality.py --data evidence/quality_data.json --out evidence/quality_verdict.json
echo "verdict -> $CONV/evidence/quality_verdict.json"
