#!/bin/bash
# Pull per-leg eval artifacts from the server into local evidence/legs/<name>/.
# usage: fetch_evidence.sh <leg> [<leg> ...] ; env: VCONV_HOST (default oab-remote)
set -eu
HOST=${VCONV_HOST:-oab-remote}
DEST=$(cd "$(dirname "$0")/.." && pwd)/evidence/legs
for leg in "$@"; do
  mkdir -p "$DEST/$leg/eval" "$DEST/$leg/train"
  scp -q "$HOST:/home/lyh/veladraft-conv/runs/$leg/eval/eval_result.json" "$DEST/$leg/eval/" 2>/dev/null || true
  scp -q "$HOST:/home/lyh/veladraft-conv/runs/$leg/eval/eval_samples.jsonl" "$DEST/$leg/eval/" 2>/dev/null || true
  scp -q "$HOST:/home/lyh/veladraft-conv/runs/$leg/train/train_summary.json" "$DEST/$leg/train/" 2>/dev/null || true
  scp -q "$HOST:/home/lyh/veladraft-conv/runs/$leg/train/train_log.jsonl" "$DEST/$leg/train/" 2>/dev/null || true
  echo "fetched $leg"
done
