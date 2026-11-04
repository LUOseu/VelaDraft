#!/usr/bin/env bash
# Run in foreground (or in a supervised terminal); never starts training.
set -uo pipefail
: "${VELA_SANITIZE:?Set VELA_SANITIZE to node-specific sanitization JSON}"
if [ "$#" -ne 1 ] || [ ! -d "$1" ]; then echo "usage: $0 CAMPAIGN_DIR" >&2; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${MONITOR_PYTHON:-python3}
INTERVAL=${MONITOR_INTERVAL:-300}
while true; do
  "$PYTHON" "$HERE/progress.py" "$1" || echo 'Progress export failed' >&2
  "$PYTHON" "$HERE/hf_sync_safe.py" "$1" --once || echo 'HF sync failed; retry next cycle' >&2
  sleep "$INTERVAL"
done
