#!/bin/bash
# One-shot dashboard for the VelaDraft scale run.
#   ssh oab-remote 'bash /home/lyh/vlm-scale/scripts/status.sh'
#   ssh oab-remote 'watch -n 60 bash /home/lyh/vlm-scale/scripts/status.sh'
ROOT=/home/lyh/vlm-scale
RUN=$ROOT/runs/qwen25vl3b_10k_dfly_recipe
echo "=========== VelaDraft scale run @ $(date '+%F %T %Z') ==========="
echo "--- GPU3 / disk ---"
nvidia-smi -i 3 --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader
df -h /home | tail -1
echo
echo "--- pipeline (tail 12) ---"
tail -n 12 "$ROOT/logs/pipeline.log" 2>/dev/null || echo "(no pipeline log yet)"
echo
if [ -f "$RUN/regen/target_data.jsonl" ] && [ ! -f "$RUN/train/train_log.jsonl" ]; then
  echo "--- regen progress (tail 3) ---"
  tail -n 3 "$ROOT/logs/03_regen.log" 2>/dev/null
  echo
fi
if [ -f "$RUN/train/train_log.jsonl" ]; then
  echo "--- train ---"
  grep -oE 'step [0-9]+/[0-9]+' "$ROOT/logs/04_train.log" 2>/dev/null | tail -1
  tail -n 3 "$RUN/train/train_log.jsonl" | python3 -c '
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print("  step {} loss={:.4f} acc={:.3f} al={:.2f}".format(
        r.get("step"), r.get("loss", float("nan")), r.get("accuracy", float("nan")),
        r.get("accept_len", float("nan"))))'
  echo
fi
if [ -f "$RUN/eval/eval_result.json" ]; then
  echo "--- eval ---"
  python3 - <<PYEOF
import json
r = json.load(open("$RUN/eval/eval_result.json"))
print("  token_exact:", r["equivalence_check"]["token_exact"])
for name, v in r["runs"].items():
    print("  {}: AL={:.4f} speedup={}".format(name, v["mean_al"], v.get("speedup_vs_target_only")))
PYEOF
  echo
fi
if [ -f "$RUN/train/train_summary.json" ]; then
  echo "--- train summary ---"
  python3 - <<PYEOF
import json
s = json.load(open("$RUN/train/train_summary.json"))
print("  steps:", s.get("steps"), " wall_seconds:", round(s.get("wall_seconds", 0)))
fs = s.get("feature_source") or {}
fc = fs.get("feature_cache")
if fc:
    print("  cache hits/misses/stores/evictions:", fc.get("hits"), fc.get("misses"), fc.get("stores"), fc.get("evictions"))
    print("  cache retained/peak GiB:", round((fc.get("retained_bytes") or 0)/2**30, 1), round((fc.get("peak_bytes") or 0)/2**30, 1))
PYEOF
fi
