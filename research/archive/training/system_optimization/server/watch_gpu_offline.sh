#!/bin/bash
# Waiter: run the offline baseline on the first genuinely idle A6000.
# Same guards/lock as watch_gpu_closing.sh.
set -u
LANE=/home/lyh/vlm-systems
mkdir -p "$LANE/logs"
exec 9>"$LANE/lane.lock"
flock -n 9 || { echo "$(date +'[%F %T]') lane busy, offline waiter exits" >> "$LANE/logs/offline_waiter.log"; exit 0; }
WATCH_HOURS="${WATCH_HOURS:-14}"
deadline=$(( $(date +%s) + WATCH_HOURS*3600 ))
echo "$(date +'[%F %T]') offline waiter started (hours=$WATCH_HOURS)" >> "$LANE/logs/offline_waiter.log"
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -f "$LANE/STOP_WAITER" ]; then echo "$(date +'[%F %T]') STOP_WAITER found" >> "$LANE/logs/offline_waiter.log"; exit 0; fi
  for dev in 0 1 2 3; do
    procs=$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free_mib=$(nvidia-smi -i "$dev" --query-gpu=memory.free --format=csv,noheader,nounits)
    util=$(nvidia-smi -i "$dev" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$procs" -eq 0 ] && [ "$free_mib" -gt 40000 ] && [ "$util" -lt 5 ]; then
      sleep 60
      procs2=$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
      if [ "$procs2" -eq 0 ]; then
        echo "$(date +'[%F %T]') GPU $dev idle (free ${free_mib} MiB, util ${util}%) -> launch offline baseline" >> "$LANE/logs/offline_waiter.log"
        bash "$LANE/run_offline_baseline.sh" "$dev" >> "$LANE/logs/offline_run.log" 2>&1
        echo "$(date +'[%F %T]') offline baseline returned (GPU $dev)" >> "$LANE/logs/offline_waiter.log"
        exit 0
      fi
    fi
  done
  sleep 120
done
echo "$(date +'[%F %T]') offline waiter timeout after ${WATCH_HOURS}h" >> "$LANE/logs/offline_waiter.log"
