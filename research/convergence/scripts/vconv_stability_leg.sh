#!/bin/bash
# Stability + canonical seed-42 leg: N steps with mid-run checkpoint copies at
# every save interval (default 200) and one eval over all of them.
# usage: vconv_stability_leg.sh <leg> <gpu> <steps> <save_every> <seed>
set -u
LEG=$1; GPU=$2; STEPS=$3; SAVE=$4; SEED=${5:-42}
RUN=/home/lyh/veladraft-conv/runs/$LEG
STATUS=/home/lyh/veladraft-conv/logs/vconv_status.log
bash /home/lyh/veladraft-conv/scripts/vconv_stage.sh "$LEG" >> "$STATUS" 2>&1 || { echo "stage FAIL $LEG" >> "$STATUS"; exit 1; }
echo "start train $LEG stability steps=$STEPS save=$SAVE seed=$SEED $(date '+%F %H:%M:%S')" >> "$STATUS"
VCONV_GPU=$GPU VCONV_STEPS=$STEPS bash /home/lyh/veladraft-conv/scripts/vconv_run.sh "$LEG" train \
  --set "train.seed=$SEED" --set "train.save_every_steps=$SAVE" >> "$STATUS" 2>&1 &
TPID=$!
declare -A KEPT=()
while kill -0 $TPID 2>/dev/null; do
  last=$(tail -1 "$RUN/train/train_log.jsonl" 2>/dev/null | python3 -c 'import sys,json;print(json.loads(sys.stdin.readline())["step"])' 2>/dev/null || echo 0)
  step=$SAVE
  while [ "$step" -lt "$STEPS" ]; do
    target=$((step + 10))
    if [ "${last:-0}" -ge "$target" ] && [ -z "${KEPT[$step]:-}" ]; then
      cp "$RUN/train/ckpt_last.pt" "$RUN/train/ckpt_$step.pt" && KEPT[$step]=1 \
        && echo "kept ckpt_$step.pt at log-step $last $LEG $(date '+%F %H:%M:%S')" >> "$STATUS"
    fi
    step=$((step + SAVE))
  done
  sleep 5
done
wait $TPID || { echo "train FAIL $LEG" >> "$STATUS"; exit 1; }
echo "train done $LEG $(date '+%F %H:%M:%S')" >> "$STATUS"
if [ -f "$RUN/train/ckpt_last.pt" ]; then rm "$RUN/train/ckpt_last.pt"; fi
CKPTS=$(cd "$RUN/train" && ls ckpt_*.pt | sort -V | paste -sd, -)
echo "start eval  $LEG ckpts=$CKPTS $(date '+%F %H:%M:%S')" >> "$STATUS"
VCONV_GPU=$GPU bash /home/lyh/veladraft-conv/scripts/vconv_run.sh "$LEG" eval \
  --set "eval.checkpoints=[$CKPTS]" >> "$STATUS" 2>&1 || { echo "eval FAIL $LEG" >> "$STATUS"; exit 1; }
for f in "$RUN"/train/ckpt_*.pt; do [ -e "$f" ] && case "$f" in *ckpt_keep*) ;; *) rm "$f";; esac; done
[ -f "$RUN/train/ckpt_init.pt" ] && rm "$RUN/train/ckpt_init.pt"
echo "done  $LEG $(date '+%F %H:%M:%S')" >> "$STATUS"
