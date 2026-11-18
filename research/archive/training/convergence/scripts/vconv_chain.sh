#!/bin/bash
# Idempotent per-leg chain: stage -> train -> [optional mid-ckpt copy] -> eval -> cleanup.
# usage: vconv_chain.sh <leg> <seed> <mode> <sub_batch> <fp32> <bypass>
# env: VCONV_GPU, VCONV_STEPS (default 800), VCONV_KEEP_AT (copy ckpt_last at this step via watcher),
#      VCONV_EVAL_CKPTS (comma list for eval.checkpoints; default ckpt_final.pt)
set -u
LEG=$1; SEED=$2; MODE=$3; SUB=$4; FP32=$5; BYPASS=$6
RUN=/home/lyh/veladraft-conv/runs/$LEG
STATUS=/home/lyh/veladraft-conv/logs/vconv_status.log
STEPS=${VCONV_STEPS:-800}
SETS=(--set "train.seed=$SEED")
if [ "$MODE" != "off" ]; then
  SETS+=(--set "train.regroup.mode=$MODE" --set "train.regroup.sub_batch=$SUB" \
         --set "train.regroup.fp32_weights=$FP32" --set "train.regroup.identity_bypass=$BYPASS")
fi
if [ ! -e "$RUN" ]; then
  bash /home/lyh/veladraft-conv/scripts/vconv_stage.sh "$LEG" >> "$STATUS" 2>&1 || { echo "stage FAIL $LEG" >> "$STATUS"; exit 1; }
fi
if [ ! -f "$RUN/train/train_summary.json" ]; then
  echo "start train $LEG mode=$MODE sub=$SUB fp32=$FP32 bypass=$BYPASS seed=$SEED $(date '+%F %H:%M:%S')" >> "$STATUS"
  if [ -n "${VCONV_KEEP_AT:-}" ]; then
    bash /home/lyh/veladraft-conv/scripts/vconv_run.sh "$LEG" train "${SETS[@]}" \
      --set "train.save_every_steps=$VCONV_KEEP_AT" & TPID=$!
    GUARD=$((VCONV_KEEP_AT + 10))
    while kill -0 $TPID 2>/dev/null; do
      last=$(tail -1 "$RUN/train/train_log.jsonl" 2>/dev/null | python3 -c 'import sys,json;print(json.loads(sys.stdin.readline())["step"])' 2>/dev/null || echo 0)
      if [ "${last:-0}" -ge "$GUARD" ] && [ ! -f "$RUN/train/ckpt_keep.pt" ]; then
        cp "$RUN/train/ckpt_last.pt" "$RUN/train/ckpt_keep.pt" && echo "kept mid ckpt at step~$VCONV_KEEP_AT $LEG $(date '+%F %H:%M:%S')" >> "$STATUS"
      fi
      sleep 5
    done
    wait $TPID || { echo "train FAIL $LEG" >> "$STATUS"; exit 1; }
  else
    bash /home/lyh/veladraft-conv/scripts/vconv_run.sh "$LEG" train "${SETS[@]}" >> "$STATUS" 2>&1 || { echo "train FAIL $LEG" >> "$STATUS"; exit 1; }
  fi
else
  echo "skip train $LEG (train_summary.json present)" >> "$STATUS"
fi
if [ -f "$RUN/train/ckpt_last.pt" ]; then rm "$RUN/train/ckpt_last.pt"; echo "cleaned ckpt_last $LEG" >> "$STATUS"; fi
if [ "${VCONV_SKIP_EVAL:-0}" = "1" ]; then
  echo "skip eval  $LEG (VCONV_SKIP_EVAL=1)" >> "$STATUS"
elif [ ! -f "$RUN/eval/eval_result.json" ]; then
  echo "start eval  $LEG $(date '+%F %H:%M:%S')" >> "$STATUS"
  EVAL_SETS=()
  [ -n "${VCONV_EVAL_CKPTS:-}" ] && EVAL_SETS+=(--set "eval.checkpoints=[$VCONV_EVAL_CKPTS]")
  bash /home/lyh/veladraft-conv/scripts/vconv_run.sh "$LEG" eval "${EVAL_SETS[@]}" >> "$STATUS" 2>&1 || { echo "eval FAIL $LEG" >> "$STATUS"; exit 1; }
else
  echo "skip eval  $LEG (eval_result.json present)" >> "$STATUS"
fi
if [ -f "$RUN/train/ckpt_init.pt" ]; then rm "$RUN/train/ckpt_init.pt"; echo "cleaned ckpt_init $LEG" >> "$STATUS"; fi
if [ -f "$RUN/train/ckpt_final.pt" ] && [ "${VCONV_KEEP_CKPT_FINAL:-0}" != "1" ]; then rm "$RUN/train/ckpt_final.pt"; echo "cleaned ckpt_final $LEG (eval_result retained)" >> "$STATUS"; fi
if [ -f "$RUN/train/ckpt_keep.pt" ] && [ "${VCONV_KEEP_CKPT:-0}" != "1" ]; then rm "$RUN/train/ckpt_keep.pt"; echo "cleaned ckpt_keep $LEG" >> "$STATUS"; fi
echo "done  $LEG $(date '+%F %H:%M:%S')" >> "$STATUS"
