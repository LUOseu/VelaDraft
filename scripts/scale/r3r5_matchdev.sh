#!/bin/bash
# Post-training supplement, deliberately separate from r3r5_postrun.sh (that one
# is already running and bash re-reads scripts incrementally).
#
# 1. Full 256-row dev confirmation of every *matched* checkpoint step that both
#    arms still have on disk, so R3-vs-R5 at equal progress is compared on the
#    confirm split, not only on the 64-row periodic subset.
# 2. Per-arm best-checkpoint confirmation when the two best steps differ.
# 3. Parallel-verification speed pass (sequential is measured by postrun.sh);
#    a parallel result that is not token-exact is reported as such, never as a
#    faithful speedup.
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
BASE=$ROOT/runs/r3r5_20k
LOGS=$BASE/logs
N=${N:-20000}
CONFIRM_PROMPTS=${CONFIRM_PROMPTS:-256}
MAX_MATCHED=${MAX_MATCHED:-3}
export PYTHONPATH=$REPO VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct PYTHONUNBUFFERED=1
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %H:%M:%S')] matchdev: $*" | tee -a "$LOGS/50_matchdev.log"; }

R3=$ROOT/runs/r3r5_n$N/r3
R5=$ROOT/runs/r3r5_n$N/r5
mkdir -p "$BASE/analysis"

say "waiting for both drivers to finish"
for _ in $(seq 1 2880); do
  d3=$(grep -c '"event": "run_complete"' "$R3/driver/progress.jsonl" 2>/dev/null || true)
  d5=$(grep -c '"event": "run_complete"' "$R5/driver/progress.jsonl" 2>/dev/null || true)
  [ "${d3:-0}" -ge 1 ] && [ "${d5:-0}" -ge 1 ] && break
  sleep 60
done
say "drivers finished (r3=${d3:-0} r5=${d5:-0})"

# r3r5_postrun.sh also measures speed on the freest GPU. Serialise the two so
# that every speed number is measured on an exclusive card: wait for postrun's
# own speed pass to be over (bounded, so a broken postrun cannot wedge this).
say "waiting for postrun's exclusive speed pass to finish (<=3h)"
for _ in $(seq 1 180); do
  grep -q "POSTRUN COMPLETE" "$LOGS/30_postrun.log" 2>/dev/null && break
  sleep 60
done
say "postrun state: $(grep -c 'POSTRUN COMPLETE' "$LOGS/30_postrun.log" 2>/dev/null || echo 0) complete-marker(s)"

pick_gpu() {  # freest GPU with >= 30 GiB free
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, '$2+0 >= 30720 {print $2+0, $1}' | sort -nr | head -1 | awk '{print $2}'
}

list_steps() {  # "$1"=run dir -> "step:ckptfile" lines, ascending, newest-last
  local run=$1
  local last
  last=$(grep -o '"step": [0-9]*' "$run/driver/progress.jsonl" 2>/dev/null | tail -1 | grep -o '[0-9]*')
  for f in "$run"/train/ckpt_step*.pt; do
    [ -e "$f" ] || continue
    local b; b=$(basename "$f")
    echo "$(echo "$b" | sed -n 's/ckpt_step0*\([0-9]*\)\.pt/\1/p'):$b"
  done
  [ -e "$run/train/ckpt_final.pt" ] && echo "${last:-0}:ckpt_final.pt"
}

# ---------------- matched full-dev confirmations ----------------
mapfile -t S3 < <(list_steps "$R3" | sort -n)
mapfile -t S5 < <(list_steps "$R5" | sort -n)
say "r3 ckpts: ${S3[@]:-none}"
say "r5 ckpts: ${S5[@]:-none}"

comm_steps=$(comm -12 <(printf '%s\n' "${S3[@]}" | cut -d: -f1 | sort -u) \
                       <(printf '%s\n' "${S5[@]}" | cut -d: -f1 | sort -u) | sort -n | tail -"$MAX_MATCHED")
say "matched steps for full-dev: $(echo $comm_steps)"

already_paired() {  # $1 = step; true when postrun already confirmed this step on both arms
  [ -f "$BASE/analysis/paired_dev_step$1.json" ] && return 0
  return 1
}

for step in $comm_steps; do
  if already_paired "$step"; then
    say "step $step: already confirmed on both arms by the driver (paired_dev_step$step.json); skipping"
    continue
  fi
  if [ "$step" = "$(echo "$comm_steps" | tr ' ' '\n' | sort -n | tail -1)" ] \
     && [ -f "$BASE/analysis/paired_dev_final.json" ]; then
    say "step $step is the final step and is already paired as dev_final; skipping"
    continue
  fi
  gpu=$(pick_gpu)
  if [ -z "${gpu:-}" ]; then say "no free GPU for matched step $step; skipping"; continue; fi
  for arm in r3 r5; do
    run=$ROOT/runs/r3r5_n$N/$arm
    ck=$("$PY" -c "
import pathlib,re,sys
step=int('$step')
for f in sorted(pathlib.Path('$run/train').glob('ckpt_step*.pt')):
    m=re.match(r'ckpt_step0*([0-9]+)\.pt\$', f.name)
    if m and int(m.group(1))==step:
        print(f.name); break
")
    [ -n "${ck:-}" ] || { say "$arm step $step: no ckpt file"; continue; }
    say "$arm full-dev confirm $ck on gpu $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_dflash.py \
        --config configs/scale/${arm}_dfly_n$N.yaml --run-dir "$run" \
        --num-prompts "$CONFIRM_PROMPTS" --verification sequential \
        --tag "mdev_step$step" --set "eval.checkpoints=[$ck]" \
        >> "$LOGS/51_matchdev_${arm}_step${step}.log" 2>&1
    say "$arm step $step rc=$?"
  done
done

# ---------------- per-arm best (when best steps differ) ----------------
for arm in r3 r5; do
  run=$ROOT/runs/r3r5_n$N/$arm
  sel=$run/driver/selection.json
  [ -f "$sel" ] || { say "$arm: no selection.json yet"; continue; }
  best=$($PY -c "import json;print(json.load(open('$sel')).get('best_ckpt') or '')")
  [ -n "$best" ] || { say "$arm: no best_ckpt"; continue; }
  if $PY -c "
import glob, json, sys
for f in glob.glob('$run/eval/eval_result_dev_*.json'):
    if '$best' in json.load(open(f)).get('runs', {}):
        sys.exit(0)
sys.exit(1)
"; then
    say "$arm best=$best already has a full-dev confirmation from the driver; skipping"
    continue
  fi
  gpu=$(pick_gpu)
  if [ -z "${gpu:-}" ]; then say "no free GPU for $arm best $best; skipping"; continue; fi
  say "$arm full-dev confirm best=$best on gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_dflash.py \
      --config configs/scale/${arm}_dfly_n$N.yaml --run-dir "$run" \
      --num-prompts "$CONFIRM_PROMPTS" --verification sequential \
      --tag "mbest_$best" --set "eval.checkpoints=[$best]" \
      >> "$LOGS/51_matchdev_${arm}_best.log" 2>&1
  say "$arm best $best rc=$?"
done

# ---------------- parallel-verification speed pass ----------------
gpu=$(pick_gpu)
if [ -z "${gpu:-}" ]; then
  say "parallel speed: no free GPU (need >=30GiB); skipping"
else
  for arm in r3 r5; do
    run=$ROOT/runs/r3r5_n$N/$arm
    ck=ckpt_final.pt
    [ -e "$run/train/ckpt_final.pt" ] || ck=$($PY -c "import json;print(json.load(open('$run/driver/selection.json')).get('best_ckpt') or 'ckpt_final.pt')" 2>/dev/null || echo ckpt_final.pt)
    say "parallel speed $arm ($ck) on gpu $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/scale/speed_paired.py \
        --config configs/scale/${arm}_dfly_n$N.yaml --run-dir "$run" --ckpt "$ck" \
        --num-prompts 32 --warmup-prompts 3 --max-new-tokens 128 \
        --verification parallel --tag "${arm}_speed_final_parallel" \
        > "$LOGS/52_speed_parallel_$arm.log" 2>&1
    say "parallel speed $arm rc=$?"
  done
fi

# ---------------- paired analyses ----------------
for tag in $(ls "$R3/eval" 2>/dev/null | sed -n 's/^eval_samples_\(mdev_step[0-9]*\)\.jsonl$/\1/p' | sort); do
  [ -f "$R5/eval/eval_samples_$tag.jsonl" ] || continue
  $PY scripts/scale/analyze_paired.py \
      --samples-a "$R3/eval/eval_samples_$tag.jsonl" \
      --samples-b "$R5/eval/eval_samples_$tag.jsonl" \
      --label-a R3 --label-b R5 --out "$BASE/analysis/paired_full_dev_$tag.json" \
      >> "$LOGS/53_matchdev_analyze.log" 2>&1
  say "paired analysis $tag done"
done
say "MATCHDEV COMPLETE"
