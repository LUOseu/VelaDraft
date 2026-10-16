#!/bin/bash
# VelaDraft scale run (2026-10-15): DFly upstream_two_stage recipe on a 10K-slice
# corpus with train.feature_mode=online + host-RAM feature cache.
# Stages: smoke -> prepare -> regen -> train -> eval -> export
#   usage: bash run_scale_10k.sh [first_stage]   (default: smoke; reruns are skipped per stage)
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
CFG=$REPO/configs/scale/qwen2_5_vl_3b_10k_recipe.yaml
RUN=$ROOT/runs/qwen25vl3b_10k_dfly_recipe
SMOKE=$ROOT/runs/recipe_online_smoke
LOGS=$ROOT/logs
GPU=${SCALE_GPU:-3}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
mkdir -p "$LOGS"
cd "$REPO" || exit 1

PIPE=$LOGS/pipeline.log
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$PIPE"; }

guard() {
  local attempt util free dh
  for attempt in $(seq 1 10); do
    util=$(nvidia-smi -i $GPU --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
    dh=$(df -BG /home | tail -1 | awk '{print $4}' | tr -dc '0-9')
    if [ "${util:-100}" -le 5 ] && [ "${free:-0}" -ge 40960 ] && [ "${dh:-0}" -ge 12 ]; then
      say "guard OK gpu=$GPU util=${util}% free=${free}MiB home=${dh}G"
      return 0
    fi
    say "guard wait $attempt/10 gpu=$GPU util=${util}% free=${free}MiB home=${dh}G"
    sleep 30
  done
  say "GUARD FAIL gpu=$GPU util=${util}% free=${free}MiB home=${dh}G"
  return 1
}

STAGE="${1:-smoke}"

# ---------- 0. smoke: online + recipe plumbing on the 2K slice ----------
if [ "$STAGE" = smoke ]; then
  guard || exit 1
  say "STAGE smoke: online+recipe, 64 rows x 40 steps + 8-prompt eval"
  rm -rf "$SMOKE"; mkdir -p "$SMOKE"
  cp -al /home/lyh/vlm-dflash/runs/qwen25vl3b_2k/data "$SMOKE/data"
  cp -al /home/lyh/vlm-dflash/runs/qwen25vl3b_2k/regen "$SMOKE/regen"
  rc=0
  $PY scripts/train_dflash.py --config "$CFG" --run-dir "$SMOKE" --limit 64 --steps 40 \
      --skip-sanity > "$LOGS/01_smoke_train.log" 2>&1 || rc=$?
  say "smoke train rc=$rc (log: logs/01_smoke_train.log)"
  [ $rc -ne 0 ] && exit 1
  rc=0
  $PY scripts/eval_dflash.py --config "$CFG" --run-dir "$SMOKE" --num-prompts 8 \
      --set "eval.checkpoints=[ckpt_final.pt]" > "$LOGS/01_smoke_eval.log" 2>&1 || rc=$?
  say "smoke eval rc=$rc (log: logs/01_smoke_eval.log)"
  [ $rc -ne 0 ] && exit 1
  te=$($PY -c "import json;print(json.load(open('$SMOKE/eval/eval_result.json'))['equivalence_check']['token_exact'])" 2>/dev/null)
  say "smoke eval token_exact=$te"
  [ "$te" != "True" ] && { say "smoke gate FAILED (token_exact != True)"; exit 1; }
  STAGE=prepare
fi

# ---------- 1. prepare: 10K slice ----------
if [ "$STAGE" = prepare ]; then
  guard || exit 1
  if [ -f "$RUN/data/data.jsonl" ]; then
    say "prepare already present, skip"
  else
    say "STAGE prepare: 10K slice"
    rc=0
    $PY scripts/prepare_data.py --config "$CFG" --run-dir "$RUN" --limit 10000 > "$LOGS/02_prepare.log" 2>&1 || rc=$?
    say "prepare rc=$rc (log: logs/02_prepare.log)"
    [ $rc -ne 0 ] && exit 1
  fi
  STAGE=regen
fi

# ---------- 2. regen: 10K greedy target generations (~2h) ----------
if [ "$STAGE" = regen ]; then
  guard || exit 1
  if [ -f "$RUN/regen/target_data.jsonl" ]; then
    say "regen already present, skip"
  else
    say "STAGE regen: 10K rows, batch 8, greedy, max_new 160 (~2h)"
    rc=0
    $PY scripts/generate_target_data.py --config "$CFG" --run-dir "$RUN" --limit 10000 > "$LOGS/03_regen.log" 2>&1 || rc=$?
    n=$(wc -l < "$RUN/regen/target_data.jsonl" 2>/dev/null || echo 0)
    say "regen rc=$rc kept=$n (log: logs/03_regen.log)"
    [ $rc -ne 0 ] && exit 1
  fi
  STAGE=train
fi

# ---------- 3. train: 12 epochs online (mid-ckpt captured at half budget) ----------
if [ "$STAGE" = train ]; then
  guard || exit 1
  if [ -f "$RUN/train/ckpt_final.pt" ]; then
    say "train already present, skip"
  else
    RESUME=""
    if [ -f "$RUN/train/ckpt_last.pt" ]; then RESUME="--resume"; say "train: resuming from ckpt_last"; fi
    say "STAGE train: 12 epochs, feature_mode=online, cache=96GiB"
    rc=0
    $PY scripts/train_dflash.py --config "$CFG" --run-dir "$RUN" --skip-sanity $RESUME \
        > "$LOGS/04_train.log" 2>&1 &
    TPID=$!
    mid_done=0
    while kill -0 $TPID 2>/dev/null; do
      line=$(grep -oE 'step [0-9]+/[0-9]+' "$LOGS/04_train.log" 2>/dev/null | tail -1)
      step=$(echo "$line" | awk '{split($2,a,"/"); print a[1]}'); step=${step:-}
      total=$(echo "$line" | awk '{split($2,a,"/"); print a[2]}'); total=${total:-}
      if [ -n "$step" ] && [ -n "$total" ]; then
        mid=$((total / 2))
        if [ "$step" -ge $((mid + 5)) ] && [ "$mid_done" -eq 0 ] && [ -f "$RUN/train/ckpt_last.pt" ]; then
          cp "$RUN/train/ckpt_last.pt" "$RUN/train/ckpt_mid.pt" && mid_done=1 \
            && say "kept ckpt_mid.pt at log-step $step/$total (half-budget point)"
        fi
      fi
      sleep 10
    done
    wait $TPID; rc=$?
    say "train rc=$rc (log: logs/04_train.log)"
    [ $rc -ne 0 ] && exit 1
  fi
  STAGE=eval
fi

# ---------- 4. eval: 128 prompts x 128 tokens, sequential + parallel diagnostic ----------
if [ "$STAGE" = eval ]; then
  guard || exit 1
  if [ -f "$RUN/eval/eval_result.json" ]; then
    say "eval already present, skip"
  else
    CKPTS=""
    for c in ckpt_mid.pt ckpt_final.pt ckpt_init.pt; do
      [ -f "$RUN/train/$c" ] && CKPTS="$CKPTS,$c"
    done
    CKPTS=${CKPTS#,}
    say "STAGE eval: ckpts=[$CKPTS]"
    rc=0
    $PY scripts/eval_dflash.py --config "$CFG" --run-dir "$RUN" --set "eval.checkpoints=[$CKPTS]" \
        > "$LOGS/05_eval.log" 2>&1 || rc=$?
    say "eval rc=$rc (log: logs/05_eval.log)"
    [ $rc -ne 0 ] && exit 1
    [ -f "$RUN/train/ckpt_last.pt" ] && [ -f "$RUN/train/ckpt_final.pt" ] && rm -f "$RUN/train/ckpt_last.pt" && say "pruned ckpt_last.pt (eval receipts kept)"
    $PY - >> "$PIPE" 2>&1 <<PYEOF
import json
r = json.load(open("$RUN/eval/eval_result.json"))
print("eval token_exact:", r["equivalence_check"]["token_exact"])
for name, v in r["runs"].items():
    sp = v.get("speedup_vs_target_only")
    print(f"  {name}: AL={v['mean_al']:.4f} speedup={sp}")
PYEOF
  fi
  STAGE=export
fi

# ---------- 5. export (HF format) ----------
if [ "$STAGE" = export ]; then
  if [ -d "$RUN/export" ]; then
    say "export already present, skip"
  else
    say "STAGE export: hf format"
    rc=0
    $PY -m vlm_dflash export "$RUN" --format hf > "$LOGS/06_export.log" 2>&1 || rc=$?
    say "export rc=$rc (log: logs/06_export.log)"
  fi
fi

say "PIPELINE COMPLETE (stage=$STAGE)"
