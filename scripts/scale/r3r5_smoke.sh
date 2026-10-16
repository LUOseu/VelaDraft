#!/bin/bash
# R3/R5 startup sanity: 3L train (+overfit sanity, milestones) -> milestone eval-load
# -> real 8-prompt sequential eval on the pilot corpus, 5L build/train (+sanity,
# gradient coverage) -> 5L snapshot eval-load, then kill/resume to completion.
#
# The run dirs live under /dev/shm: /home has ~6G free on this host, while the
# milestones + ckpt_init of a single stage need up to ~4G transiently.
# The corpus is a *real copy* of the pilot regen output (cross-filesystem: no
# hardlinks), i.e. the exact artifacts the full run will consume.
#
# usage: [SMOKE_GPU=1] bash r3r5_smoke.sh [3l|eval|5l|resume|all]
set -u
ROOT=/home/lyh/vlm-scale
REPO=/home/lyh/VelaDraft-scale
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
BASE=$ROOT/runs/r3r5_20k
PILOT=$BASE/regen_pilot
SMOKE=${SMOKE_DIR:-/dev/shm/r3r5_smoke}
SMOKE_GPU=${SMOKE_GPU:-1}
# Optional batch-geometry override for the smoke runs (e.g. MBS=4 ACCUM=2 when
# the audit session shares the card). Records what it used in the log.
SMOKE_SETS=${SMOKE_SETS:-}
CFG3=$REPO/scripts/scale/r3r5_smoke_r3.yaml
CFG5=$REPO/scripts/scale/r3r5_smoke_r5.yaml
LOGS=$BASE/logs
export PYTHONPATH=$REPO VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct PYTHONUNBUFFERED=1
mkdir -p "$LOGS" "$SMOKE"

say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOGS/smoke.log"; }

stage_corpus() {  # $1 = run tag
  mkdir -p "$SMOKE/$1"
  cp -a "$PILOT/data" "$SMOKE/$1/data"
  cp -a "$PILOT/regen" "$SMOKE/$1/regen"
}

STAGE="${1:-all}"
cd "$REPO" || exit 1

if [ "$STAGE" = 3l ] || [ "$STAGE" = all ]; then
  say "SMOKE 3L: 40 steps on the pilot corpus (1230 rows), overfit sanity + milestone snapshots, gpu=$SMOKE_GPU"
  rm -rf "$SMOKE/r3"; stage_corpus r3
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY scripts/train_dflash.py --config "$CFG3" --run-dir "$SMOKE/r3" \
      --steps 40 $SMOKE_SETS > "$LOGS/10_smoke3l_train.log" 2>&1
  rc=$?
  say "smoke 3L train rc=$rc"
  [ $rc -ne 0 ] && exit 1
  ls "$SMOKE/r3/train/" | tr '\n' ' ' | tee -a "$LOGS/smoke.log"; echo
  $PY - >> "$LOGS/smoke.log" <<PYEOF
import json
rows = [json.loads(l) for l in open("$SMOKE/r3/train/train_log.jsonl") if l.strip()]
first, last = rows[0], rows[-1]
print(f"  3L train: steps={len(rows)} first_step={first['step']} last_step={last['step']} "
      f"first_loss={first.get('loss')} last_loss={last.get('loss')} "
      f"grad_norm_last={last.get('grad_norm')} lr_last={last.get('lr')}")
sane = json.load(open("$SMOKE/r3/train/overfit_sanity.json")) if __import__("pathlib").Path("$SMOKE/r3/train/overfit_sanity.json").exists() else None
print("  3L overfit sanity:", json.dumps(sane)[:400] if sane else "missing")
PYEOF
fi

if [ "$STAGE" = eval ] || [ "$STAGE" = all ]; then
  say "SMOKE eval: load a milestone snapshot + ckpt_final, 8-prompt real eval, token_exact gate"
  snap=$(ls "$SMOKE/r3/train/" | grep -E '^ckpt_step' | head -1)
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY scripts/eval_dflash.py --config "$CFG3" --run-dir "$SMOKE/r3" \
      --num-prompts 8 --verification sequential --tag smoke \
      --set "eval.checkpoints=[$snap,ckpt_final.pt]" > "$LOGS/11_smoke_eval.log" 2>&1
  rc=$?
  say "smoke eval rc=$rc (checkpoints: $snap, ckpt_final.pt)"
  [ $rc -ne 0 ] && exit 1
  $PY - >> "$LOGS/smoke.log" <<PYEOF
import json
r = json.load(open("$SMOKE/r3/eval/eval_result_smoke.json"))
print("  token_exact:", r["equivalence_check"]["token_exact"])
for name, v in r["runs"].items():
    print(f"  {name}: AL={v['mean_al']:.4f} rounds={v['verify_steps']} tfw={v['target_forward_calls']} "
          f"slot0={v['per_slot_acceptance'][0]:.3f} new_tokens/tfw={v['tokens_per_target_forward']:.3f}")
PYEOF
fi

if [ "$STAGE" = 5l ] || [ "$STAGE" = all ]; then
  say "SMOKE 5L: 24 steps on the pilot corpus + overfit sanity + gradient coverage + snapshot eval-load"
  rm -rf "$SMOKE/r5"; stage_corpus r5
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY scripts/train_dflash.py --config "$CFG5" --run-dir "$SMOKE/r5" \
      --steps 24 $SMOKE_SETS > "$LOGS/12_smoke5l_train.log" 2>&1
  rc=$?
  say "smoke 5L train rc=$rc"
  [ $rc -ne 0 ] && exit 1
  $PY - >> "$LOGS/smoke.log" <<PYEOF
import json
cov = json.load(open("$SMOKE/r5/train/gradient_coverage.json"))
print(f"  5L gradient coverage: {cov['params_with_grad']}/{cov['trainable_tensors']} tensors, "
      f"fraction={cov['fraction_with_grad']:.3f}, groups={[g['group'] for g in cov['groups'] if g['params_with_grad']]}")
rows = [json.loads(l) for l in open("$SMOKE/r5/train/train_log.jsonl") if l.strip()]
print(f"  5L train: steps={len(rows)} last_step={rows[-1]['step']} last_loss={rows[-1].get('loss')}")
PYEOF
  snap5=$(ls "$SMOKE/r5/train/" | grep -E '^ckpt_step' | head -1)
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY scripts/eval_dflash.py --config "$CFG5" --run-dir "$SMOKE/r5" \
      --num-prompts 4 --verification sequential --tag smoke5l \
      --set "eval.checkpoints=[${snap5:-ckpt_final.pt}]" > "$LOGS/13_smoke5l_eval.log" 2>&1
  rc=$?
  say "smoke 5L eval-load rc=$rc (ckpt: ${snap5:-ckpt_final.pt})"
  [ $rc -ne 0 ] && exit 1
fi

if [ "$STAGE" = resume ] || [ "$STAGE" = all ]; then
  say "SMOKE resume: launch a 60-step run, kill after the first saved state, resume to completion"
  rm -rf "$SMOKE/resume"; stage_corpus resume
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU nohup $PY scripts/train_dflash.py --config "$CFG3" --run-dir "$SMOKE/resume" \
      --steps 60 --skip-sanity $SMOKE_SETS > "$LOGS/14_smoke_resume_a.log" 2>&1 &
  tp=$!
  for _ in $(seq 1 120); do
    s=$(grep -oE '"step": [0-9]+' "$SMOKE/resume/train/train_log.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    [ -n "${s:-}" ] && [ "$s" -ge 30 ] && break
    sleep 5
  done
  kill -9 $tp 2>/dev/null
  sleep 3
  last=$(grep -oE '"step": [0-9]+' "$SMOKE/resume/train/train_log.jsonl" | tail -1 | grep -oE '[0-9]+')
  say "killed at logged step ${last:-?}; resuming"
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY scripts/train_dflash.py --config "$CFG3" --run-dir "$SMOKE/resume" \
      --steps 60 --skip-sanity --resume $SMOKE_SETS > "$LOGS/15_smoke_resume_b.log" 2>&1
  rc=$?
  final=$(grep -oE '"step": [0-9]+' "$SMOKE/resume/train/train_log.jsonl" | tail -1 | grep -oE '[0-9]+')
  say "resume rc=$rc final logged step=$final (expect 60), ckpt_last=$($PY -c "import torch;print(torch.load('$SMOKE/resume/train_state/ckpt_last.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo '?')"
  [ $rc -ne 0 ] && exit 1
  [ "${final:-0}" -ne 60 ] && { say "RESUME STEP MISMATCH"; exit 1; }
fi

if [ "$STAGE" = resume5l ]; then
  # Same kill/resume gate as ``resume`` but on the 5-layer config (the depth the
  # R5 run uses); SMOKE_SETS can pin the exact geometry under test.
  say "SMOKE resume5l: launch a 60-step 5L run, kill after the first saved state, resume to completion sets=${SMOKE_SETS:-none}"
  rm -rf "$SMOKE/resume5l"; stage_corpus resume5l
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU nohup $PY scripts/train_dflash.py --config "$CFG5" --run-dir "$SMOKE/resume5l" \
      --steps 60 --skip-sanity $SMOKE_SETS > "$LOGS/16_smoke_resume5l_a.log" 2>&1 &
  tp=$!
  for _ in $(seq 1 240); do
    s=$(grep -oE '"step": [0-9]+' "$SMOKE/resume5l/train/train_log.jsonl" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    [ -n "${s:-}" ] && [ "$s" -ge 30 ] && break
    sleep 5
  done
  kill -9 $tp 2>/dev/null
  sleep 3
  last=$(grep -oE '"step": [0-9]+' "$SMOKE/resume5l/train/train_log.jsonl" | tail -1 | grep -oE '[0-9]+')
  say "5L killed at logged step ${last:-?}; resuming"
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU $PY scripts/train_dflash.py --config "$CFG5" --run-dir "$SMOKE/resume5l" \
      --steps 60 --skip-sanity --resume $SMOKE_SETS > "$LOGS/16_smoke_resume5l_b.log" 2>&1
  rc=$?
  final=$(grep -oE '"step": [0-9]+' "$SMOKE/resume5l/train/train_log.jsonl" | tail -1 | grep -oE '[0-9]+')
  say "resume5l rc=$rc final logged step=$final (expect 60), ckpt_last=$($PY -c "import torch;print(torch.load('$SMOKE/resume5l/train_state/ckpt_last.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo '?')"
  [ $rc -ne 0 ] && exit 1
  [ "${final:-0}" -ne 60 ] && { say "RESUME5L STEP MISMATCH"; exit 1; }
fi

say "SMOKE COMPLETE (stage=$STAGE) sets=${SMOKE_SETS:-none}"
