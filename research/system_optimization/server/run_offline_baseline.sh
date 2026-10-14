#!/bin/bash
# Offline baseline for the H-C1 cache claim (goal doc 8.3; REPORT 4.6):
#   Leg e  : FULL offline pipeline on the S3 staged inputs - capture stage
#            (single-sample teacher forwards; bf16 shards written to
#            run_dir/features, ~6.5 GiB) + train from the store (6 epochs).
#            End-to-end includes build + write + read I/O.
#   Leg e2 : reuse scenario - train only from e's store via --features, in a
#            fresh run dir; reports the store-reuse wall.
# Disk guard >=15 GiB before capture (store ~6.5 GiB); idempotent per leg.
# Timing: /usr/bin/time end-to-end per leg + stage walls in the summaries and
# the stage log; progress at logs/offline_progress.log.
set -u
LANE=/home/lyh/vlm-systems
REPO=/home/lyh/VelaDraft-systems
DEV="${1:?device index}"
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
LOG=$LANE/logs
RUNS=$LANE/runs/s3
E=$RUNS/e_off_store
E2=$RUNS/e2_off_reuse
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_FLEX_FORCE_MAIN=1
export VLM_DFLASH_SKIP_RESUME_CKPT=1
export VLM_DFLASH_LLAVA_JSON=/home/lyh/vlm-systems/data/llava_v1safe.json
export VLM_DFLASH_COCO_IMAGES=/home/dhz/Data/COCO-caption/train2017
export CUDA_VISIBLE_DEVICES="$DEV"
plog() { echo "$(date +'[%F %T]') $*" >> "$LOG/offline_progress.log"; }
free_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

plog "offline baseline start on GPU $DEV (free $(free_gb)G)"

# ---- leg e: build (capture) + train ----
if [ -f "$E/train/train_summary.json" ]; then plog "e already done; skip"; else
  if [ "$(free_gb)" -lt 15 ]; then plog "e skipped: <15G free"; exit 0; fi
  mkdir -p "$E"
  if [ ! -f "$E/contract.json" ]; then cp -a "$RUNS/a_off/data" "$RUNS/a_off/regen" "$RUNS/a_off/contract.json" "$E/"; fi
  rm -rf "$E/train" "$E/features"
  plog "e start (capture+train; free $(free_gb)G)"
  /usr/bin/time -v -o "$LOG/offline_e.time" \
  $PY -m vlm_dflash.cli train --mode offline --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$E" \
    --set train.overfit_sanity.enabled=false --set train.log_every_steps=10 \
    --set train.save_every_steps=100000 --set train.micro_batch_size=8 \
    > "$LOG/offline_e.log" 2>&1
  plog "e exit=$?"
  plog "store size: $(du -sh "$E/features" 2>/dev/null | cut -f1)"
fi

# ---- leg e2: reuse the store (train only) ----
if [ -f "$E2/train/train_summary.json" ]; then plog "e2 already done; skip"; else
  if [ ! -f "$E/features/index.jsonl" ]; then plog "e2 skipped: no store"; exit 0; fi
  mkdir -p "$E2"
  if [ ! -f "$E2/contract.json" ]; then cp -a "$RUNS/a_off/data" "$RUNS/a_off/regen" "$RUNS/a_off/contract.json" "$E2/"; fi
  rm -rf "$E2/train"
  plog "e2 start (reuse e_off_store/features; free $(free_gb)G)"
  /usr/bin/time -v -o "$LOG/offline_e2.time" \
  $PY -m vlm_dflash.cli train --mode offline --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$E2" --features "$E/features" \
    --set train.overfit_sanity.enabled=false --set train.log_every_steps=10 \
    --set train.save_every_steps=100000 --set train.micro_batch_size=8 \
    > "$LOG/offline_e2.log" 2>&1
  plog "e2 exit=$?"
fi

for leg in e_off_store e2_off_reuse; do
  d="$RUNS/$leg"
  [ -f "$d/train/train_summary.json" ] && cp "$d/train/train_summary.json" "$LOG/offline_${leg}_summary.json"
  [ -f "$d/train/train_log.jsonl" ] && cp "$d/train/train_log.jsonl" "$LOG/offline_${leg}_train_log.jsonl"
done
$PY - >> "$LOG/offline_progress.log" 2>&1 <<'PYEOF'
import json
from pathlib import Path
L = Path("/home/lyh/vlm-systems/logs")
for t in ["offline_e_off_store_summary", "offline_e2_off_reuse_summary",
          "s3_a_off_summary", "s3_b_on_summary"]:
    p = L / f"{t}.json"
    if p.exists():
        s = json.load(open(p))
        print(f"  {t}: wall={s.get('wall_seconds')} steps={s.get('steps')} final_loss={s.get('final_loss')}")
PYEOF
plog "offline baseline DONE on GPU $DEV"
