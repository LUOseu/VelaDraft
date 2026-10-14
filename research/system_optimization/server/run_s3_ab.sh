#!/bin/bash
# S3 — H-C1 canonical 2K paired A/B (LLaVA 2000-row pool -> 1050 regen rows,
# 6 epochs, micro-batch 8). Leg a_off: fresh dir; this run also BUILDS the
# master data+regen (prepare ~3.5 min + regen ~23.5 min). Leg b_on: copies
# a_off's data+regen+contract (including stage manifests) so staged inputs are
# reused, not rebuilt; cache ON 16 GiB. All other settings identical.
# Guards (amended 2026-10-14 after S2c; REPORT.md section 4.5): skip unless
# the amended H-C1 gate passes - (a) S2 cache run had hits; (b) the S2c
# fixed-order pair is ckpt sha256-identical (cache value-exactness under a
# stable grouping); (c) the S2 shuffled pair has an identical loss /
# valid_tokens sequence with bounded (<=1e-3) grad_norm drift. The original
# "sha parity under the shuffled loader" gate was retired: probe 2 + S2c
# showed it tested loader-grouping stability, not the cache.
# require >=5 GiB free on /home before each leg (resume-ckpt writes off); run once (rerun guard).
set -u
LANE=/home/lyh/vlm-systems
REPO=/home/lyh/VelaDraft-systems
DEV="${1:?device index}"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export VLM_DFLASH_TARGET=/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_FLEX_FORCE_MAIN=1
export VLM_DFLASH_SKIP_RESUME_CKPT=1
export VLM_DFLASH_LLAVA_JSON=/home/lyh/vlm-systems/data/llava_v1safe.json
export VLM_DFLASH_COCO_IMAGES=/home/dhz/Data/COCO-caption/train2017
PY=/home/lyh/miniforge3/envs/c1vllm/bin/python
export CUDA_VISIBLE_DEVICES="$DEV"
RUNS=$LANE/runs/s3
LOG=$LANE/logs
mkdir -p "$RUNS" "$LOG"
ts() { date +"[%Y-%m-%d %H:%M:%S]"; }
plog() { echo "$(ts) $*" >> "$LOG/s3_progress.log"; }
free_gb() { local v; v=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9'); echo "${v:-0}"; }

if [ -f "$LOG/s3_b_on_summary.json" ]; then plog "S3 already has a b_on summary; skipping"; exit 0; fi

gate() {
  $PY - <<'PY'
import json, sys
L = "/home/lyh/vlm-systems/logs"
try:
    s = json.load(open(f"{L}/s2_b1_summary.json"))
except Exception as exc:
    print(f"gate: s2_b1_summary.json unreadable: {exc}"); sys.exit(1)
fc = (s.get("feature_source") or {}).get("feature_cache") or {}
if not fc.get("enabled") or int(fc.get("hits", 0)) <= 0:
    print(f"gate: s2_b1 feature_cache not working: {fc}"); sys.exit(1)
hashes = {}
try:
    for line in open(f"{L}/s2_ckpt_hashes.txt"):
        h, p = line.split()
        hashes[p] = h
except Exception as exc:
    print(f"gate: cannot read s2_ckpt_hashes.txt: {exc}"); sys.exit(1)
try:
    s2c = dict(line.split()[::-1] for line in open(f"{L}/s2c_ckpt_hashes.txt"))
except Exception as exc:
    print(f"gate: cannot read s2c_ckpt_hashes.txt: {exc}"); sys.exit(1)
if len(set(s2c.values())) != 1 or len(s2c) < 2:
    print(f"gate: S2c fixed-order sha parity not confirmed: {s2c}"); sys.exit(1)

def load(tag):
    return {int(json.loads(line)["step"]): json.loads(line)
            for line in open(f"{L}/s2_{tag}_train_log.jsonl")}

a, b = load("a1"), load("b1")
worst = 0.0
for step in a:
    if a[step]["loss"] != b[step]["loss"] or a[step]["valid_tokens"] != b[step]["valid_tokens"]:
        print(f"gate: S2 loss/valid_tokens differ at step {step}"); sys.exit(1)
    worst = max(worst, abs(float(a[step]["grad_norm"]) - float(b[step]["grad_norm"])))
if worst > 1e-3:
    print(f"gate: S2 grad_norm drift {worst:.2e} exceeds 1e-3"); sys.exit(1)
print(f"gate: OK (amended; S2c fixed-order sha match; S2 loss identical, grad_norm drift <= {worst:.2e})")
PY
}
gate || { plog "S3 SKIPPED: amended gate failed"; exit 0; }

run_leg() { # $1 dir $2 cache_gb $3 label
  local dir="$1" cache="$2" label="$3"
  echo "$(ts) $label start (cache_gb=$cache, GPU $DEV, free $(free_gb)G)" >> "$LOG/s3_progress.log"
  /usr/bin/time -v -o "$LOG/$label.time" \
  $PY -m vlm_dflash.cli train --mode online --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir "$dir" \
    --set train.overfit_sanity.enabled=false \
    --set train.log_every_steps=10 \
    --set train.save_every_steps=100000 \
    --set train.online_batch_size=8 \
    --set train.online_feature_cache_gb="$cache" \
    > "$LOG/$label.log" 2>&1
  local ec=$?
  echo "$(ts) $label exit=$ec (log $LOG/$label.log)" >> "$LOG/s3_progress.log"
  return $ec
}

A=$RUNS/a_off
B=$RUNS/b_on
rm -rf "$A" "$B"

if [ "$(free_gb)" -lt 5 ]; then plog "S3 SKIPPED: <5G free before leg A"; exit 0; fi
run_leg "$A" 0 s3_a_off || true
if [ ! -f "$A/regen/target_data.jsonl" ] || [ ! -f "$A/train/train_summary.json" ]; then
  plog "leg A incomplete (regen or summary missing); skipping leg B"
else
  plog "leg A regen rows: $(wc -l < "$A/regen/target_data.jsonl")"
  mkdir -p "$B"
  cp -a "$A/data" "$A/regen" "$A/contract.json" "$B/" 2>>"$LOG/s3_progress.log"
  [ -d "$A/online" ] && cp -a "$A/online" "$B/" 2>/dev/null
  rm -rf "$B/train"
  if [ "$(free_gb)" -lt 5 ]; then plog "S3: <5G free before leg B; skipping"; else
    run_leg "$B" 16 s3_b_on || true
    if grep -m1 "data stage present" "$LOG/s3_b_on.log" >> "$LOG/s3_progress.log" 2>/dev/null; then :; else
      plog "WARN: leg B did not log a data-stage skip (check digest reuse)"
    fi
  fi
fi

for leg in a_off b_on; do
  d="$RUNS/$leg"
  [ -f "$d/train/train_summary.json" ] && cp "$d/train/train_summary.json" "$LOG/s3_${leg}_summary.json"
  [ -f "$d/train/train_log.jsonl" ] && cp "$d/train/train_log.jsonl" "$LOG/s3_${leg}_train_log.jsonl"
done
: > "$LOG/s3_ckpt_hashes.txt"
for leg in a_off b_on; do
  for ck in ckpt_init.pt ckpt_final.pt; do
    f="$RUNS/$leg/train/$ck"
    [ -f "$f" ] && sha256sum "$f" >> "$LOG/s3_ckpt_hashes.txt"
  done
done
rm -f "$RUNS/a_off/train/ckpt_last.pt" "$RUNS/b_on/train/ckpt_last.pt"
for ck in ckpt_init.pt ckpt_final.pt; do
  ha=$(grep "a_off/train/$ck" "$LOG/s3_ckpt_hashes.txt" | awk '{print $1}')
  hb=$(grep "b_on/train/$ck" "$LOG/s3_ckpt_hashes.txt" | awk '{print $1}')
  if [ -n "$ha" ] && [ "$ha" = "$hb" ]; then
    rm -f "$RUNS/a_off/train/$ck" "$RUNS/b_on/train/$ck"
    plog "pruned $ck (sha256 matched)"
  else
    plog "kept $ck (hashes absent or mismatched)"
  fi
done
plog "S3 DONE"
