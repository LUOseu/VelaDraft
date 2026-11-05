#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_qwen3vl8b_cont_mmspec.sh [options]

Options:
  --preflight             Check paths and all requested method exports only.
  --gpus LIST             Physical GPU ids, e.g. 0,1,2,3 (default: 0).
  --methods LIST          Comma list or all (default: all).
  --suite smoke|formal|all (default: all).
  --resume                Skip an already valid report; archive invalid old runs.
  --publish               Upload method directories, summary and manifest to HF.
  --campaign-id ID        Reuse a campaign id instead of generating one.
  --help                  Show this help.
USAGE
}

PRE_FLIGHT=0
GPUS="0"
METHODS="all"
SUITE="all"
RESUME=0
PUBLISH=0
CAMPAIGN_ID=""

while (($#)); do
  case "$1" in
    --preflight) PRE_FLIGHT=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --publish) PUBLISH=1; shift ;;
    --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SUITE" in smoke|formal|all) ;; *) echo "invalid --suite: $SUITE" >&2; exit 2 ;; esac

# Machine paths come from a site env file kept outside the repository (Q3VL8B_SITE), defining
# Q3VL8B_ROOT (task root holding exports/), TARGET, SOURCE, IMAGES, PYTHON, RESULT_ROOT and
# optionally SHM (node-local scratch), HF_HOME and Q3VL8B_PRELAUNCH (a hook given one GPU index
# that may release site placeholder processes on that card; the protocol never stops a foreign
# process and refuses a busy card).
if [[ -n "${Q3VL8B_SITE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$Q3VL8B_SITE"
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${Q3VL8B_ROOT:?set Q3VL8B_ROOT (or Q3VL8B_SITE)}"
: "${TARGET:?set TARGET}" "${SOURCE:?set SOURCE}" "${IMAGES:?set IMAGES}" "${PYTHON:?set PYTHON}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
SHM="${SHM:-/dev/shm/${USER:-root}_q3vl8b_cont}"
HF_REPO_ID="${HF_REPO_ID:-khazic/veladraft-qwen3vl8b-200k-cont-dspark}"
HF_PREFIX="${HF_PREFIX:-eval/qwen3vl8b_mmspec600}"

if [[ -z "$CAMPAIGN_ID" ]] && ((RESUME)); then
  # Resume the most recent campaign (ids start with a UTC timestamp) unless one is named.
  CAMPAIGN_ID="$(find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*T[0-9]*Z_*' \
    -printf '%f\n' 2>/dev/null | grep -v '\.' | sort | tail -n 1 || true)"
  [[ -n "$CAMPAIGN_ID" ]] || { echo "ERROR: --resume found no campaign under $RESULT_ROOT" >&2; exit 1; }
fi
if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)"
fi
CAMPAIGN_DIR="$RESULT_ROOT/$CAMPAIGN_ID"
LOG_DIR="$CAMPAIGN_DIR/logs"
mkdir -p "$LOG_DIR"

# Every method uses the port in research/qwen3vl8b_cont_mmspec/native: one CPU export and
# one runtime manifest per method, all at K=7 over the trained block of 8 (see README.md).
ALL_METHODS=(dflash dspark domino jetspec dflash2 dflare dfly)
draft_of() { echo "$Q3VL8B_ROOT/exports/$1"; }
manifest_of() { echo "$Q3VL8B_ROOT/exports/$1_runtime.json"; }

# Frozen protocol (AGENTS.md section 2). K must equal native/contract.py SPECULATIVE_TOKENS.
K=7
SESSIONS=600
TURNS=723
REPEATS=3
MAX_NEW_TOKENS=1024
GPU_MEM=0.78
TIMED_PAIRS=$((TURNS * REPEATS))
MIN_PIXELS=65536
MAX_PIXELS=1048576
MAX_MODEL_LEN=8192
TEMPLATE_KWARGS='{"enable_thinking":false}'
# Verbatim from the handoff launcher (the MMSpec reference reads "the human's questions").
SYSTEM_PROMPT='A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human questions.'

# Native-eval runtime: offline hub, node-local compile caches, count-mode GPU ownership (a container
# PID namespace hides host PIDs from nvidia-smi), one client context + one engine core per card.
mkdir -p "$SHM/gpu_locks" "$SHM/triton" "$SHM/xdg"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$SHM/triton" XDG_CACHE_HOME="$SHM/xdg"
export QWEN8B_DIGEST_CACHE="$SHM/digests.sqlite"
export VELADRAFT_GPU_LOCK_DIR="$SHM/gpu_locks"
export VELADRAFT_GPU_OWNERSHIP="${VELADRAFT_GPU_OWNERSHIP:-count}" VELADRAFT_GPU_MAX_OWN_APPS="${VELADRAFT_GPU_MAX_OWN_APPS:-2}"
cd "$REPO"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date -Is)] $*" | tee -a "$LOG_DIR/launcher.log"; }

wait_for_gpu() {
  local gpu="$1" used pids
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for native GPU evaluation"
  while :; do
    if [[ -n "${Q3VL8B_PRELAUNCH:-}" ]]; then "$Q3VL8B_PRELAUNCH" "$gpu" || true; fi
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 && -z "$pids" ]]; then
      return 0
    fi
    log "WAIT_GPU gpu=$gpu used=${used:-unknown} pids=${pids:-none}"
    sleep 30
  done
}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
((${#GPU_LIST[@]} > 0)) || die "at least one GPU is required"
for gpu in "${GPU_LIST[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"; done

if [[ "$METHODS" == all ]]; then
  SELECTED=("${ALL_METHODS[@]}")
else
  IFS=',' read -r -a SELECTED <<< "$METHODS"
fi
for method in "${SELECTED[@]}"; do
  [[ " ${ALL_METHODS[*]} " == *" $method "* ]] || die "unknown method: $method"
done

[[ -f "$TARGET/config.json" ]] || die "target config missing: $TARGET/config.json"
[[ -f "$SOURCE" ]] || die "MMSpec source missing: $SOURCE"
[[ -d "$IMAGES" ]] || die "MMSpec images directory missing: $IMAGES"
[[ -x "$PYTHON" ]] || die "Python runtime missing/executable bit absent: $PYTHON"
if ((PUBLISH)); then command -v hf >/dev/null 2>&1 || die "--publish requires the hf CLI"; fi
"$PYTHON" - <<'PY'
import importlib.util
for name in ("torch", "vllm"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"{name} is not importable in the selected runtime")
PY
for method in "${SELECTED[@]}"; do
  [[ -d "$(draft_of "$method")" ]] || die "$method draft export missing: $(draft_of "$method")"
  [[ -f "$(manifest_of "$method")" ]] || die "$method runtime manifest missing: $(manifest_of "$method")"
  log "PREFLIGHT_OK method=$method draft=$(draft_of "$method") manifest=$(manifest_of "$method") K=$K python=$PYTHON"
done
if ((PRE_FLIGHT)); then
  log "PREFLIGHT_COMPLETE methods=${SELECTED[*]}"
  exit 0
fi

# Hash the shared inputs once into the digest cache before the method workers start, so
# concurrent workers do not each read the target weights and every image in full.
"$PYTHON" - "$QWEN8B_DIGEST_CACHE" "$TARGET" "$SOURCE" "$IMAGES" <<'PY'
import sys
from pathlib import Path
from vlm_dflash.eval.digest_cache import cached_sha256
database, target, source, images = sys.argv[1:]
files = [p for p in sorted(Path(target).iterdir()) if p.is_file()]
files += [Path(source)] + [p for p in sorted(Path(images).iterdir()) if p.is_file()]
for path in files:
    cached_sha256(path, database)
PY

# Written once per campaign; each launch appends its commit to launches.log. Per-file SHA256
# pins (target, images, source, export, adapter sources, checkpoint) are in every stage's
# report.evidence/plan.json.
CODE_COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "$(date -Is) commit=$CODE_COMMIT suite=$SUITE methods=${SELECTED[*]} gpus=$GPUS" >> "$CAMPAIGN_DIR/launches.log"
if [[ ! -f "$CAMPAIGN_DIR/run_manifest.json" ]]; then
cat > "$CAMPAIGN_DIR/run_manifest.json" <<EOF_MANIFEST
{
  "campaign_id": "${CAMPAIGN_ID}",
  "target": "${TARGET}",
  "source": "${SOURCE}",
  "images": "${IMAGES}",
  "methods": [$(printf '"%s",' "${SELECTED[@]}" | sed 's/,$//')],
  "dataset": "MMSpec full test",
  "sessions": ${SESSIONS},
  "turns": ${TURNS},
  "repeats": ${REPEATS},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "gpu_memory_utilization": ${GPU_MEM},
  "speculative_tokens": ${K},
  "first_launch_code_commit": "${CODE_COMMIT}",
  "gpu_name": "$(nvidia-smi -i "${GPU_LIST[0]}" --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)",
  "exports_and_manifests": "${Q3VL8B_ROOT}/exports (export_receipt.json per method)",
  "file_sha256": "<method>/<stage>/report.evidence/plan.json",
  "protocol_notes": "research/qwen3vl8b_cont_mmspec/README.md",
  "hf_repo_id": "${HF_REPO_ID}",
  "hf_prefix": "${HF_PREFIX}/${CAMPAIGN_ID}"
}
EOF_MANIFEST
fi

# valid_report <report.json> smoke|formal: complete, resource-verified, and for formal the full
# paired-turn count with the smoke flag off.
valid_report() {
  "$PYTHON" - "$1" "$2" "$TIMED_PAIRS" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); stage=sys.argv[2]
ok=d.get("status")=="complete" and d.get("resources_verified") is True
if stage=="formal": ok=ok and d.get("timed_pairs")==int(sys.argv[3]) and not d.get("smoke",False)
else: ok=ok and d.get("smoke",False)
raise SystemExit(0 if ok else 1)
PY
}

archive_invalid() {
  local dir="$1"
  if [[ -e "$dir" ]]; then
    mv "$dir" "${dir}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "$dir"
}

# hf_put <local path> <name under the campaign prefix> <commit message>; no-op without --publish.
hf_put() {
  ((PUBLISH)) || return 0
  hf upload "$HF_REPO_ID" "$1" "$HF_PREFIX/$CAMPAIGN_ID/$2" --repo-type model --commit-message "$3"
}

# A failed upload is logged and retried by a later --resume --publish; it never stops a worker.
upload_method() {
  hf_put "$CAMPAIGN_DIR/$1" "$1" "eval: Qwen3-VL-8B CONT $1 MMSpec ${CAMPAIGN_ID}" ||
    log "UPLOAD_FAILED method=$1"
}

run_stage() {
  local method="$1" stage="$2" gpu="$3" out="$4"; shift 4
  local stage_dir
  stage_dir="$(dirname "$out")"
  # native-eval keeps per-stage compile caches (many small files) inside its evidence
  # directory, so the run itself lives on node-local scratch; the report and evidence
  # without caches are archived next to $out afterwards.
  local scratch="$SHM/runs/$CAMPAIGN_ID/$method/$stage"
  local local_out="$scratch/$(basename "$out")" rc=0
  wait_for_gpu "$gpu"
  mkdir -p "$stage_dir"
  rm -rf "$scratch" && mkdir -p "$scratch"
  local -a cmd=(
    "$PYTHON" -u -m vlm_dflash.cli native-eval
    --target "$TARGET" --draft "$(draft_of "$method")"
    --source "$SOURCE" --images-dir "$IMAGES"
    --runtime-manifest "$(manifest_of "$method")" --method "$method" --gpu "$gpu"
    --gpu-memory-utilization "$GPU_MEM" --speculative-tokens "$K"
    --min-pixels "$MIN_PIXELS" --max-pixels "$MAX_PIXELS" --max-model-len "$MAX_MODEL_LEN"
    --chat-template-kwargs "$TEMPLATE_KWARGS" --system-prompt "$SYSTEM_PROMPT"
    "$@" --out "$local_out"
  )
  printf '%q ' "${cmd[@]}" > "$stage_dir/command.txt"
  printf '\n' >> "$stage_dir/command.txt"
  "${cmd[@]}" > "$stage_dir/stdout.log" 2>&1 || rc=$?
  local evidence="${local_out%.json}.evidence" archived=0
  if [[ -d "$evidence" ]]; then
    tar -C "$scratch" --exclude='*/cache' --exclude='*/inductor' -cf - "$(basename "$evidence")" \
      | tar -C "$stage_dir" -xf - || archived=1
  fi
  if [[ -f "$local_out" ]]; then
    # The archived report must point at the archived evidence, not at the scratch copy.
    "$PYTHON" - "$local_out" "$out" "$scratch" "$stage_dir" <<'PY' || archived=1
import json,sys
src,dst,old,new=sys.argv[1:]
d=json.load(open(src))
for key in ("evidence","resource_evidence"):
    if isinstance(d.get(key),str) and d[key].startswith(old):d[key]=new+d[key][len(old):]
json.dump(d,open(dst,"w"),indent=2)
PY
  fi
  if ((archived)); then
    log "ARCHIVE_FAILED method=$method stage=$stage scratch kept at $scratch"
    return 1
  fi
  rm -rf "$scratch"
  return "$rc"
}

# run_suite <method> <gpu> smoke|formal <native-eval args...>: skip a valid report on resume,
# otherwise run the stage; returns non-zero (after recording FAILED_<STAGE>.txt) on failure.
run_suite() {
  local method="$1" gpu="$2" stage="$3"; shift 3
  local dir="$CAMPAIGN_DIR/$method"
  local stage_dir="$dir/$stage" tag="${stage^^}"
  local report="$stage_dir/report.json"
  if ((RESUME)) && [[ -f "$report" ]] && valid_report "$report" "$stage"; then
    log "SKIP_VALID_${tag} method=$method"
    return 0
  fi
  if [[ -f "$report" ]]; then archive_invalid "$stage_dir"; fi
  log "RUN_${tag} method=$method gpu=$gpu"
  if ! run_stage "$method" "$stage" "$gpu" "$report" "$@"; then
    echo "$stage failed; inspect $stage_dir/report.evidence" > "$dir/FAILED_${tag}.txt"
    log "${tag}_FAILED method=$method"
    return 1
  fi
  if ! valid_report "$report" "$stage"; then
    echo "$stage report invalid or resource evidence contaminated; inspect $stage_dir/report.evidence" > "$dir/FAILED_${tag}.txt"
    log "${tag}_INVALID method=$method"
    return 1
  fi
  rm -f "$dir/FAILED_${tag}.txt"
}

run_method() {
  local method="$1" gpu="$2"
  mkdir -p "$CAMPAIGN_DIR/$method"
  log "START method=$method gpu=$gpu K=$K"
  if [[ "$SUITE" == smoke || "$SUITE" == all ]] &&
      ! run_suite "$method" "$gpu" smoke --num-prompts 2 --max-new-tokens 128 --repeats 1; then
    upload_method "$method"
    return 0
  fi
  if [[ "$SUITE" == formal || "$SUITE" == all ]] &&
      ! run_suite "$method" "$gpu" formal --num-prompts "$SESSIONS" --max-new-tokens "$MAX_NEW_TOKENS" --repeats "$REPEATS"; then
    upload_method "$method"
    return 0
  fi
  upload_method "$method"
  log "COMPLETE method=$method gpu=$gpu"
}

log "CAMPAIGN_START id=$CAMPAIGN_ID gpus=$GPUS methods=${SELECTED[*]} suite=$SUITE"
declare -A WORKER_ARGS
for i in "${!SELECTED[@]}"; do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  WORKER_ARGS[$gpu]="${WORKER_ARGS[$gpu]-} ${SELECTED[$i]}"
done
for gpu in "${GPU_LIST[@]}"; do
  [[ -n "${WORKER_ARGS[$gpu]-}" ]] || continue
  # shellcheck disable=SC2086
  (for method in ${WORKER_ARGS[$gpu]}; do run_method "$method" "$gpu"; done) > "$LOG_DIR/gpu${gpu}.log" 2>&1 &
done
wait

# Only reports that pass the formal validity check enter the table as complete; anything else
# is listed with its failure reason (AGENTS.md section 5).
"$PYTHON" - "$CAMPAIGN_DIR" "$TIMED_PAIRS" "${ALL_METHODS[@]}" > "$CAMPAIGN_DIR/summary.csv" <<'PY'
import csv,json,os,sys
root,pairs,methods=sys.argv[1],int(sys.argv[2]),sys.argv[3:]
fields=["method","status","sessions","turns","repeats","timed_pairs","speedup_sum_ar_over_sd","ci_low","ci_high",
        "ar_seconds","sd_seconds","exact_output_pairs","error"]
def failure(method):
    for stage in ("FORMAL","SMOKE"):
        path=os.path.join(root,method,f"FAILED_{stage}.txt")
        if os.path.isfile(path):return open(path).read().strip()
    return "formal report missing"
rows=[]
for method in methods:
    if not os.path.isdir(os.path.join(root,method)):continue
    row=dict.fromkeys(fields); row["method"]=method
    p=os.path.join(root,method,"formal","report.json")
    d=json.load(open(p)) if os.path.isfile(p) else {}
    valid=(d.get("status")=="complete" and d.get("resources_verified") is True
           and d.get("timed_pairs")==pairs and not d.get("smoke",False))
    if valid:
        row.update({k:d[k] for k in fields if k in d})
        row["turns"]=d["timed_pairs"]//d["repeats"]
        row["ci_low"],row["ci_high"]=d.get("session_cluster_CI95") or (None,None)
    else:
        row["status"]="invalid" if d else "failed_or_missing"
        row["error"]=failure(method)
    rows.append(row)
w=csv.DictWriter(sys.stdout,fieldnames=fields); w.writeheader(); w.writerows(rows)
PY

hf_put "$CAMPAIGN_DIR/summary.csv" summary.csv "eval: summary ${CAMPAIGN_ID}" || log "UPLOAD_FAILED summary.csv"
hf_put "$CAMPAIGN_DIR/run_manifest.json" run_manifest.json "eval: manifest ${CAMPAIGN_ID}" ||
  log "UPLOAD_FAILED run_manifest.json"
log "CAMPAIGN_COMPLETE id=$CAMPAIGN_ID results=$CAMPAIGN_DIR"
