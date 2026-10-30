#!/bin/bash
# Qwen3-VL-8B training-systems campaign on one node.
#
#   CAMPAIGN=<campaign dir> bash launch_campaign.sh configs   # freeze configs + provenance
#   CAMPAIGN=<campaign dir> bash launch_campaign.sh <phase>   # run a phase (prep | core) in the background
#
# Machine paths come from TRAINOPT_ENV, an environment file kept on the node (not in the
# repository) that exports VELA_OLD_TASK, VELA_CORPUS, VELA_DEVTEST, VELA_TARGET, VELA_LOCAL,
# HF_CACHE_HOME, TP and optionally FLOOR_PY and FLOOR_SCRIPT. The campaign directory follows the contract
# layout (configs/, provenance/, raw/, tables/; work/ holds the run directories).
set -euo pipefail
: "${TRAINOPT_ENV:?set TRAINOPT_ENV to the node environment file}" "${CAMPAIGN:?set CAMPAIGN to the campaign directory}"
source "$TRAINOPT_ENV"
export TRAINOPT_ENV
HERE=$(cd "$(dirname "$0")" && pwd)
export VELA_REPO=$(cd "$HERE/../.." && pwd)
OLD=$VELA_OLD_TASK
DEVTEST=$VELA_DEVTEST
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
phase=$1
mkdir -p "$CAMPAIGN"/{configs,provenance,manifests,raw,tables,work,scripts}

if [ "$phase" = configs ]; then
  python3 "$HERE/make_configs.py" --old-task "$OLD" --devtest "$DEVTEST" --out "$CAMPAIGN/configs" --work "$CAMPAIGN/work" \
    --local "$VELA_LOCAL"
  cp "$HERE"/*.sh "$HERE"/*.py "$CAMPAIGN/scripts/"
  cp "$TRAINOPT_ENV" "$CAMPAIGN/scripts/node_env.sh"
  {
    echo "commit $(git -C "$VELA_REPO" rev-parse HEAD)"
    echo "branch $(git -C "$VELA_REPO" rev-parse --abbrev-ref HEAD)"
    echo "base_main 7d7eab4ff39eaadfa3033c6d5093fc3e7db03926"
  } > "$CAMPAIGN/provenance/source.txt"
  git -C "$VELA_REPO" diff 7d7eab4ff39eaadfa3033c6d5093fc3e7db03926 HEAD > "$CAMPAIGN/provenance/source_patch_vs_main.diff"
  git -C "$VELA_REPO" status --porcelain > "$CAMPAIGN/provenance/worktree_status.txt"
  "$TP" - > "$CAMPAIGN/provenance/environment.json" <<'EOF'
import json, platform, sys
import torch, transformers
print(json.dumps({
    "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, indent=1))
EOF
  nvidia-smi > "$CAMPAIGN/provenance/nvidia_smi.txt" 2>&1 || true
  echo CONFIGS_DONE
  exit 0
fi

python3 "$HERE/jobs.py" "$phase" "$CAMPAIGN" > "$CAMPAIGN/raw/jobs_$phase.json"
if ! ps -eo args | grep -qE "^bash \S*gpu_sampler\.sh $CAMPAIGN\$"; then
  setsid -f nohup bash "$HERE/gpu_sampler.sh" "$CAMPAIGN" > /dev/null 2>&1 < /dev/null
fi
setsid -f nohup python3 -u "$HERE/scheduler.py" --jobs "$CAMPAIGN/raw/jobs_$phase.json" \
  --state-dir "$CAMPAIGN/raw/scheduler_$phase" --gpus "$GPUS" >> "$CAMPAIGN/raw/scheduler_$phase.log" 2>&1 < /dev/null
echo "phase $phase scheduled (log $CAMPAIGN/raw/scheduler_$phase.log)"
