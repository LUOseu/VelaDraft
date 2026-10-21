#!/usr/bin/env bash
source /root/veladraft-env.sh
cd /root/veladraft-npu
for data in natural shape; do
  for shape in '1024 64' '4096 256' '4096 512'; do
    set -- $shape
    context=$1
    anchors=$2
    for mode in dense dense_checkpoint chunked_checkpoint; do
      extra=()
      if [ "$data" = natural ]; then extra+=(--features-dir /root/veladraft-audit-20261021/corpus/features); fi
      if [ "$context" = 1024 ]; then extra+=(--correctness-output "/tmp/audit-B-fp32-$data-$mode.pt"); fi
      python scripts/npu/audit_attention.py --target /root/veladraft-models/Qwen2.5-VL-3B-Instruct --head /root/veladraft-audit-20261021/corpus/features/target_head.pt --contract /root/veladraft-audit-20261021/corpus/contract.json --context "$context" --anchors "$anchors" --mode "$mode" --output "runs/npu/audit-20261021/memory/$data-$context-$anchors-$mode.json" "${extra[@]}"
    done
  done
done
