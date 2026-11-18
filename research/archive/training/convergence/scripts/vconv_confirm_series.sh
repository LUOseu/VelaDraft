#!/bin/bash
# Launch the preregistered confirmatory + pilot leg matrix at L*.
# Per-GPU serial queues; idempotent per leg via vconv_chain.sh (skips done legs).
# usage: vconv_confirm_series.sh <L> <gpu1,gpu2[,...]> [pilot]
set -u
L=$1; GPUS=$2; PILOT=${3:-}
STATUS=/home/lyh/veladraft-conv/logs/vconv_status.log
SCRIPTS=/home/lyh/veladraft-conv/scripts
LOGS=/home/lyh/veladraft-conv/logs
IFS=, read -ra G <<< "$GPUS"
for gpu in "${G[@]}"; do
  if [ -s "$LOGS/queue_gpu$gpu.txt" ]; then echo "refusing: $LOGS/queue_gpu$gpu.txt exists (rerun after cleaning it)"; exit 1; fi
done

specs=("conv_canon_s42|42|off|-|-|-" "conv_grid2_s42|42|grid|2|true|true"
       "conv_canon_s43|43|off|-|-|-" "conv_grid2_s43|43|grid|2|true|true"
       "conv_canon_s44|44|off|-|-|-" "conv_grid2_s44|44|grid|2|true|true"
       "conv_canon_s45|45|off|-|-|-" "conv_grid2_s45|45|grid|2|true|true")
if [ "$PILOT" = "pilot" ]; then
  specs+=("conv_token2_s42|42|token|2|true|true" "conv_split2_s42|42|canonical|2|true|true"
          "conv_token2_s43|43|token|2|true|true" "conv_split2_s43|43|canonical|2|true|true")
fi

n=0
for spec in "${specs[@]}"; do
  gpu=${G[$((n % ${#G[@]}))]}
  echo "$spec" >> "$LOGS/queue_gpu$gpu.txt"
  n=$((n+1))
done

for gpu in "${G[@]}"; do
  queue="$LOGS/queue_gpu$gpu.txt"
  [ -s "$queue" ] || { echo "gpu$gpu queue empty" >> "$STATUS"; continue; }
  worker="$SCRIPTS/.vconv_worker_gpu$gpu.sh"
  cat > "$worker" <<EOF
#!/bin/bash
set -u
STATUS=$STATUS
SCRIPTS=$SCRIPTS
L=$L
gpu=$gpu
while IFS='|' read -r name seed mode sub fp32 bypass; do
  [ -z "\$name" ] && continue
  echo "gpu\$gpu start \$name L=\$L \$(date '+%F %H:%M:%S')" >> "\$STATUS"
  VCONV_GPU=\$gpu VCONV_STEPS=\$L bash "\$SCRIPTS/vconv_chain.sh" "\$name" "\$seed" "\$mode" "\$sub" "\$fp32" "\$bypass" >>"\$STATUS" 2>&1 \\
    || echo "gpu\$gpu LEG-FAIL \$name \$(date '+%F %H:%M:%S')" >> "\$STATUS"
done < "$queue"
echo "gpu\$gpu queue done \$(date '+%F %H:%M:%S')" >> "\$STATUS"
EOF
  chmod +x "$worker"
  nohup bash "$worker" >>"$STATUS" 2>&1 &
  echo "launched gpu$gpu worker pid=$! $(date '+%F %H:%M:%S') queue=$(wc -l < "$queue") legs" >> "$STATUS"
done
echo "series launched L=$L gpus=$GPUS pilot=$PILOT $(date '+%F %H:%M:%S')" >> "$STATUS"
