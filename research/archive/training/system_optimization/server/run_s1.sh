#!/bin/bash
# S1 queue: S1a (smoke re-measurement) then S1b (real-row sweep) on one device.
set -u
LANE=/home/lyh/vlm-systems
DEV="${1:?device index}"
echo "$(date +'[%F %T]') S1 queue start on GPU $DEV" >> "$LANE/logs/waiter.log"
# Order: S1a (cheap smoke, first corrected-timing baseline) -> S2 (cache A/B,
# highest value per GPU-minute; builds the real256 master itself) -> S1b
# (profiles that same master; H-A closure) -> S3 (canonical 2K pair, gated on
# S2 and disk).
bash "$LANE/run_s1a_smoke_remeasure.sh" "$DEV"
bash "$LANE/run_s2_ab.sh" "$DEV"
bash "$LANE/run_s1b_sweep.sh" "$DEV"
bash "$LANE/run_s3_ab.sh" "$DEV"
echo "$(date +'[%F %T]') S1 queue done on GPU $DEV" >> "$LANE/logs/waiter.log"
