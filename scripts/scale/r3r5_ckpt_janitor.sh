#!/bin/bash
# /home is at 100% with ~2.6 GB free on this host while /dev/shm has tens of GB.
# This janitor moves *completed* eval-format checkpoints (milestones + final) of
# the R3/R5 runs from <run>/train onto /dev/shm and leaves a symlink with the
# same name behind, so every path-based reader (driver eval, pruning, export,
# speed harness) is unaffected. Files younger than MIN_AGE are left alone
# (still being written). Orphaned copies whose symlink was pruned are removed.
# Log: logs/40_ckpt_janitor.log
set -u
BASE=/home/lyh/vlm-scale/runs/r3r5_n20000
DEST=/dev/shm/r3r5_cp
LOG=/home/lyh/vlm-scale/runs/r3r5_20k/logs/40_ckpt_janitor.log
MIN_AGE=${MIN_AGE:-150}
mkdir -p "$DEST"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "janitor up (MIN_AGE=${MIN_AGE}s, dest=$DEST)"
while true; do
  for tag in r3 r5; do
    mkdir -p "$DEST/$tag"
    for f in "$BASE/$tag"/train/ckpt_step*.pt "$BASE/$tag"/train/ckpt_final.pt; do
      [ -f "$f" ] || continue
      [ -L "$f" ] && continue
      age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      [ "$age" -lt "$MIN_AGE" ] && continue
      size=$(stat -c %s "$f")
      if mv "$f" "$DEST/$tag/$(basename "$f")"; then
        ln -s "$DEST/$tag/$(basename "$f")" "$f"
        say "moved $tag/$(basename "$f") ($size bytes, age ${age}s)"
      fi
    done
    for g in "$DEST/$tag"/*.pt; do
      [ -e "$g" ] || continue
      name=$(basename "$g")
      if [ ! -e "$BASE/$tag/train/$name" ]; then
        rm -f "$g" && say "collected orphan $tag/$name"
      fi
    done
  done
  # R3's trainer stdout was unlinked while the run kept the inode open (the
  # /dev/shm migration of 01:42); keep a visible copy of the live file.
  tr3=$(pgrep -f "train_dflash.py --config configs/scale/r3_dfly_n20000" | head -1)
  if [ -n "${tr3:-}" ] && [ -e "/proc/$tr3/fd/1" ]; then
    cp "/proc/$tr3/fd/1" "$DEST/r3_driver/train_stdout.live.log" 2>/dev/null
  fi
  sleep 60
done
