#!/bin/bash
# The host OOM killer SIGKILLs whichever process has the largest RSS, and the
# two 40 GB trainers are the biggest things on a 251 GB host that also carries
# ~104 GB of tmpfs. Log the trend so any future kill can be attributed instead
# of guessed at. Read-only: never kills anything.
LOG=${LOG:-/home/lyh/vlm-scale/runs/r3r5_20k/logs/60_mem_watch.log}
while true; do
  {
    printf '[%s] ' "$(date '+%F %T')"
    free -g | awk 'NR==2{printf "mem used=%sG free=%sG shared=%sG avail=%sG ", $3,$4,$5,$7} NR==3{printf "swap used=%sG\n", $3}'
    ps -eo rss,pid,args --sort=-rss --no-headers 2>/dev/null | head -5 \
      | awk '{printf "    %8.1fG pid=%s %s\n", $1/1048576, $2, substr($0, index($0,$3), 70)}'
  } >> "$LOG"
  sleep 60
done
