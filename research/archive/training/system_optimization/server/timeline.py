#!/usr/bin/env python3
"""Stage timeline from a vlm-dflash run log ([HH:MM:SS] or full-date prefixed).

Usage: python3 timeline.py LOG [LOG ...]

Prints first/last timestamp, total wall, and for each timestamped line either
its stage tag (prepare_data / regen / train / step / drafter / export ...) when
one is present, or its gap when the gap since the previous line is >= 3 s.
Dates roll over correctly across midnight.

NOTE: for S3's single-process run, prepare+regen+train stages run in one
process, so the log is the authoritative per-stage timeline.
"""
import re
import sys
from datetime import datetime, timedelta

TIME_RE = re.compile(r"^\[(?:(\d{4}-\d{2}-\d{2}) )?(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")
TAG_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:\-]")


def parse_log(path):
    rows = []
    day = None
    prev_dt = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TIME_RE.match(line.rstrip("\n"))
            if not m:
                continue
            date_s, hh, mm, ss, rest = m.group(1), *m.groups()[1:]
            if date_s:
                day = date_s
            if day is None:
                day = "1970-01-01"
            dt = datetime.strptime(f"{day} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
            if prev_dt and dt + timedelta(hours=12) < prev_dt:
                day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                dt = datetime.strptime(f"{day} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
            rows.append((dt, rest))
            prev_dt = dt
    return rows


def main():
    for path in sys.argv[1:]:
        rows = parse_log(path)
        if not rows:
            print(f"== {path}: no timestamped lines")
            continue
        t0, t1 = rows[0][0], rows[-1][0]
        print(f"== {path}")
        print(f"   first {t0.time()}  last {t1.time()}  span {t1 - t0}  "
              f"lines {len(rows)}")
        prev = None
        for dt, rest in rows:
            tag = TAG_RE.match(rest)
            gap = (dt - prev).total_seconds() if prev else 0.0
            if tag:
                print(f"   {dt.time()}  +{gap:7.1f}s  {rest[:110]}")
            elif gap >= 3.0:
                print(f"   {dt.time()}  +{gap:7.1f}s  (gap) {rest[:100]}")
            prev = dt
    return 0


if __name__ == "__main__":
    sys.exit(main())
