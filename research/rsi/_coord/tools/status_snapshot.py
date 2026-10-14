#!/usr/bin/env python3
"""Read-only coordination snapshot for the RSI lanes (C0 tool).

Scans the known local worktrees for research/rsi/<role> task files and prints
per-lane status: branch/SHA, dirty count, task-file mtimes, EXPERIMENTS entry
count and last decision. stdout only; never writes.

Usage:
    python3 status_snapshot.py [--root /Users/Admin/grounding-parallel]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

LANES = {
    "c0": "veladraft-c0",
    "d1": "vlm-dflash-d1",
    "d2": "vlm-dflash-d2",
    "d3": "vlm-dflash-d3",
    "d4": "vlm-dflash-d4",
    "d5": "vlm-dflash-d5",
}
FILES = ("STATE.md", "REPORT.md", "handoff.md", "EXPERIMENTS.jsonl")


def git(cwd: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else f"ERR:{out.stderr.strip()[:60]}"
    except Exception as exc:  # pragma: no cover - tooling convenience
        return f"ERR:{exc}"


def mtime(path: Path) -> str:
    try:
        return time.strftime("%H:%M", time.localtime(path.stat().st_mtime))
    except FileNotFoundError:
        return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/Admin/grounding-parallel")
    args = ap.parse_args()
    root = Path(args.root)
    print(f"# RSI status snapshot {time.strftime('%F %T')}  root={root}")
    for role, wt in LANES.items():
        d = root / wt
        if not d.exists():
            print(f"{role}: MISSING worktree {wt}")
            continue
        sha = git(d, "rev-parse", "--short", "HEAD")
        branch = git(d, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = git(d, "status", "--porcelain")
        dirty_n = len(dirty.splitlines()) if dirty and not dirty.startswith("ERR") else 0
        lane = d / "research" / "rsi" / ("_coord" if role == "c0" else role)
        parts = [f"{role}: {branch}@{sha} dirty={dirty_n}"]
        for name in FILES:
            parts.append(f"{name}@{mtime(lane / name)}")
        exp = lane / "EXPERIMENTS.jsonl"
        if exp.exists():
            ok, last = 0, None
            for line in exp.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    last = json.loads(line)
                    ok += 1
                except Exception:
                    pass
            decision = (last or {}).get("decision") if isinstance(last, dict) else None
            parts.append(f"exp_entries={ok} last_decision={decision}")
        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
