"""Re-check count-mode GPU ownership evidence with per-GPU bounds.

    python audit_ownership.py <run dir> [<run dir> ...]

Each run dir holds ``result.evidence`` (plan.json, resources.jsonl, the stage
reports). Inside every timed window the first leased GPU may carry two compute
apps (client context and TP rank 0) and every other GPU one (its TP worker); the
evaluator at 69efbba applied the two-app bound to every GPU. Prints one JSON
record per run and exits non-zero when any run has an over-bound timed sample.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def timed_windows(evidence: Path) -> list[tuple[int, int]]:
    windows = []
    for stage in sorted(evidence.glob("[as][rd]_r*.json")):
        rows = [r for r in json.loads(stage.read_text())["rows"] if r.get("phase") == "timed"]
        if rows:
            windows.append((min(r["began_wall_ns"] for r in rows), max(r["ended_wall_ns"] for r in rows)))
    return windows


def audit(run: Path, max_own_apps: int = 2) -> dict:
    evidence = run / "result.evidence"
    uuids = json.loads((evidence / "plan.json").read_text())["gpu_uuids"]
    bound = {u: max_own_apps if u == uuids[0] else 1 for u in uuids}
    windows = timed_windows(evidence)
    samples, flagged = 0, []
    for line in (evidence / "resources.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "per_gpu_pids" not in row or row["phase"] in ("start", "boundary", "boundary_wait"):
            continue
        if not any(row["began_wall_ns"] <= end and row["ended_wall_ns"] >= begin for begin, end in windows):
            continue
        samples += 1
        over = {u: len(p) for u, p in row["per_gpu_pids"].items() if len(p) > bound[u]}
        if over:
            flagged.append({"began_wall_ns": row["began_wall_ns"], "apps": over})
    return {"run": run.name, "bounds": bound, "timed_windows": len(windows), "timed_samples": samples,
            "over_bound_timed_samples": len(flagged), "first_over_bound": flagged[:5]}


def main() -> None:
    records = [audit(Path(arg)) for arg in sys.argv[1:]]
    for record in records:
        print(json.dumps(record))
    sys.exit(1 if any(r["over_bound_timed_samples"] for r in records) else 0)


if __name__ == "__main__":
    main()
