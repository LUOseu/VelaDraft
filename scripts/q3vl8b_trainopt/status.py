"""Write ``<campaign>/STATUS.md``: core-run progress, lane jobs, scheduler state.

    python3 status.py <campaign_dir> [--watch SECONDS]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TOTAL = 75000


def tail_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 65536))
        lines = [line for line in fh.read().decode(errors="replace").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def render(camp: Path) -> str:
    work = camp / "work"
    out = [f"# STATUS: {camp.name}", "", f"Updated {time.strftime('%F %T %Z')}.", "",
           "## Core runs (fixed_config, 200K x 3 epochs, 75,000 updates)", "",
           "| run | update | epoch | s/update (last segment) | ETA h | last dev objective (step) | state |",
           "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for run in sorted(work.glob("fixed_config_*")) + sorted(work.glob("best_deployment_*")):
        row = tail_json(run / "train" / "train_log.jsonl") or {}
        segment = tail_json(run / "train" / "timing_segments.jsonl") or {}
        step = int(row.get("step", 0))
        per_update = segment.get("update_time_ms", 0) / 1000 if segment else None
        eta = (TOTAL - step) * per_update / 3600 if per_update else None
        dev = None
        validation = run / "train" / "validation.jsonl"
        if validation.exists():
            devs = [json.loads(line) for line in validation.read_text().splitlines()
                    if line.strip() and json.loads(line)["split"] == "dev"]
            if devs:
                dev = f"{devs[-1]['objective_mean']:.4f} ({devs[-1]['step']})"
        log = (run / "stdout.log").read_text(errors="replace") if (run / "stdout.log").exists() else ""
        attempt = log[log.rfind("] START ") :]  # only the latest attempt counts
        state = "done" if "EXIT_CODE=0" in attempt else ("failed" if "EXIT_CODE=" in attempt else "running")
        out.append(f"| {run.name} | {step:,} | {step // 25000 + (1 if step % 25000 else 0)} | "
                   f"{per_update or 0:.2f} | {eta or 0:.1f} | {dev or '-'} | {state} |")
    out += ["", "## Scheduler", ""]
    for state_path in sorted((camp / "raw").glob("scheduler_*/jobs_state.json")):
        state = json.loads(state_path.read_text())
        counts: dict[str, int] = {}
        for entry in state.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        running = [f"{name} (GPU {entry.get('gpus')})" for name, entry in state.items() if entry["status"] == "running"]
        failed = [name for name, entry in state.items() if entry["status"] in ("failed", "blocked")]
        out.append(f"- {state_path.parent.name}: {counts}")
        out.append(f"  - running: {', '.join(running) or '-'}")
        if failed:
            out.append(f"  - failed/blocked: {', '.join(failed)}")
    return "\n".join(out) + "\n"


def main() -> None:
    camp = Path(sys.argv[1])
    watch = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--watch" else 0
    while True:
        (camp / "STATUS.md").write_text(render(camp))
        if not watch or (camp / "UPLOAD_FINAL").exists():
            break
        time.sleep(watch)


if __name__ == "__main__":
    main()
