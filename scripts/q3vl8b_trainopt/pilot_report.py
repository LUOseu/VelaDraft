"""Layout pilot summary (contract section 3.1): one synchronized window per pilot run.

    python3 pilot_report.py <campaign_dir>

Reads ``work/pilot_*/train/timing_windows.jsonl`` (20 warmup + 50 measured updates) and
writes ``raw/pilot/layout_pilot.json`` plus a printed table: update time, samples/s,
samples per GPU-second and the per-device allocated/reserved peaks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    camp = Path(sys.argv[1])
    rows = []
    for run in sorted((camp / "work").glob("pilot_*")):
        path = run / "train" / "timing_windows.jsonl"
        if not path.exists():
            rows.append({"run": run.name, "status": "no_window"})
            continue
        window = json.loads(path.read_text().splitlines()[-1])
        camp_cfg = json.loads((run / "config.json").read_text())["campaign"]
        method, arm, layout = camp_cfg["method"], camp_cfg["arm"], camp_cfg["layout"]
        stage = run.name[len(f"pilot_{method}_") : -len(f"_{arm}_{layout}")]
        gpus = 1 if layout == "single" else 2
        peaks = window["peak"]
        rows.append({
            "run": run.name, "method": method, "stage": stage, "arm": arm, "layout": layout, "gpu_count": gpus,
            "updates": window["updates"], "wall_s": window["wall_s"], "update_time_ms": window["update_time_ms"],
            "samples_per_s": window["samples_per_s"],
            "samples_per_gpu_s": window["samples_per_s"] / gpus if window["samples_per_s"] else None,
            "max_device_peak_allocated_gib": max(p["allocated_gib"] for p in peaks.values()),
            "max_device_peak_reserved_gib": max(p["reserved_gib"] for p in peaks.values()),
            "sum_device_peaks_allocated_gib": sum(p["allocated_gib"] for p in peaks.values()),
            "peaks": peaks,
            "target_forward_s": window["target"].get("target_forward_s"),
        })
    out = camp / "raw" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    (out / "layout_pilot.json").write_text(json.dumps(rows, indent=1))
    print(f"{'run':44s} {'ms/upd':>8s} {'smp/s':>7s} {'smp/GPU-s':>9s} {'alloc':>7s} {'resv':>7s}")
    for row in rows:
        if "update_time_ms" not in row:
            print(f"{row['run']:44s} {row['status']}")
            continue
        print(f"{row['run']:44s} {row['update_time_ms']:8.0f} {row['samples_per_s']:7.3f} "
              f"{row['samples_per_gpu_s']:9.3f} {row['max_device_peak_allocated_gib']:7.1f} "
              f"{row['max_device_peak_reserved_gib']:7.1f}")


if __name__ == "__main__":
    main()
