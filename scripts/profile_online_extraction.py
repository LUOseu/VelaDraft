#!/usr/bin/env python3
"""Profile the online extraction path: render half vs batched target forward.

WP4 profiling evidence (`docs/plan/WP4_DESIGN.md`): for a run directory with
regenerated rows this times

- the per-sample render half (`OnlineFeatureSource.prepare`: image decode +
  processor + drift validation), repeated `--render-repeats` times so the
  warm page-cache cost (what a render cache could remove after the first
  epoch) is visible;
- the batched target forward half (`extract_prepared`) for each requested
  capture-group size, with the CUDA peak on the target device.

It loads the real frozen target on the run's configured device, does no
training, and writes nothing. The printed JSON feeds the WP4 measurement
report; the numbers decide whether a render/cache optimization is worth
keeping at all.

Usage:
  python scripts/profile_online_extraction.py \
    --config configs/smoke_multiturn.yaml --run-dir runs/wp4/ab4 --groups 1,4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.config import load_config
from vlm_dflash.contract import ensure_contract
from vlm_dflash.target.online import OnlineFeatureSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="run YAML (same file the training used)")
    parser.add_argument("--run-dir", required=True, help="run directory with regen/target_data.jsonl")
    parser.add_argument("--groups", default="1,4", help="capture-group sizes to time (comma-separated)")
    parser.add_argument("--limit", type=int, default=None, help="rows to profile (default: all)")
    parser.add_argument("--render-repeats", type=int, default=2, help="fresh render passes over all rows")
    parser.add_argument("--forward-repeats", type=int, default=3, help="timed passes per capture group")
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    source = OnlineFeatureSource(cfg, contract, run_dir, limit=args.limit)
    try:
        rows = len(source.rows)
        device = source.target_device
        groups = [int(value) for value in str(args.groups).split(",") if value.strip()]

        render_passes: list[float] = []
        prepared = None
        for _ in range(max(1, int(args.render_repeats))):
            started = time.perf_counter()
            pass_prepared = [source.prepare(index) for index in range(rows)]
            _sync(device)
            render_passes.append(time.perf_counter() - started)
            if prepared is None:
                prepared = pass_prepared

        forward = []
        for group in groups:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            walls = []
            for _ in range(max(1, int(args.forward_repeats))):
                started = time.perf_counter()
                for start in range(0, rows, group):
                    source.extract_prepared(prepared[start : start + group])
                _sync(device)
                walls.append(time.perf_counter() - started)
            wall = statistics.median(walls)
            calls = len(range(0, rows, group))
            peak_gb = float(torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None
            forward.append(
                {
                    "group": group,
                    "forwards": calls,
                    "samples": rows,
                    "passes_seconds": [round(value, 4) for value in walls],
                    "median_wall_seconds": round(wall, 4),
                    "forward_ms": round(1000.0 * wall / max(1, calls), 2),
                    "sample_ms": round(1000.0 * wall / rows, 2),
                    "target_device_peak_gb": None if peak_gb is None else round(peak_gb, 3),
                }
            )

        report = {
            "run_dir": str(run_dir),
            "rows": rows,
            "seq_length": source.seq_length,
            "device": str(device),
            "storage_dtype": str(source.dtype).replace("torch.", ""),
            "render": {
                "passes_seconds": [round(value, 4) for value in render_passes],
                "sample_ms": round(1000.0 * render_passes[0] / rows, 2),
                "warm_sample_ms": round(1000.0 * render_passes[-1] / rows, 2),
                "note": "per-sample render (image decode + processor + drift check), "
                "no target work; a render cache removes the warm pass after the first epoch",
            },
            "forward": forward,
            "counters": source.summary()["extractions"],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
