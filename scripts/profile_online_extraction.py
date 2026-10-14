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

  # per-row, per-length-bucket attribution (the length-distribution report;
  # O(rows) single-sample passes, pin a --limit for big stores):
  python scripts/profile_online_extraction.py \
    --config configs/smoke_multiturn.yaml --run-dir runs/wp4/ab4 \
    --length-buckets 512,1024 --limit 64
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
    parser.add_argument(
        "--length-buckets",
        default=None,
        help="comma-separated token-length upper edges; adds a per-row, per-bucket timing section",
    )
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_length_edges(raw: str) -> list[int]:
    """Parse "512,1024,2048" into sorted, unique, positive bucket edges."""
    edges: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"--length-buckets edges must be positive, got {value}")
        edges.append(value)
    edges = sorted(set(edges))
    if not edges:
        raise ValueError("--length-buckets needs at least one positive edge")
    return edges


def assign_length_bucket(length: int, edges: list[int]) -> int:
    """Bucket index for a row length; edges are upper bounds.

    ``0`` is ``[0, e0)``, ``i`` is ``[e(i-1), e(i))`` and the last bucket
    ``len(edges)`` is ``[e(last), inf)``. A length exactly on an edge belongs
    to the bucket above it (a 512-token row is in ``[512, 1024)``).
    """
    for index, edge in enumerate(edges):
        if length < edge:
            return index
    return len(edges)


def summarise_length_buckets(per_row: list[dict], edges: list[int]) -> list[dict]:
    """Per-bucket medians of the per-row render/forward timings.

    Empty buckets are skipped; ``length_max`` is ``None`` for the unbounded
    top bucket. ``forward_ms_per_token`` divides the bucket's median forward
    by its median token length, so vision-heavy short rows can be told apart
    from text-heavy long rows.
    """
    grouped: dict[int, list[dict]] = {}
    for record in per_row:
        grouped.setdefault(assign_length_bucket(int(record["length"]), edges), []).append(record)
    summary = []
    for bucket in sorted(grouped):
        rows = grouped[bucket]
        render_ms = statistics.median([float(r["render_ms"]) for r in rows])
        forward_ms = statistics.median([float(r["forward_ms"]) for r in rows])
        median_length = statistics.median([int(r["length"]) for r in rows])
        summary.append(
            {
                "bucket": bucket,
                "length_min": 0 if bucket == 0 else edges[bucket - 1],
                "length_max": edges[bucket] if bucket < len(edges) else None,
                "rows": len(rows),
                "median_length": median_length,
                "median_image_tokens": statistics.median([int(r["image_tokens"]) for r in rows]),
                "render_ms_per_row": round(render_ms, 2),
                "forward_ms_per_row": round(forward_ms, 2),
                "forward_ms_per_token": round(1000.0 * forward_ms / max(1.0, float(median_length)), 3),
            }
        )
    return summary


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

        length_buckets = None
        if args.length_buckets:
            edges = parse_length_edges(args.length_buckets)
            per_row = []
            for index in range(rows):
                prepared = None
                render_walls = []
                for _ in range(max(1, int(args.render_repeats))):
                    started = time.perf_counter()
                    prepared = source.prepare(index)
                    _sync(device)
                    render_walls.append(time.perf_counter() - started)
                forward_walls = []
                for _ in range(max(1, int(args.forward_repeats))):
                    started = time.perf_counter()
                    source.extract_prepared([prepared])
                    _sync(device)
                    forward_walls.append(time.perf_counter() - started)
                per_row.append(
                    {
                        "index": index,
                        "length": int(prepared["full_ids"].numel()),
                        "image_tokens": int((prepared["full_ids"] == source.image_token_id).sum()),
                        "render_ms": 1000.0 * statistics.median(render_walls),
                        "forward_ms": 1000.0 * statistics.median(forward_walls),
                    }
                )
            length_buckets = {
                "edges": edges,
                "rows": per_row,
                "buckets": summarise_length_buckets(per_row, edges),
                "note": "per-row single-sample attribution (median of the repeat passes); "
                "the batch-group numbers above remain the trainer-facing view",
            }

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
        if length_buckets is not None:
            report["length_buckets"] = length_buckets
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
