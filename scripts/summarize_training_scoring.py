#!/usr/bin/env python3
"""Recompute paired probe summaries from raw JSON; keep all negative cells."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text())
    if not data.get("complete"):
        raise ValueError(f"incomplete benchmark: {path}")
    results = []
    for case in data["cases"]:
        pairs = {}
        for row in case["raw"]:
            pair = pairs.setdefault(row["pair"], {})
            if row["reuse"] in pair:
                raise ValueError(f"duplicate arm in pair {row['pair']}")
            pair[row["reuse"]] = row
        if any(set(pair) != {False, True} for pair in pairs.values()):
            raise ValueError("unpaired observations")
        pairs = list(pairs.values())
        base = [p[False]["seconds_per_update"] for p in pairs]
        opt = [p[True]["seconds_per_update"] for p in pairs]
        rng = random.Random(31)
        ratios = []
        for _ in range(5000):
            indices = rng.choices(range(len(pairs)), k=len(pairs))
            ratios.append(statistics.median(base[i] for i in indices) / statistics.median(opt[i] for i in indices))
        ratios.sort()
        peaks = {}
        for name, mode in [("baseline", False), ("optimized", True)]:
            values = [p[mode]["peak_allocated_bytes"] for p in pairs]
            peaks[name] = max(values) if all(v is not None for v in values) else None
        counts = pairs[-1]
        results.append({
            "source": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "measurement_context": data.get("measurement_context"),
            "optimization": data["config"].get("optimization", "teacher"),
            "ce_backward": data["config"].get("ce_backward"),
            "chunk_size": data["config"].get("ce_chunk_size"),
            "compile_tv_kl": data["config"].get("compile_tv_kl", False),
            "anchors": data["config"]["anchors"], "stride": case["stride"], "stage": case["stage"],
            "pairs": len(pairs), "baseline_ms": 1000*statistics.median(base),
            "optimized_ms": 1000*statistics.median(opt),
            "ratio_of_medians": statistics.median(base)/statistics.median(opt),
            "paired_bootstrap_95_interval": [ratios[124], ratios[4874]],
            "peak_allocated_bytes": peaks,
            "parity": case["parity"],
            "last_workload": {"baseline": counts[False]["workload"], "optimized": counts[True]["workload"]},
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    results = [row for path in args.inputs for row in summarize(path)]
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Paired scoring probes", "", "Synthetic shape probes; occupancy context is retained in JSON (exclusive cells are named exclusive). Intervals resample timing pairs, not training seeds; they do not remove interference bias.", "",
             "| Source / case | Baseline ms | Optimized ms | Ratio [95% paired interval] | Peak allocated GiB baseline → optimized | Numerical gate |",
             "|---|---:|---:|---:|---:|---|"]
    for r in results:
        lo, hi = r["paired_bootstrap_95_interval"]
        peak = r["peak_allocated_bytes"]
        memory = "n/a" if peak["baseline"] is None else f'{peak["baseline"]/2**30:.3f} → {peak["optimized"]/2**30:.3f}'
        lines.append(f'| {r["source"]}, stride {r["stride"]}, stage {r["stage"]} | {r["baseline_ms"]:.2f} | {r["optimized_ms"]:.2f} | {r["ratio_of_medians"]:.3f} [{lo:.3f}, {hi:.3f}] | {memory} | {r["parity"]["passed"]} |')
    args.out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
