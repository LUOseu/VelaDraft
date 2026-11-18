#!/usr/bin/env python3
"""Systems table for the D1 harness series.

Reads `evidence/harness/r3_<tag>.jsonl` (one file per series run, arms alternate
per logical batch, warmup untimed) and reports per (store, sub-batch) the paired
per-logical-batch wall ratios vs canonical, median wall, peak alloc, padding-slot
waste over `shapes`, and the loss/grad parity columns.

    python analyze_systems.py --dir evidence/harness --out evidence/systems_table
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(directory: Path):
    by_tag = {}
    for path in sorted(directory.glob("r3_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_tag[path.stem.removeprefix("r3_")] = rows
    return by_tag


def tag_key(tag: str) -> str:
    base = tag.rsplit("_r", 1)[0] if "_r" in tag and tag.rsplit("_r", 1)[1].isdigit() else tag
    return base


def summarize_rows(rows: list[dict]):
    timed = [r for r in rows if not r.get("warmup") and r.get("wall_ms") is not None]
    arms = defaultdict(list)
    for r in timed:
        arms[r["arm"]].append(r)
    # paired wall ratios vs canonical per logical batch
    canon_wall = {r["logical_batch"]: r["wall_ms"] for r in timed if r["arm"] == "canonical"}
    out = {}
    for arm, arm_rows in arms.items():
        if not arm_rows:
            continue
        ctx = sum(s["context_slots"] for r in arm_rows for s in r.get("shapes", []))
        real = sum(s["real_tokens"] for r in arm_rows for s in r.get("shapes", []))
        ratios = [canon_wall[r["logical_batch"]] / r["wall_ms"]
                  for r in arm_rows if r["logical_batch"] in canon_wall and r["wall_ms"]]
        loss_rel = [r["loss_rel_err"] for r in arm_rows if r.get("loss_rel_err") is not None]
        grad_rel = [r["grad_rel_err"] for r in arm_rows if r.get("grad_rel_err") is not None]
        out[arm] = {
            "batches": len(arm_rows),
            "wall_median_ms": statistics.median(r["wall_ms"] for r in arm_rows),
            "paired_speedup_vs_canonical": statistics.median(ratios) if ratios else None,
            "peak_alloc_gib": max(r["peak_alloc_bytes"] for r in arm_rows) / 2**30,
            "padding_waste": (1 - real / ctx) if ctx else None,
            "loss_rel_err_median": statistics.median(loss_rel) if loss_rel else None,
            "loss_rel_err_max": max(loss_rel) if loss_rel else None,
            "grad_rel_err_max": max(grad_rel) if grad_rel else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="evidence/harness")
    ap.add_argument("--out", default="evidence/systems_table")
    args = ap.parse_args()
    by_tag = load_rows(Path(args.dir))
    grouped = defaultdict(list)
    for tag, rows in by_tag.items():
        grouped[tag_key(tag)].extend(rows)
    summary = {}
    for base, rows in sorted(grouped.items()):
        summary[base] = summarize_rows(rows)
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    lines = ["# Systems A/B — paired per-logical-batch medians (canonical = 1.0)", ""]
    order = ["canonical", "cp", "ref", "split", "token", "dual", "cost", "grid"]
    for base, arms in summary.items():
        lines += [f"## {base}", "",
                  "| arm | batches | wall ms | speedup vs canon | peak GiB | pad waste | loss rel med | loss rel max | grad rel max |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for arm in order + [a for a in arms if a not in order]:
            if arm not in arms:
                continue
            s = arms[arm]
            fmt = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
            lines.append("| {arm} | {b} | {w} | {sp} | {pk} | {pw} | {lr} | {lx} | {gx} |".format(
                arm=arm, b=s["batches"], w=fmt(s["wall_median_ms"], 1), sp=fmt(s["paired_speedup_vs_canonical"], 3),
                pk=fmt(s["peak_alloc_gib"], 2), pw=fmt((s["padding_waste"] or 0) * 100, 1) + "%",
                lr=fmt(s["loss_rel_err_median"], 5), lx=fmt(s["loss_rel_err_max"], 5),
                gx=fmt(s["grad_rel_err_max"], 5)))
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    for base in summary:
        arms = summary[base]
        if "grid" in arms and arms["grid"]["paired_speedup_vs_canonical"]:
            print(f"{base}: grid speedup {arms['grid']['paired_speedup_vs_canonical']:.3f}, "
                  f"split {arms['split']['paired_speedup_vs_canonical']:.3f}, "
                  f"token {arms['token']['paired_speedup_vs_canonical']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
