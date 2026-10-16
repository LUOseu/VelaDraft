#!/usr/bin/env python
"""Collect every R3/R5 dev/test eval point plus train-side summaries into one
machine-readable table and a markdown dump for the round report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = [
    "mean_al", "acceptance_rate", "per_slot_acceptance", "target_forward_calls",
    "speculative_rounds", "verify_steps", "tokens_per_target_forward",
    "target_forwards_per_round", "num_samples", "new_tokens", "accepted_tokens",
    "draft_tokens", "draft_seconds_per_sample", "target_verify_seconds_per_sample",
    "target_prefill_seconds_per_sample", "wall_seconds_per_sample",
    "speedup_vs_target_only", "role",
]


def arm_points(root: Path, arm: str) -> dict:
    out: dict = {"arm": arm, "eval": {}, "periodic": [], "selection": None, "train": {}}
    evaldir = root / arm / "eval"
    for f in sorted(evaldir.glob("eval_result_*.json")):
        tag = f.name.replace("eval_result_", "").replace(".json", "")
        data = json.loads(f.read_text())
        runs = {}
        for ck, row in data.get("runs", {}).items():
            runs[ck] = {k: row.get(k) for k in FIELDS}
        out["eval"][tag] = {
            "runs": runs,
            "equivalence_check": data.get("equivalence_check"),
            "target_only": data.get("target_only"),
            "settings": data.get("settings"),
            "al_definition": data.get("al_definition"),
        }
    per = root / arm / "driver" / "periodic_summary.jsonl"
    if per.exists():
        for line in per.read_text().splitlines():
            if line.strip():
                out["periodic"].append(json.loads(line))
    sel = root / arm / "driver" / "selection.json"
    if sel.exists():
        out["selection"] = json.loads(sel.read_text())
    tlog = root / arm / "train" / "train_log.jsonl"
    if tlog.exists():
        rows = [json.loads(l) for l in tlog.read_text().splitlines() if l.strip()]
        last = rows[-1] if rows else {}
        out["train"] = {
            "rows": len(rows),
            "last_step": last.get("step"),
            "last": last,
            "wall_seconds": last.get("wall_seconds"),
        }
    summ = root / arm / "train" / "train_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        out["train_summary"] = {
            "steps": s.get("steps"),
            "final_loss": s.get("final_loss"),
            "feature_source": s.get("feature_source"),
        }
    prog = root / arm / "driver" / "progress.jsonl"
    if prog.exists():
        ticks = [json.loads(l) for l in prog.read_text().splitlines() if l.strip()]
        ev = [t for t in ticks if t.get("event") == "run_complete"]
        out["run_complete"] = ev[-1] if ev else None
        last_tick = next((t for t in reversed(ticks) if t.get("event") == "tick"), None)
        out["last_tick"] = last_tick
    return out


def md_table(points: dict) -> str:
    lines = ["| arm | tag | ckpt | AL | acc | slots | tok/tfw | tfw | rounds | n |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, data in points.items():
        for tag, block in sorted(data["eval"].items()):
            for ck, row in block["runs"].items():
                slots = row.get("per_slot_acceptance") or []
                lines.append(
                    f"| {arm} | {tag} | {ck} | {row['mean_al']:.4f} | {row['acceptance_rate']:.4f} | "
                    f"{' '.join(f'{s:.3f}' for s in slots)} | {row['tokens_per_target_forward']:.4f} | "
                    f"{row['target_forward_calls']:.0f} | {row['speculative_rounds']:.0f} | "
                    f"{row['num_samples']} |"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lyh/vlm-scale/runs/r3r5_n20000")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    points = {arm: arm_points(root, arm) for arm in ("r3", "r5")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(points, indent=2))
    table = md_table(points)
    Path(str(args.out) + ".md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
