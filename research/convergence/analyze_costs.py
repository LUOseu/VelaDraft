#!/usr/bin/env python3
"""Effect + cost table from fetched leg evidence (train_summary + eval_result).

    python analyze_costs.py --legs-dir evidence/legs --out evidence/costs_table

Per leg: steps, train wall, peak mem, regroup counters; eval run ckpt_final:
mean_al, acceptance_rate, per-sample wall, target prefill/verify, draft seconds,
speedup vs target-only, token_exact. GPU-hours (train+eval) use the recorded
wall where available and the per-sample eval timing × num samples otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_leg(leg_dir: Path):
    row = {"leg": leg_dir.name}
    ts = leg_dir / "train" / "train_summary.json"
    if ts.exists():
        s = json.loads(ts.read_text())
        row.update(train_steps=s.get("steps"), train_wall_s=s.get("wall_seconds"),
                   peak_gpu_mem_gb=s.get("peak_gpu_mem_gb"), final_loss=s.get("final_loss"),
                   loss_decreased=s.get("loss_decreased"), nan_seen=s.get("nan_seen"),
                   skipped_no_anchor=s.get("skipped_no_anchor_batches"))
        if s.get("regroup"):
            row["regroup"] = {k: s["regroup"].get(k) for k in ("mode", "sub_batch", "identity_batches", "batches")}
    er = leg_dir / "eval" / "eval_result.json"
    if er.exists():
        d = json.loads(er.read_text())
        runs = {k: v for k, v in d.get("runs", {}).items() if "init" not in k}
        ck = "ckpt_final.pt" if "ckpt_final.pt" in runs else (sorted(runs)[-1] if runs else None)
        if ck:
            r = runs[ck]
            n = r.get("num_samples") or 64
            row.update(ckpt=ck, mean_al=r.get("mean_al"), acceptance_rate=r.get("acceptance_rate"),
                       per_slot=r.get("per_slot_acceptance"), token_exact=d.get("equivalence_check", {}).get("token_exact"),
                       eval_wall_s_per_sample=r.get("wall_seconds_per_sample"),
                       target_prefill_s=r.get("target_prefill_seconds_per_sample"),
                       target_verify_s=r.get("target_verify_seconds_per_sample"),
                       draft_s=r.get("draft_seconds_per_sample"),
                       speedup_vs_target_only=r.get("speedup_vs_target_only"))
            if row.get("eval_wall_s_per_sample"):
                row["eval_wall_s_total"] = row["eval_wall_s_per_sample"] * n
        row["target_only_wall_s_per_sample"] = d.get("target_only", {}).get("wall_seconds_per_sample")
    return row


def fmt(v, d=3):
    return "" if v is None else f"{v:.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs-dir", default="evidence/legs")
    ap.add_argument("--out", default="evidence/costs_table")
    args = ap.parse_args()
    legs = sorted(p for p in Path(args.legs_dir).iterdir() if p.is_dir())
    rows = [load_leg(p) for p in legs]
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    lines = ["# Effect + cost (per leg)", "",
             "| leg | steps | train wall s | train GPU-h | peak GiB | AL | acc | eval s/sample | draft s | verify s | target-only s | speedup | token_exact |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {leg} | {st} | {tw} | {gh} | {pk} | {al} | {ac} | {ew} | {dr} | {tv} | {to} | {sp} | {te} |".format(
            leg=r["leg"], st=fmt(r.get("train_steps"), 0), tw=fmt(r.get("train_wall_s"), 0),
            gh=fmt((r.get("train_wall_s") or 0) / 3600, 2), pk=fmt(r.get("peak_gpu_mem_gb"), 1),
            al=fmt(r.get("mean_al")), ac=fmt(r.get("acceptance_rate"), 4),
            ew=fmt(r.get("eval_wall_s_per_sample"), 3), dr=fmt(r.get("draft_s"), 3),
            tv=fmt(r.get("target_verify_s"), 3), to=fmt(r.get("target_only_wall_s_per_sample"), 3),
            sp=fmt(r.get("speedup_vs_target_only"), 2), te=r.get("token_exact")))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')} ({len(rows)} legs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
