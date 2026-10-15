#!/usr/bin/env python3
"""Frozen analysis for the D1 convergence quality confirmation (PREREGISTRATION.md §5).

Reads per-leg eval artifacts (eval_result.json + eval_samples.jsonl, checkpoint
`ckpt_final.pt`), pairs prompts by id, and computes the preregistered
non-inferiority statistics: per-seed paired mean deltas, relative AL drop, and a
hierarchical bootstrap (prompts within seed, seeds across seeds).

    python analyze_quality.py --data evidence/quality_data.json --out evidence/quality_verdict.json

`quality_data.json` schema:
{
  "reference_arm": "canonical",
  "arms": {"canonical": {"42": "<leg dir>", ...}, "grid2": {...}},
  "gate_al_rel": -0.01, "gate_acc_abs": -0.005, "bootstrap": 10000, "bootstrap_seed": 20261015
}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_leg(leg_dir: Path):
    result = json.loads((leg_dir / "eval" / "eval_result.json").read_text())
    trained = {name: run for name, run in result["runs"].items() if "init" not in name}
    if "ckpt_final.pt" in trained:
        run = trained["ckpt_final.pt"]
        ckpt = "ckpt_final.pt"
    else:  # stability legs carry extra checkpoints; take the largest step
        ckpt = sorted(trained, key=lambda n: (n != "ckpt_final.pt", n))[-1]
        run = trained[ckpt]
    samples = [
        json.loads(line)
        for line in (leg_dir / "eval" / "eval_samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    token_exact = bool(result["equivalence_check"]["token_exact"])
    if ckpt != "ckpt_final.pt":
        # per-sample rows for a non-final checkpoint are fine for stability curves
        pass
    per_prompt = {}
    for row in samples:
        if row.get("checkpoint", ckpt) != ckpt:
            continue
        steps = row["verify_steps"]
        drafted = row["draft_tokens"]
        per_prompt[row["id"]] = {
            "al": row["new_tokens"] / steps,
            "acc": (row["accepted_tokens"] / drafted) if drafted else 0.0,
        }
    return {
        "mean_al": run["mean_al"],
        "acceptance_rate": run["acceptance_rate"],
        "per_slot": run["per_slot_acceptance"],
        "token_exact": token_exact,
        "per_prompt": per_prompt,
        "checkpoint": ckpt,
    }


def paired_deltas(ref: dict, arm: dict):
    ids = sorted(set(ref["per_prompt"]) & set(arm["per_prompt"]))
    assert ids, "no shared prompt ids"
    d_al = [arm["per_prompt"][i]["al"] - ref["per_prompt"][i]["al"] for i in ids]
    d_acc = [arm["per_prompt"][i]["acc"] - ref["per_prompt"][i]["acc"] for i in ids]
    return ids, d_al, d_acc


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def rel_drop(ref_al, arm_al):
    return (arm_al - ref_al) / ref_al


def bootstrap_ci(seeds, deltas, seed_al, B, rng):
    """Hierarchical bootstrap over seeds and prompts; returns (lo_al, lo_acc) one-sided 5%."""
    per_seed_al = {}
    per_seed_acc = {}
    for s in seeds:
        per_seed_al[s] = [d["al"] for d in deltas[s]]
        per_seed_acc[s] = [d["acc"] for d in deltas[s]]
    stats_al, stats_acc = [], []
    for _ in range(B):
        picked = [rng.choice(seeds) for _ in seeds]
        rs, accs = [], []
        for s in picked:
            idx = [rng.randrange(len(per_seed_al[s])) for _ in per_seed_al[s]]
            dal = mean([per_seed_al[s][i] for i in idx])
            dacc = mean([per_seed_acc[s][i] for i in idx])
            rs.append(dal / seed_al[s])
            accs.append(dacc)
        stats_al.append(mean(rs))
        stats_acc.append(mean(accs))
    stats_al.sort()
    stats_acc.sort()
    k = max(0, int(0.05 * B) - 1)
    return stats_al[k], stats_acc[k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    spec = json.loads(Path(args.data).read_text())
    ref_arm = spec["reference_arm"]
    gate_al = float(spec.get("gate_al_rel", -0.01))
    gate_acc = float(spec.get("gate_acc_abs", -0.005))
    B = int(spec.get("bootstrap", 10000))
    rng = random.Random(int(spec.get("bootstrap_seed", 20261015)))

    legs = {}
    for arm, per_seed in spec["arms"].items():
        legs[arm] = {}
        for seed, leg_dir in per_seed.items():
            legs[arm][seed] = load_leg(Path(leg_dir))

    report = {"reference_arm": ref_arm, "gate_al_rel": gate_al, "gate_acc_abs": gate_acc, "arms": {}}
    ref_legs = legs[ref_arm]
    for arm, per_seed in legs.items():
        if arm == ref_arm:
            continue
        seeds = [s for s in per_seed if s in ref_legs]
        deltas = {}
        seed_al = {}
        rows = []
        for s in sorted(seeds):
            ids, d_al, d_acc = paired_deltas(ref_legs[s], per_seed[s])
            deltas[s] = [{"al": a, "acc": c} for a, c in zip(d_al, d_acc)]
            seed_al[s] = ref_legs[s]["mean_al"]
            rows.append({
                "seed": s,
                "canonical_al": ref_legs[s]["mean_al"], "arm_al": per_seed[s]["mean_al"],
                "d_al": mean(d_al), "rel": rel_drop(seed_al[s], per_seed[s]["mean_al"]),
                "canonical_acc": ref_legs[s]["acceptance_rate"], "arm_acc": per_seed[s]["acceptance_rate"],
                "d_acc": mean(d_acc),
                "canonical_token_exact": ref_legs[s]["token_exact"], "arm_token_exact": per_seed[s]["token_exact"],
                "prompts": len(ids),
            })
        mean_rel = mean([row["rel"] for row in rows])
        mean_dacc = mean([row["d_acc"] for row in rows])
        lo_al, lo_acc = bootstrap_ci(sorted(seeds), deltas, seed_al, B, rng)
        verdict = "PASS" if (lo_al > gate_al and lo_acc > gate_acc) else "FAIL"
        report["arms"][arm] = {
            "seeds": rows, "mean_rel_al_drop": mean_rel, "mean_dacc": mean_dacc,
            "bootstrap_lo_rel_al": lo_al, "bootstrap_lo_dacc": lo_acc,
            "pass_al": lo_al > gate_al, "pass_acc": lo_acc > gate_acc, "verdict": verdict,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    for arm, block in report["arms"].items():
        print(f"[{args.tag or 'quality'}] {arm}: mean relAL {block['mean_rel_al_drop']*100:+.3f}% "
              f"(lo {block['bootstrap_lo_rel_al']*100:+.3f}%) dacc {block['mean_dacc']:+.5f} "
              f"(lo {block['bootstrap_lo_dacc']:+.5f}) -> {block['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
