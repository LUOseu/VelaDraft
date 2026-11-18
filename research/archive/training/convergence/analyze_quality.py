#!/usr/bin/env python3
"""Frozen analysis for the D1 convergence quality confirmation (PREREGISTRATION.md §5).

Registered statistic (prereg §5): per-seed relative drop
``R_s = mean_p(dAL) / mean_p(AL_canon)`` over the paired prompts, arm statistic
``mean_s(R_s)``, and a hierarchical bootstrap CI (prompts within seed, seeds
across seeds, B=10,000) of the same functional. The run-level aggregate
``mean_al`` (sum(new_tokens)/sum(verify_steps)) is reported for reference only
and never used in the registered statistic.

Leg validity (prereg §5) is enforced before any statistic is computed; invalid
or incomplete evidence yields verdict **HOLD** with explicit reasons instead of
a silent PASS/FAIL:

- ``train/train_summary.json`` present with ``loss_decreased=true``,
  ``nan_seen=false``, ``skipped_no_anchor_batches=0``, finite ``final_loss``;
- ``eval`` token_exact=true for the evaluated checkpoint;
- ``eval_samples.jsonl`` carries exactly ``prompts_per_leg`` unique prompt rows
  for that checkpoint;
- paired arms carry identical prompt-id sets (no intersection fallback);
- every registered seed for an arm is present.

    python analyze_quality.py --data evidence/quality_data.json --out evidence/quality_verdict.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def pick_checkpoint(result: dict) -> str:
    runs = result.get("runs") or {}
    trained = {name: run for name, run in runs.items() if "init" not in name}
    if not trained:
        raise ValueError("eval_result.json has no trained checkpoints")
    if "ckpt_final.pt" in trained:
        return "ckpt_final.pt"
    return sorted(trained, key=lambda name: (name != "ckpt_final.pt", name))[-1]


def load_leg(leg_dir: Path, expected_prompts: int | None, reasons: list[str]) -> dict:
    """Load one leg and enforce the preregistered validity checks."""
    out: dict = {"valid": True, "reasons": []}

    def fail(message: str) -> None:
        out["valid"] = False
        out["reasons"].append(message)
        reasons.append(f"{leg_dir.name}: {message}")

    train_path = leg_dir / "train" / "train_summary.json"
    if not train_path.exists():
        fail("train/train_summary.json missing (leg validity undetermined)")
    else:
        try:
            summary = json.loads(train_path.read_text())
        except Exception as exc:  # noqa: BLE001 - report, then HOLD
            fail(f"train_summary.json unreadable: {exc}")
        else:
            for key, want in (("loss_decreased", True), ("nan_seen", False),
                              ("skipped_no_anchor_batches", 0)):
                if summary.get(key) != want:
                    fail(f"train_summary.{key}={summary.get(key)!r} (want {want!r})")
            final_loss = summary.get("final_loss")
            if not isinstance(final_loss, (int, float)) or not math.isfinite(final_loss):
                fail(f"train_summary.final_loss={final_loss!r} is not finite")

    eval_dir = leg_dir / "eval"
    result = None
    try:
        result = json.loads((eval_dir / "eval_result.json").read_text())
    except Exception as exc:  # noqa: BLE001
        fail(f"eval_result.json unreadable: {exc}")
    if result is not None:
        try:
            ckpt = pick_checkpoint(result)
        except ValueError as exc:
            fail(str(exc))
            ckpt = None
        if ckpt is not None:
            equivalence = result.get("equivalence_check") or {}
            eq_runs = equivalence.get("runs") or {}
            if ckpt in eq_runs:
                token_exact = eq_runs[ckpt].get("token_exact")
            else:
                token_exact = equivalence.get("token_exact")
            if token_exact is not True:
                fail(f"token_exact={token_exact!r} for {ckpt} (prereg: leg invalid)")
            per_prompt: dict = {}
            duplicate = False
            row_error = None
            try:
                lines = (eval_dir / "eval_samples.jsonl").read_text().splitlines()
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("checkpoint", ckpt) != ckpt:
                        continue
                    prompt_id = row.get("id")
                    steps = row.get("verify_steps")
                    drafted = row.get("draft_tokens")
                    if prompt_id is None or not isinstance(steps, (int, float)) or steps <= 0:
                        row_error = f"invalid eval_samples row (id={prompt_id!r}, verify_steps={steps!r})"
                        continue
                    if prompt_id in per_prompt:
                        duplicate = True
                    accepted = row.get("accepted_tokens", 0)
                    per_prompt[prompt_id] = {
                        "al": row["new_tokens"] / steps,
                        "acc": (accepted / drafted) if drafted else 0.0,
                    }
            except Exception as exc:  # noqa: BLE001
                fail(f"eval_samples.jsonl unreadable: {exc}")
            if row_error:
                fail(row_error)
            if duplicate:
                fail("duplicate prompt ids in eval_samples.jsonl")
            if expected_prompts is not None and len(per_prompt) != expected_prompts:
                fail(f"prompts={len(per_prompt)} (expected {expected_prompts} per prereg)")
            out.update({
                "checkpoint": ckpt,
                "mean_al": (result.get("runs", {}).get(ckpt, {}) or {}).get("mean_al"),
                "acceptance_rate": (result.get("runs", {}).get(ckpt, {}) or {}).get("acceptance_rate"),
                "per_slot": (result.get("runs", {}).get(ckpt, {}) or {}).get("per_slot_acceptance"),
                "token_exact": token_exact,
                "per_prompt": per_prompt,
            })
    return out


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(seeds, deltas, ref_al, B, rng):
    """Hierarchical bootstrap of the registered functional; one-sided 5% lower bounds."""
    per_seed_dal = {s: [d["al"] for d in deltas[s]] for s in seeds}
    per_seed_dacc = {s: [d["acc"] for d in deltas[s]] for s in seeds}
    per_seed_ref = {s: list(ref_al[s]) for s in seeds}
    stats_al, stats_acc = [], []
    for _ in range(B):
        picked = [rng.choice(seeds) for _ in seeds]
        rels, accs = [], []
        for seed in picked:
            count = len(per_seed_dal[seed])
            idx = [rng.randrange(count) for _ in range(count)]
            dal = mean([per_seed_dal[seed][i] for i in idx])
            ref = mean([per_seed_ref[seed][i] for i in idx])
            rels.append(dal / ref)
            accs.append(mean([per_seed_dacc[seed][i] for i in idx]))
        stats_al.append(mean(rels))
        stats_acc.append(mean(accs))
    stats_al.sort()
    stats_acc.sort()
    k = max(0, int(0.05 * B) - 1)
    return stats_al[k], stats_acc[k]


def t_ci95(values):
    """Secondary t-based 95% CI across seeds (reported, not gating)."""
    count = len(values)
    if count < 2:
        return None, None
    avg = mean(values)
    sd = math.sqrt(sum((v - avg) ** 2 for v in values) / (count - 1))
    tcrit = _T975.get(count - 1, 1.96)
    half = tcrit * sd / math.sqrt(count)
    return avg - half, avg + half


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
    expected_prompts = spec.get("prompts_per_leg")
    rng = random.Random(int(spec.get("bootstrap_seed", 20261015)))

    global_reasons: list[str] = []
    seed_set_problems: dict[str, list[str]] = {}
    legs: dict[str, dict] = {}
    for arm, per_seed in spec["arms"].items():
        legs[arm] = {}
        registered = spec.get("registered", {}).get(arm)
        if registered is not None:
            missing = [s for s in registered if str(s) not in per_seed]
            extra = [s for s in per_seed if s not in {str(r) for r in registered}]
            if missing:
                reason = f"{arm}: registered seed legs missing from data spec: {missing}"
                global_reasons.append(reason)
                seed_set_problems.setdefault(arm, []).append(reason)
            if extra:
                reason = f"{arm}: data spec has seeds outside the registered set: {extra}"
                global_reasons.append(reason)
                seed_set_problems.setdefault(arm, []).append(reason)
        for seed, leg_dir in per_seed.items():
            legs[arm][seed] = load_leg(Path(leg_dir), expected_prompts, global_reasons)

    report = {
        "reference_arm": ref_arm,
        "gate_al_rel": gate_al,
        "gate_acc_abs": gate_acc,
        "statistic": "R_s = mean_p(dAL) / mean_p(AL_canon); seed mean of R_s; "
                     "hierarchical bootstrap of the same functional (prompts within seed, seeds across seeds)",
        "prompts_per_leg": expected_prompts,
        "arms": {},
        "invalid_evidence": list(global_reasons),
    }
    ref_legs = legs[ref_arm]
    for arm, per_seed in legs.items():
        if arm == ref_arm:
            continue
        arm_reasons = [r for r in global_reasons if r.startswith(f"{arm}:")]
        seeds_all = [s for s in per_seed if s in ref_legs]
        rows, deltas, ref_prompt_al = [], {}, {}
        valid_seeds = []
        for seed in sorted(seeds_all):
            ref_leg, arm_leg = ref_legs[seed], per_seed[seed]
            row = {
                "seed": seed,
                "valid": bool(ref_leg["valid"] and arm_leg["valid"]),
                "reasons": ref_leg["reasons"] + arm_leg["reasons"],
                "canonical_token_exact": ref_leg.get("token_exact"),
                "arm_token_exact": arm_leg.get("token_exact"),
                "canonical_al_aggregate": ref_leg.get("mean_al"),
                "arm_al_aggregate": arm_leg.get("mean_al"),
                "canonical_acc_aggregate": ref_leg.get("acceptance_rate"),
                "arm_acc_aggregate": arm_leg.get("acceptance_rate"),
            }
            if row["valid"]:
                ids_ref = sorted(ref_leg["per_prompt"])
                ids_arm = sorted(arm_leg["per_prompt"])
                if ids_ref != ids_arm:
                    row["valid"] = False
                    row["reasons"].append(
                        f"prompt-id sets differ (ref={len(ids_ref)}, arm={len(ids_arm)})"
                    )
                    arm_reasons.append(f"{arm} seed {seed}: prompt-id sets differ")
                else:
                    d_al = [arm_leg["per_prompt"][i]["al"] - ref_leg["per_prompt"][i]["al"] for i in ids_ref]
                    d_acc = [arm_leg["per_prompt"][i]["acc"] - ref_leg["per_prompt"][i]["acc"] for i in ids_ref]
                    ref_vals = [ref_leg["per_prompt"][i]["al"] for i in ids_ref]
                    ref_mean = mean(ref_vals)
                    row.update({
                        "prompts": len(ids_ref),
                        "canonical_al": ref_mean,
                        "arm_al": mean([arm_leg["per_prompt"][i]["al"] for i in ids_ref]),
                        "d_al": mean(d_al),
                        "rel": mean(d_al) / ref_mean,
                        "canonical_acc": ref_leg.get("acceptance_rate"),
                        "arm_acc": arm_leg.get("acceptance_rate"),
                        "d_acc": mean(d_acc),
                    })
                    deltas[seed] = [{"al": a, "acc": c} for a, c in zip(d_al, d_acc)]
                    ref_prompt_al[seed] = ref_vals
                    valid_seeds.append(seed)
            rows.append(row)
        for row in rows:
            for reason in row["reasons"]:
                arm_reasons.append(f"{arm}/{ref_arm} seed {row['seed']}: {reason}")
        block = {
            "seeds": rows,
            "mean_rel_al_drop": mean([r["rel"] for r in rows if r.get("rel") is not None]),
            "mean_dacc": mean([r["d_acc"] for r in rows if r.get("d_acc") is not None]),
            "bootstrap_lo_rel_al": None,
            "bootstrap_lo_dacc": None,
            "pass_al": None,
            "pass_acc": None,
            "verdict": "HOLD",
            "hold_reasons": list(dict.fromkeys(arm_reasons)),
        }
        if valid_seeds:
            lo_al, lo_acc = bootstrap_ci(sorted(valid_seeds), deltas, ref_prompt_al, B, rng)
            block["bootstrap_lo_rel_al"] = lo_al
            block["bootstrap_lo_dacc"] = lo_acc
            block["pass_al"] = lo_al > gate_al
            block["pass_acc"] = lo_acc > gate_acc
            rel_values = [r["rel"] for r in rows if r.get("rel") is not None]
            t_lo, t_hi = t_ci95(rel_values)
            block["t_ci95_rel_al"] = [t_lo, t_hi]
            block["mean_rel_al_drop_sd"] = (
                math.sqrt(sum((v - mean(rel_values)) ** 2 for v in rel_values) / (len(rel_values) - 1))
                if len(rel_values) > 1 else None
            )
            blocked = ([r for r in rows if not r["valid"]]
                       + seed_set_problems.get(arm, []) + seed_set_problems.get(ref_arm, []))
            block["hold_reasons"] = list(dict.fromkeys(
                arm_reasons + list(seed_set_problems.get(arm, []))
                + list(seed_set_problems.get(ref_arm, []))))
            if blocked:
                block["verdict"] = "HOLD"
            else:
                block["verdict"] = "PASS" if (block["pass_al"] and block["pass_acc"]) else "FAIL"
        else:
            block["hold_reasons"] = list(dict.fromkeys(
                arm_reasons + list(seed_set_problems.get(arm, []))
                + list(seed_set_problems.get(ref_arm, []))
                + ["no seed with valid paired evidence"]
            ))
        report["arms"][arm] = block
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("arms",)}, indent=2))
    for arm, block in report["arms"].items():
        if block["bootstrap_lo_rel_al"] is None:
            print(f"[{args.tag or 'quality'}] {arm}: HOLD - " + "; ".join(block["hold_reasons"][:4]))
            continue
        print(f"[{args.tag or 'quality'}] {arm}: mean relAL {block['mean_rel_al_drop']*100:+.3f}% "
              f"(lo {block['bootstrap_lo_rel_al']*100:+.3f}%) dacc {block['mean_dacc']:+.5f} "
              f"(lo {block['bootstrap_lo_dacc']:+.5f}) -> {block['verdict']}")
        for reason in block["hold_reasons"]:
            print(f"    HOLD reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
