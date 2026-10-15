#!/usr/bin/env python3
"""Apply the frozen L* settle rule (PREREGISTRATION.md, section 3) to the
canonical-only stability leg.

settled(L) := |AL(L) - AL(L/2)| / AL(L/2) <= 0.5%%
L* := 400 if settled(400); else 800 if settled(800); else 1600 (extend with a
fresh canonical 1600-step leg: same rule on {800, 1600}; if still unsettled,
L* = 1600 with an explicit "not settled" disclosure).

Reads the stability leg's eval artifacts (`eval/eval_result.json`, plus any
`--extra-result` files such as a tagged multi-checkpoint eval; runs are merged
by checkpoint name, first file wins per step). The end-of-training checkpoint
is named `ckpt_final.pt`; its step is read from `train/train_summary.json`
(`steps`) or `--final-step`.

Default mode prints the canonical AL curve and the settled(400)/settled(800)
ladder. `--extension` mode applies the {800, 1600} clause to the fresh
1600-step leg and emits the final L* verdict. If a checkpoint needed by the
rule is missing, it says exactly which run is required instead of guessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 0.005


def _merge_runs(path: Path, curve: dict, exact: dict, final_step) -> None:
    result = json.loads(path.read_text())
    for name, run in result.get("runs", {}).items():
        digits = "".join(ch for ch in name if ch.isdigit())
        if "init" in name:
            continue
        if digits:
            step = int(digits)
        elif "final" in name and final_step:
            step = int(final_step)
        else:
            continue
        curve.setdefault(step, float(run["mean_al"]))
        ex = result.get("equivalence_check", {}).get("runs", {}).get(name, {})
        if "token_exact" in ex:
            exact.setdefault(step, bool(ex["token_exact"]))


def al_curve(leg_dir: Path, extras=(), final_step=None):
    if final_step is None:
        summary = leg_dir / "train" / "train_summary.json"
        if summary.exists():
            final_step = json.loads(summary.read_text()).get("steps")
    curve, exact = {}, {}
    _merge_runs(leg_dir / "eval" / "eval_result.json", curve, exact, final_step)
    for extra in extras:
        _merge_runs(Path(extra), curve, exact, final_step)
    return curve, exact


def settled(curve, step):
    half = curve.get(step // 2)
    full = curve.get(step)
    if half is None or full is None or half == 0:
        return None
    rel = abs(full - half) / half
    return rel, rel <= TOL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", required=True, help="stability leg run dir")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--extra-result", action="append", default=[],
                    help="additional eval_result*.json to merge (fills missing steps)")
    ap.add_argument("--final-step", type=int, default=None,
                    help="step number of ckpt_final.pt (default: train_summary.json steps)")
    ap.add_argument("--extension", action="store_true",
                    help="apply the {800, 1600} extension clause (fresh 1600-step leg)")
    args = ap.parse_args()
    curve, exact = al_curve(Path(args.leg), args.extra_result, args.final_step)
    if not curve:
        raise SystemExit("no checkpoint runs found in eval_result.json")

    print("canonical stability curve (held-out AL):")
    for step in sorted(curve):
        print(f"  step {step:5d}: AL={curve[step]:.6f} token_exact={exact.get(step)}")

    verdict = {"curve": curve, "token_exact": exact, "tol": TOL}
    if args.extension:
        s1600 = settled(curve, 1600)
        if s1600 is None:
            raise SystemExit(
                f"extension mode needs checkpoints 800 and 1600; available: {sorted(curve)}"
            )
        rel1600, ok1600 = s1600
        verdict["settled_1600"] = {"rel": rel1600, "settled": ok1600}
        print(f"settled(1600): |AL(1600)-AL(800)|/AL(800) = {rel1600:.4%} -> {ok1600}")
        verdict["L_star"] = 1600
        if ok1600:
            verdict["action"] = (
                "L* = 1600, settled(1600) True on the fresh extension leg "
                "(same rule on {800, 1600})"
            )
        else:
            verdict["action"] = (
                f"L* = 1600, NOT settled ((800,1600) rel={rel1600:.4%} > 0.5%): "
                "explicit disclosure path; claim scoped to the matched budget"
            )
        print("DECISION:", verdict["action"])
        if args.out:
            Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n")
        return 0
    s400 = settled(curve, 400)
    if s400 is None:
        raise SystemExit("need checkpoints 200 and 400 to test settled(400)")
    rel400, ok400 = s400
    verdict["settled_400"] = {"rel": rel400, "settled": ok400}
    print(f"settled(400): |AL(400)-AL(200)|/AL(200) = {rel400:.4%} -> {ok400}")
    if ok400:
        verdict["L_star"] = 400
        verdict["action"] = "frozen L* = 400; no extension needed"
    else:
        s800 = settled(curve, 800)
        if s800 is None:
            verdict["L_star"] = None
            verdict["action"] = (
                "settled(400) failed; need ckpt_800 (resume stability leg from step 600 "
                "for 200 more steps) to test settled(800)"
            )
        else:
            rel800, ok800 = s800
            verdict["settled_800"] = {"rel": rel800, "settled": ok800}
            print(f"settled(800): |AL(800)-AL(400)|/AL(400) = {rel800:.4%} -> {ok800}")
            if ok800:
                verdict["L_star"] = 800
                verdict["action"] = "frozen L* = 800"
            else:
                verdict["L_star"] = 1600
                verdict["action"] = "unsettled; L*=1600 per rule (extend with fresh canonical 1600-step leg)"
    if verdict["L_star"] is None:
        print("DECISION:", verdict["action"])
    else:
        print(f"DECISION: L* = {verdict['L_star']}")
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
