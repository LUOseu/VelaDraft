#!/usr/bin/env python3
"""Apply the frozen L* settle rule (PREREGISTRATION.md, section 3) to the
canonical-only stability leg.

settled(L) := |AL(L) - AL(L/2)| / AL(L/2) <= 0.5%%
L* := 400 if settled(400); else 800 if settled(800); else 1600 (extend).

Reads the stability leg's eval_result.json (multi-checkpoint eval); prints the
canonical AL curve for steps 200/400/600/800 (whichever exist), the settle
tests, and the resulting L* decision. If a checkpoint needed by the rule is
missing, it says exactly which extension run is required instead of guessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 0.005


def al_curve(leg_dir: Path):
    result = json.loads((leg_dir / "eval" / "eval_result.json").read_text())
    curve, exact = {}, {}
    for name, run in result.get("runs", {}).items():
        digits = "".join(ch for ch in name if ch.isdigit())
        if not digits or "init" in name:
            continue
        step = int(digits)
        curve[step] = float(run["mean_al"])
        ex = result.get("equivalence_check", {}).get("runs", {}).get(name, {})
        exact[step] = bool(ex.get("token_exact", False))
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
    args = ap.parse_args()
    curve, exact = al_curve(Path(args.leg))
    if not curve:
        raise SystemExit("no checkpoint runs found in eval_result.json")

    print("canonical stability curve (held-out AL):")
    for step in sorted(curve):
        print(f"  step {step:5d}: AL={curve[step]:.6f} token_exact={exact.get(step)}")

    verdict = {"curve": curve, "token_exact": exact, "tol": TOL}
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
