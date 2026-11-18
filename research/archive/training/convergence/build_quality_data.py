#!/usr/bin/env python3
"""Build `quality_data.json` for analyze_quality.py from fetched leg evidence.

    python build_quality_data.py --legs-dir evidence/legs --out evidence/quality_data.json

Expects trees fetched by scripts/fetch_evidence.sh:
    evidence/legs/conv_canon_s42/{eval,train}/...
    evidence/legs/conv_grid2_s42/...

Arm mapping is fixed by PREREGISTRATION.md section 4 (canonical/grid2 = confirmatory
seeds 42-45; token2/split2 = pilot seeds 42-43). Gate constants are copied from
the prereg and are not to be edited here without a prereg amendment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {
    "canonical": {s: f"conv_canon_s{s}" for s in (42, 43, 44, 45)},
    "grid2": {s: f"conv_grid2_s{s}" for s in (42, 43, 44, 45)},
    "token2": {s: f"conv_token2_s{s}" for s in (42, 43)},
    "split2": {s: f"conv_split2_s{s}" for s in (42, 43)},
}
GATES = {"gate_al_rel": -0.01, "gate_acc_abs": -0.005, "bootstrap": 10000, "bootstrap_seed": 20261015}
PROMPTS_PER_LEG = 64  # d1q_val64 held-out split (prereg section 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs-dir", default="evidence/legs")
    ap.add_argument("--out", default="evidence/quality_data.json")
    ap.add_argument("--include-pilot", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the spec even when registered legs are absent (exploration only)")
    args = ap.parse_args()
    legs_dir = Path(args.legs_dir)
    arms, missing = {}, []
    for arm, seeds in ARMS.items():
        if arm in ("token2", "split2") and not args.include_pilot:
            continue
        arms[arm] = {}
        for seed, leg in seeds.items():
            if (legs_dir / leg / "eval" / "eval_result.json").exists():
                arms[arm][str(seed)] = str((legs_dir / leg).resolve())
            else:
                missing.append(leg)
    if not arms.get("canonical") or not arms.get("grid2") or (missing and not args.allow_missing):
        raise SystemExit(f"registered legs incomplete; missing: {missing} (use --allow-missing to override)")
    registered = {arm: sorted(seeds) for arm, seeds in ARMS.items()
                  if arm in arms or (arm in ("token2", "split2") and args.include_pilot)}
    data = {"reference_arm": "canonical", "arms": arms, "registered": registered,
            "prompts_per_leg": PROMPTS_PER_LEG, **GATES}
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: " + ", ".join(f"{a}={sorted(s)}" for a, s in arms.items()))
    if missing:
        print("missing (not included):", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
