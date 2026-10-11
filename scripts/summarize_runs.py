#!/usr/bin/env python3
"""Collect eval headlines across run directories into one table.

    python scripts/summarize_runs.py runs/qwen25vl3b_refresh runs/qwen25vl3b_2k --tag strict32

Reads `eval/eval_result[<tag>].json` from each run dir and prints a markdown
table (run, prompts, verification, token-exact, AL trained / random-init,
per-slot acceptance). Used to build the scaling table in the README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(run_dir: Path, tag: str | None) -> dict | None:
    name = f"eval_result_{tag}.json" if tag else "eval_result.json"
    path = run_dir / "eval" / name
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--tag", default=None, help="read eval_result_<tag>.json instead")
    args = parser.parse_args()

    header = "| run | prompts | verification | token-exact | AL trained | AL random-init | per-slot (trained) |"
    print(header)
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for run_dir in args.run_dirs:
        run = Path(run_dir)
        result = load_result(run, args.tag)
        if result is None:
            print(f"| {run.name} | - | - | (no eval result) | - | - | - |")
            continue
        eq = result["equivalence_check"]
        runs = result.get("runs", {})
        trained = next((v for name, v in runs.items() if "init" not in name), None)
        baseline = next((v for name, v in runs.items() if "init" in name), None)
        slots = trained["per_slot_acceptance"] if trained else []
        slot_text = "[" + ", ".join("null" if s is None else f"{s:.3f}" for s in slots) + "]"
        print(
            f"| {run.name} | {result['settings']['num_prompts']} | {eq.get('verification', 'sequential')} | "
            f"{eq['token_exact']} | {trained['mean_al']:.3f} | "
            f"{baseline['mean_al']:.3f} | {slot_text} |"
            if trained and baseline
            else f"| {run.name} | {result['settings']['num_prompts']} | - | {eq['token_exact']} | - | - | - |"
        )


if __name__ == "__main__":
    main()
