#!/usr/bin/env python
"""Paired per-sample comparison of two evaluation runs (e.g. R5 vs R3 at the
same progress) with bootstrap CIs. Single-seed limitation is reported, not
papered over.

Pairing rule: prompts are keyed by (id, occurrence_index) so that duplicate ids
inside a split are paired by position instead of being silently dropped.
Both AL flavours are reported:
  * al_micro  = sum(new_tokens) / sum(verify_steps)  -- token-level, matches
                the headline `mean_al` written by the eval script
  * al_macro  = mean over prompts of (new_tokens / verify_steps)
They are different numbers and must not be mixed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_rows(path: str) -> tuple[list[dict], dict[tuple, dict]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    seen: dict[str, int] = {}
    keyed: dict[tuple, dict] = {}
    for row in rows:
        pid = row["id"]
        idx = seen.get(pid, 0)
        seen[pid] = idx + 1
        keyed[(pid, idx)] = row
    return rows, keyed


def sample_metrics(row: dict) -> dict:
    rounds = max(1, int(row["verify_steps"]))
    drafted = max(1, int(row["draft_tokens"]))
    tfw = max(1, int(row.get("target_forward_calls", 0)) or 1)
    return {
        "al_macro": row["new_tokens"] / rounds,
        "accept_rate": row["accepted_tokens"] / drafted,
        "tokens_per_tfw": row["new_tokens"] / tfw,
        "new_tokens": row["new_tokens"],
        "rounds": rounds,
        "accepted_tokens": row["accepted_tokens"],
        "draft_tokens": drafted,
        "target_forward_calls": tfw,
    }


def corpus_al(rows: list[dict]) -> dict:
    new_tokens = sum(int(r["new_tokens"]) for r in rows)
    rounds = sum(int(r["verify_steps"]) for r in rows)
    accepted = sum(int(r["accepted_tokens"]) for r in rows)
    drafted = sum(int(r["draft_tokens"]) for r in rows)
    tfw = sum(int(r.get("target_forward_calls", 0)) for r in rows)
    return {
        "num_prompts": len(rows),
        "new_tokens": new_tokens,
        "verify_steps": rounds,
        "accepted_tokens": accepted,
        "draft_tokens": drafted,
        "target_forward_calls": tfw,
        "al_micro": new_tokens / max(1, rounds),
        "accept_rate_micro": accepted / max(1, drafted),
        "tokens_per_target_forward": new_tokens / max(1, tfw),
    }


def bootstrap_ci(diffs: list[float], *, seed: int = 42, samples: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(samples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * samples)]
    hi = means[min(samples - 1, int(0.975 * samples))]
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-a", required=True, help="eval_samples jsonl of arm A (e.g. R3)")
    parser.add_argument("--samples-b", required=True, help="eval_samples jsonl of arm B (e.g. R5)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows_a, keyed_a = load_rows(args.samples_a)
    rows_b, keyed_b = load_rows(args.samples_b)

    common = sorted(set(keyed_a) & set(keyed_b), key=lambda k: (str(k[0]), k[1]))
    dropped_a = set(keyed_a) - set(common)
    dropped_b = set(keyed_b) - set(common)
    if not common:
        raise SystemExit("no common (id, occurrence) keys between the two sample files")

    metrics = ["al_macro", "accept_rate", "tokens_per_tfw"]
    per_metric = {}
    for metric in metrics:
        values_a = []
        values_b = []
        diffs = []
        for pid in common:
            va = sample_metrics(keyed_a[pid])[metric]
            vb = sample_metrics(keyed_b[pid])[metric]
            values_a.append(va)
            values_b.append(vb)
            diffs.append(vb - va)
        lo, hi = bootstrap_ci(diffs)
        per_metric[metric] = {
            "mean_a": sum(values_a) / len(values_a),
            "mean_b": sum(values_b) / len(values_b),
            "mean_diff_b_minus_a": sum(diffs) / len(diffs),
            "median_diff": sorted(diffs)[len(diffs) // 2],
            "bootstrap_95ci": [lo, hi],
            "wins_b": sum(1 for d in diffs if d > 0),
            "losses_b": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
            "per_prompt_diff": {f"{pid[0]}#{pid[1]}": diff for pid, diff in zip(common, diffs)},
        }

    # micro (corpus-level) AL, computed independently inside each arm
    micro = {"a": corpus_al(rows_a), "b": corpus_al(rows_b)}
    micro["diff_b_minus_a_al_micro"] = micro["b"]["al_micro"] - micro["a"]["al_micro"]

    report = {
        "format": "vlm-dflash-paired-analysis-v2",
        "label_a": args.label_a,
        "label_b": args.label_b,
        "num_prompts": len(common),
        "prompts_a": len(rows_a),
        "prompts_b": len(rows_b),
        "unpaired_a": len(dropped_a),
        "unpaired_b": len(dropped_b),
        "metrics": per_metric,
        "corpus_micro": micro,
        "al_definitions": {
            "al_micro": "sum(new_tokens)/sum(verify_steps) over the split; equals headline mean_al of the eval script",
            "al_macro": "mean over prompts of new_tokens/verify_steps",
        },
        "caveat": (
            "single training seed; bootstrap CI quantifies prompt-sampling uncertainty only, "
            "not seed-to-seed variance"
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"paired analysis {args.label_a} vs {args.label_b} on {len(common)} prompts"
          f" (a={len(rows_a)}, b={len(rows_b)}, unpaired {len(dropped_a)}/{len(dropped_b)})")
    for metric, block in per_metric.items():
        print(
            f"  {metric}: {args.label_a}={block['mean_a']:.4f} {args.label_b}={block['mean_b']:.4f} "
            f"diff={block['mean_diff_b_minus_a']:+.4f} CI95=[{block['bootstrap_95ci'][0]:+.4f},"
            f"{block['bootstrap_95ci'][1]:+.4f}] wins={block['wins_b']} losses={block['losses_b']}"
        )
    print(
        f"  al_micro: {args.label_a}={micro['a']['al_micro']:.4f} {args.label_b}={micro['b']['al_micro']:.4f} "
        f"diff={micro['diff_b_minus_a_al_micro']:+.4f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
