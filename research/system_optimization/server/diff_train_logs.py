"""Compare per-step training logs across S2 legs (a=OFF, b=ON).

Prints, per pair, the first step at which each field diverges and the max abs
delta over the run (and whether identical). Also cross-checks the three OFF
legs and the three ON legs for run-to-run determinism.
"""
import json
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent.parent / "evidence"
FIELDS = ["loss", "accuracy", "accept_len", "valid_tokens", "grad_norm", "slot0_accuracy"]


def load(tag):
    rows = [json.loads(line) for line in (EV / f"s2_{tag}_train_log.jsonl").read_text().splitlines()]
    return {int(r["step"]): r for r in rows}


def first_diff(x, y, field):
    for step in sorted(x):
        a, b = x[step][field], y[step][field]
        if a != b:
            return step
    return None


def max_delta(x, y, field):
    worst = 0.0
    for step in x:
        if step in y:
            worst = max(worst, abs(float(x[step][field]) - float(y[step][field])))
    return worst


def compare(tag_a, tag_b):
    a, b = load(tag_a), load(tag_b)
    print(f"== {tag_a} vs {tag_b} ==")
    for field in FIELDS:
        step = first_diff(a, b, field)
        if step is None:
            print(f"  {field:14s}: identical over all steps")
        else:
            print(f"  {field:14s}: first diff step {step}, max|delta| = {max_delta(a, b, field):.6g}")


if __name__ == "__main__":
    pairs = [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a1", "a2"), ("a2", "a3"), ("b1", "b2"), ("b2", "b3")]
    sys.stdout.write("\n".join("") )
    for pair in pairs:
        compare(*pair)
        print()
    # exact loss-delta histogram for the interesting pair
    a, b = load("a1"), load("b1")
    deltas = {}
    for step in a:
        d = abs(float(a[step]["loss"]) - float(b[step]["loss"]))
        deltas[d] = deltas.get(d, 0) + 1
    print("a1-vs-b1 loss |delta| histogram:", dict(sorted(deltas.items())))
