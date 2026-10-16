#!/usr/bin/env python
"""R3/R5 curves for the round report: train loss / train-side acceptance, and the
real-decode dev AL at each milestone. Reads the live logs of both runs and writes
one PNG (+ a small CSV dump) next to the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_train_log(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def read_total_steps(path: str) -> int | None:
    """Total optimizer steps of a run, from its last progress tick."""
    if not os.path.exists(path):
        return None
    total = None
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "tick" and row.get("total_steps"):
                total = int(row["total_steps"])
    return total


def read_dev_points(path: str, final_step: int | None = None) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("mean_al") is not None and row.get("num_prompts") == 64:
                ckpt = row["checkpoint"]
                if ckpt == "ckpt_final.pt":
                    # `pfinal` is evaluated on the final weights; it belongs at
                    # the end of the x-axis, not at a step parsed out of its name.
                    if final_step is None:
                        continue
                    step = int(final_step)
                else:
                    step = int(ckpt.replace("ckpt_step", "").replace(".pt", ""))
                rows.append({"step": step, "mean_al": row["mean_al"],
                             "acceptance_rate": row.get("acceptance_rate")})
    return sorted(rows, key=lambda r: r["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lyh/vlm-scale/runs/r3r5_n20000")
    parser.add_argument("--out", default="reports/figures/r3r5_curves.png")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for tag, color in (("r3", "tab:blue"), ("r5", "tab:red")):
        train = read_train_log(str(root / tag / "train" / "train_log.jsonl"))
        if not train:
            continue
        steps = [r["step"] for r in train]
        losses = [r.get("loss") for r in train]
        acc = [r.get("accuracy") for r in train]
        axes[0].plot(steps, losses, color=color, lw=0.7, label=f"{tag.upper()} train loss")
        final_step = read_total_steps(str(root / tag / "driver" / "progress.jsonl"))
        dev = read_dev_points(str(root / tag / "driver" / "periodic_summary.jsonl"), final_step)
        if dev:
            axes[1].plot([d["step"] for d in dev], [d["mean_al"] for d in dev], marker="o",
                         color=color, label=f"{tag.upper()} dev AL (64)")
        # train-side acceptance in a window average for readability
        win = 50
        xs = [steps[i] for i in range(0, len(steps), win)]
        ys = [sum(acc[i:i + win]) / len(acc[i:i + win]) for i in range(0, len(acc), win)]
        axes[2].plot(xs, ys, color=color, lw=1.2, label=f"{tag.upper()} train-side accepted/step")
        with (out.parent / f"r3r5_curve_{tag}.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "loss", "accuracy", "accept_len", "grad_norm", "lr", "recipe_stage"])
            for row in train:
                writer.writerow([row.get("step"), row.get("loss"), row.get("accuracy"),
                                 row.get("accept_len"), row.get("grad_norm"), row.get("lr"),
                                 row.get("recipe_stage")])

    axes[0].set(xlabel="optimizer step", ylabel="block-CE loss (train)",
                title="train loss (stage-0 LK vs stage-1 TV definitions differ)")
    axes[1].set(xlabel="optimizer step", ylabel="mean accepted length",
                title="real-decode dev AL, 64-prompt fixed subset")
    axes[2].set(xlabel="optimizer step", ylabel="accepted tokens / step (train-side)",
                title="train-side acceptance (teacher-forced; not dev AL)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
