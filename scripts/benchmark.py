#!/usr/bin/env python3
"""Benchmark a trained run against fixed workload manifests.

    python scripts/benchmark.py --run-dir runs/qwen25vl3b_2k \
        --workloads benchmarks/manifests --workload caption_general

Writes `benchmarks/reports/benchmark_<run>.json` + `.md` with the fixed
protocol (greedy, sequential verification) and per-workload AL / per-slot
acceptance / token-exactness / wall-time numbers. See docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.benchmark.harness import load_workloads, run_benchmark, select_workloads
from vlm_dflash.config import apply_overrides, load_config
from vlm_dflash.contract import ensure_contract
from vlm_dflash.utils import log

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None, help="run config override (default: <run>/config.json)")
    parser.add_argument("--workloads", default=str(REPO_ROOT / "benchmarks" / "manifests"))
    parser.add_argument("--workload", action="append", default=[], help="repeatable workload name filter")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "benchmarks" / "reports"))
    parser.add_argument("--checkpoint", default="ckpt_final.pt")
    parser.add_argument("--num-prompts", type=int, default=None, help="override every workload's prompt count")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--verification",
        default="sequential",
        choices=["sequential", "parallel", "strict_parallel"],
        help="sequential (default) is the correctness protocol; parallel is experimental runtime only",
    )
    parser.add_argument("--strict-margin", type=float, default=None)
    parser.add_argument("--name", default=None, help="report file stem (default: benchmark_<run>)")
    parser.add_argument("--list", action="store_true", help="list workloads and exit")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir {run_dir} does not exist.")
    workloads = load_workloads(args.workloads)
    workloads = select_workloads(workloads, args.workload or None)
    if args.list:
        for workload in workloads:
            print(f"{workload['name']:>20}  [{workload['status']}]  {workload['description']}")
        return

    if args.config:
        cfg = apply_overrides(load_config(args.config), args.set)
    else:
        import json

        with open(run_dir / "config.json", "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
    cfg.setdefault("run", {})["dir"] = str(run_dir)
    contract = ensure_contract(cfg, run_dir)

    run_benchmark(
        cfg,
        run_dir,
        contract,
        workloads,
        out_dir=args.out_dir,
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        verification=args.verification,
        strict_parallel_margin=args.strict_margin,
        checkpoint=args.checkpoint,
        name=args.name,
    )
    log("benchmark: done")


if __name__ == "__main__":
    main()
