#!/usr/bin/env python3
"""Stage 3: acceptance evaluation with the token-exact equivalence gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.config import apply_overrides, dump_effective_config, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.eval.run import evaluate
from vlm_dflash.utils import log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--verification", default=None, choices=["sequential", "parallel", "strict_parallel"])
    parser.add_argument("--strict-margin", type=float, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    dump_effective_config(cfg, run_dir / "config.json")
    if args.verification is not None:
        cfg.setdefault("eval", {})["verification"] = args.verification
    if args.strict_margin is not None:
        cfg.setdefault("eval", {})["strict_parallel_margin"] = float(args.strict_margin)
    contract = ensure_contract(cfg, run_dir)
    result = evaluate(
        cfg, run_dir, contract, num_prompts=args.num_prompts, max_new_tokens=args.max_new_tokens, tag=args.tag
    )
    equivalence = result["equivalence_check"]
    log(f"eval_dflash done: token_exact={equivalence['token_exact']}")
    for name, run in result["runs"].items():
        log(f"  {name}: AL={run['mean_al']:.3f} accept_rate={run['acceptance_rate']:.3f}")


if __name__ == "__main__":
    main()
