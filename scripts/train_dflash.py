#!/usr/bin/env python3
"""Stage 2b: train the DFlash drafter from staged features (target not loaded)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.config import apply_overrides, dump_effective_config, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.dflash.train import train_dflash
from vlm_dflash.utils import log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--steps", type=int, default=None, help="override total optimizer steps")
    parser.add_argument("--limit", type=int, default=None, help="cap training samples")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.skip_sanity:
        cfg.setdefault("train", {}).setdefault("overfit_sanity", {})["enabled"] = False
    run_dir = resolve_run_dir(cfg, args.run_dir)
    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    summary = train_dflash(cfg, run_dir, contract, steps=args.steps, limit=args.limit)
    log(f"train_dflash done: steps={summary['steps']} final_loss={summary['final_loss']}")


if __name__ == "__main__":
    main()
