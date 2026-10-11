#!/usr/bin/env python3
"""Stage 1b: regenerate assistant answers with the target (greedy, token ids stored)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.config import apply_overrides, dump_effective_config, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.target.regen import run_regen
from vlm_dflash.utils import log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    stats = run_regen(cfg, run_dir, contract, limit=args.limit)
    log(f"generate_target_data done: kept={stats['num_kept']} skipped={stats['num_skipped']}")


if __name__ == "__main__":
    main()
