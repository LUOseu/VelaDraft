#!/usr/bin/env python3
"""Stage 2a: capture fused target hidden states into staged feature shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.config import apply_overrides, dump_effective_config, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.target.capture import run_capture
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
    manifest = run_capture(cfg, run_dir, contract, limit=args.limit)
    log(f"capture_features done: samples={manifest['num_samples']} truncated={manifest['num_truncated']}")


if __name__ == "__main__":
    main()
