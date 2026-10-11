#!/usr/bin/env python3
"""Stage 1a: build the ShareGPT training/validation data from a public source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.config import apply_overrides, dump_effective_config, load_config, resolve_run_dir
from vlm_dflash.data.prepare import build_dataset
from vlm_dflash.utils import ensure_dir, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="override dataset.slice")
    parser.add_argument("--set", action="append", default=[], help="dotted override, e.g. dataset.seed=7")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    ensure_dir(run_dir)
    dump_effective_config(cfg, run_dir / "config.json")
    manifest = build_dataset(cfg, run_dir / "data", limit=args.limit)
    log(f"prepare_data done: {run_dir} train={manifest['num_train']} val={manifest['num_val']}")


if __name__ == "__main__":
    main()
