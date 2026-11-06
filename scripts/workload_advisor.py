#!/usr/bin/env python3
"""Thin entry point for the enterprise workload advisor (see the package CLI)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.workload_advisor.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
