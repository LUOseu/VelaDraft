#!/usr/bin/env python3
"""Compatibility entry point for the native-vLLM paired E2E protocol.

Requires --target, --draft, --source and --out. See docs/benchmark.md.
Existing bridge exports with spatial MRoPE need an explicit pinned runtime
manifest; this command never removes spatial conditioning or trained modules.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vlm_dflash.eval.native import main

if __name__ == "__main__":
    raise SystemExit(main())
