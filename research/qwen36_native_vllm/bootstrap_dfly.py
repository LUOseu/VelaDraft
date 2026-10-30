"""Opt-in Qwen3.6 DFly adapter activation (spatial DFlash base + DFly modules)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfly_adapter import install

install()
