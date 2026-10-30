"""Opt-in Qwen3.6 DFlash2 adapter activation (spatial DFlash base + convolutions + selector)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash2_adapter import install

install()
