"""Opt-in Qwen3.6 DFlare adapter activation (spatial DFlash base + DFlare context)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflare_adapter import install

install()
