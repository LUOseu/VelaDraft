"""Opt-in Qwen3.6 DSpark/Domino head adapter activation (spatial DFlash base + anchor-slot heads)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_adapters import install

install()
