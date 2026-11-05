"""Opt-in Qwen3-VL-8B DFlash spatial adapter activation; no custom kernels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_adapter import install

install()
