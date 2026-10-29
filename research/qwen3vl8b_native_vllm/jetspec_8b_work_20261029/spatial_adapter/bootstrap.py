"""Opt-in reviewed 8B DFlash spatial adapter; no custom kernels."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_spatial_graph_adapter_v3 import install
install()
