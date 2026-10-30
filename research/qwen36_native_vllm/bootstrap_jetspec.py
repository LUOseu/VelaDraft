"""Opt-in Qwen3.6 local-block JetSpec activation (spatial DFlash base + causal draft queries)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jetspec_adapter import install

install()
