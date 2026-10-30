"""Helpers shared by the CPU-side Qwen3.6 evaluation scripts (not pinned by any manifest)."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Reporting order of the seven methods.
METHODS = ("dflash", "dspark", "domino", "jetspec", "dflash2", "dflare", "dfly")


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
