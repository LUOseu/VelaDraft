"""Enterprise workload advisor for VelaDraft (platform extension).

This package is deliberately importable without torch/CUDA: validation,
offline reporting and recommendations run on CPU-only hosts. The HF backend
module imports torch lazily and declares a blocker when no device is present.

Wire format: ``veladraft-platform-v1`` (see ``schema.py``).
"""

from .schema import (
    COUNT_DEFINITION_VERSION,
    CORRECTNESS_STATUSES,
    EVIDENCE_STATES,
    PROJECT_SCOPE,
    SCHEMA_VERSION,
)

__all__ = [
    "COUNT_DEFINITION_VERSION",
    "CORRECTNESS_STATUSES",
    "EVIDENCE_STATES",
    "PROJECT_SCOPE",
    "SCHEMA_VERSION",
]
