"""Drafter algorithm plugins (registry + implementations).

Usage::

    from vlm_dflash.algorithms import available_algorithms, get_algorithm

    algorithm = get_algorithm(cfg.get("algorithm"))   # defaults to "dflash"
"""

from .base import AlgorithmMetadata, DrafterAlgorithm
from .registry import (
    DEFAULT_ALGORITHM,
    UnknownAlgorithmError,
    available_algorithms,
    get_algorithm,
    register,
)

__all__ = [
    "AlgorithmMetadata",
    "DrafterAlgorithm",
    "DEFAULT_ALGORITHM",
    "UnknownAlgorithmError",
    "available_algorithms",
    "get_algorithm",
    "register",
]
