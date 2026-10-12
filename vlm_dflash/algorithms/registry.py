"""Algorithm registry: name -> :class:`DrafterAlgorithm` instance.

Default registrations are loaded lazily so ``import vlm_dflash.algorithms``
stays cheap; ``get_algorithm(None)`` resolves the default (``dflash``), which
is also what runs without an explicit ``algorithm:`` config key produce.
"""

from __future__ import annotations

from .base import AlgorithmMetadata, DrafterAlgorithm

_REGISTRY: dict[str, DrafterAlgorithm] = {}
_DEFAULTS_LOADED = False

DEFAULT_ALGORITHM = "dflash"


class UnknownAlgorithmError(KeyError):
    def __init__(self, name: str, available: list[str]):
        super().__init__(name)
        self.name = name
        self.available = available

    def __str__(self) -> str:
        return (
            f"Unknown drafter algorithm {self.name!r}. Registered algorithms: "
            f"{', '.join(self.available) or '(none)'}. "
            "Algorithms are added as plugins under vlm_dflash/algorithms/ "
            "(see docs/adding_an_algorithm.md)."
        )


def register(algorithm: DrafterAlgorithm) -> DrafterAlgorithm:
    """Register (or replace) an algorithm by name. Import-time registration."""
    if not isinstance(algorithm, DrafterAlgorithm):
        raise TypeError(f"register() expects a DrafterAlgorithm, got {type(algorithm).__name__}")
    if not algorithm.name:
        raise ValueError("A drafter algorithm must define a non-empty name.")
    _REGISTRY[algorithm.name] = algorithm
    return algorithm


def _load_defaults() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    _DEFAULTS_LOADED = True
    from . import dflash as _dflash  # noqa: F401  (self-registers DFlashAlgorithm)
    from . import dspark as _dspark  # noqa: F401  (self-registers DSparkAlgorithm)
    from . import domino as _domino  # noqa: F401  (self-registers DominoAlgorithm)


def available_algorithms() -> list[str]:
    _load_defaults()
    return sorted(_REGISTRY)


def algorithm_name(algorithm_cfg) -> str:
    """Resolve an algorithm name from a config value.

    Accepts the plain string form (``algorithm: dflash``) and the namespaced
    mapping form (``algorithm: {name: dspark, dspark: {...}}``).
    """
    if isinstance(algorithm_cfg, dict):
        raw = algorithm_cfg.get("name")
    else:
        raw = algorithm_cfg
    key = str(raw).strip() if raw else ""
    return key or DEFAULT_ALGORITHM


def get_algorithm(name: str | dict | None = None) -> DrafterAlgorithm:
    _load_defaults()
    key = algorithm_name(name)
    if key not in _REGISTRY:
        raise UnknownAlgorithmError(key, available_algorithms())
    return _REGISTRY[key]


__all__ = [
    "AlgorithmMetadata",
    "DrafterAlgorithm",
    "UnknownAlgorithmError",
    "algorithm_name",
    "available_algorithms",
    "get_algorithm",
    "register",
    "DEFAULT_ALGORITHM",
]
