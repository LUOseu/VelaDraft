"""Target-family adapter registry."""

from __future__ import annotations

from .base import AdapterInfo, VLMAdapter
from .qwen25_vl import Qwen25VLAdapter
from .qwen3_vl import Qwen3VLAdapter
from .smolvlm import SmolVLMAdapter

_ADAPTERS: tuple[VLMAdapter, ...] = (Qwen25VLAdapter(), Qwen3VLAdapter(), SmolVLMAdapter())


class UnknownTargetError(RuntimeError):
    """Raised when a target model family has no registered adapter."""


def supported_model_types() -> list[str]:
    return [model_type for adapter in _ADAPTERS for model_type in adapter.info.model_types]


def get_adapter(model_type: str) -> VLMAdapter:
    for adapter in _ADAPTERS:
        if model_type in adapter.info.model_types:
            return adapter
    raise UnknownTargetError(
        f"No adapter registered for target model_type {model_type!r}; "
        f"supported: {supported_model_types()}."
    )


def adapter_for_contract(contract: dict) -> VLMAdapter:
    return get_adapter(str(contract["target"]["model_type"]))


def adapter_for_model(model) -> VLMAdapter:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type is None:
        raise UnknownTargetError(f"Cannot detect target family: {type(model).__name__} has no config.model_type.")
    return get_adapter(str(model_type))


def adapter_from_config_path(model_path: str) -> VLMAdapter:
    """Detect the adapter from a local model directory's ``config.json``."""
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise UnknownTargetError(f"Target directory has no config.json: {model_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        model_type = json.load(fh).get("model_type")
    if not model_type:
        raise UnknownTargetError(f"{config_path} has no model_type field.")
    return get_adapter(str(model_type))


__all__ = [
    "AdapterInfo",
    "VLMAdapter",
    "Qwen25VLAdapter",
    "Qwen3VLAdapter",
    "SmolVLMAdapter",
    "UnknownTargetError",
    "adapter_for_contract",
    "adapter_for_model",
    "adapter_from_config_path",
    "get_adapter",
    "supported_model_types",
]
