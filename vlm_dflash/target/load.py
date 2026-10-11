"""Load the target model/processor via the family adapter from the frozen contract."""

from __future__ import annotations

import torch


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def load_target(cfg: dict, contract: dict, device: str | None = None):
    """Return ``(model, processor)`` for the contract's target checkpoint.

    The adapter is selected from the contract's resolved ``model_type``; the
    processor is pinned to the contract's effective pixel bounds so regen,
    capture and eval all preprocess images identically.
    """
    from transformers import AutoProcessor

    from ..adapters import adapter_for_contract

    adapter = adapter_for_contract(contract)
    target_cfg = cfg.get("target", {})
    model_path = contract["target"]["model_path"]
    local_only = bool(target_cfg.get("local_files_only", True))
    model = adapter.load_model(
        model_path,
        dtype=_dtype(target_cfg.get("dtype", "bfloat16")),
        attn_implementation=target_cfg.get("attn_implementation", "sdpa"),
        local_files_only=local_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        **adapter.processor_kwargs(contract),
        local_files_only=local_only,
    )
    adapter.finalize_processor(processor, contract)
    model.to(device or target_cfg.get("device", "cuda:0"))
    model.eval()
    model.requires_grad_(False)
    return model, processor
