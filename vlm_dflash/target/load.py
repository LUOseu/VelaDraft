"""Load the Qwen2.5-VL target and its processor from the frozen contract."""

from __future__ import annotations

import torch


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def load_target(cfg: dict, contract: dict, device: str | None = None):
    """Return ``(model, processor)`` for the contract's target checkpoint.

    The processor is pinned to the contract's effective pixel bounds so regen,
    capture and eval all preprocess images identically.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    target_cfg = cfg.get("target", {})
    model_path = contract["target"]["model_path"]
    local_only = bool(target_cfg.get("local_files_only", True))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=_dtype(target_cfg.get("dtype", "bfloat16")),
        attn_implementation=target_cfg.get("attn_implementation", "sdpa"),
        local_files_only=local_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=contract["image"]["min_pixels"],
        max_pixels=contract["image"]["max_pixels"],
        local_files_only=local_only,
    )
    processor.tokenizer.padding_side = "left"
    model.to(device or target_cfg.get("device", "cuda:0"))
    model.eval()
    model.requires_grad_(False)
    return model, processor
