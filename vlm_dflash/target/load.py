"""Load the target model/processor via the family adapter from the frozen contract."""

from __future__ import annotations

import torch


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def require_single_device_target(cfg: dict, context: str) -> None:
    """Refuse spec-decode verification on a device-mapped target (not supported yet).

    The verify loop keeps logits, posteriors and the output buffer on one
    device; a sharded target returns them from the lm_head's GPU.
    """
    if (cfg.get("target") or {}).get("device_map") is not None:
        raise RuntimeError(
            f"{context}: target.device_map shards the target, which spec-decode verification does not "
            "support yet; it is for regen, capture and online training. Drop target.device_map for eval."
        )


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
    # ``target.device_map`` shards a target that does not fit next to the
    # drafter on one GPU across devices (accelerate); ``target.max_memory``
    # caps each device so the drafter keeps room on its own.
    device_map = target_cfg.get("device_map")
    if device_map is None and target_cfg.get("max_memory") is not None:
        raise ValueError("target.max_memory only applies with target.device_map; set both or neither.")
    load_kwargs: dict = {}
    if device_map is not None:
        if device is not None:
            raise ValueError("load_target got both an explicit device and target.device_map; pick one.")
        load_kwargs["device_map"] = device_map
        max_memory = target_cfg.get("max_memory")
        if max_memory is not None:
            load_kwargs["max_memory"] = {
                (int(key) if str(key).isdigit() else key): value for key, value in max_memory.items()
            }
    model = adapter.load_model(
        model_path,
        dtype=_dtype(target_cfg.get("dtype", "bfloat16")),
        attn_implementation=target_cfg.get("attn_implementation", "sdpa"),
        local_files_only=local_only,
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        **adapter.processor_kwargs(contract),
        local_files_only=local_only,
    )
    adapter.finalize_processor(processor, contract)
    if device_map is None:
        model.to(device or target_cfg.get("device", "cuda:0"))
    model.eval()
    model.requires_grad_(False)
    return model, processor
