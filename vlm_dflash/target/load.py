"""Load the target model/processor via the family adapter from the frozen contract."""

from __future__ import annotations

import torch

from .. import runtime


def _device_map_backend_hint(device_map) -> str | None:
    """Infer the accelerator backend from an explicit module-to-device map."""
    if not isinstance(device_map, dict):
        if isinstance(device_map, str):
            text = device_map.strip()
            if text == "cpu":
                return "cpu"
            try:
                kind = runtime.device_type(text)
            except runtime.DeviceResolutionError:
                return None
            return kind if kind in ("cpu", "cuda", "npu") else None
        return None
    kinds: set[str] = set()
    for raw in device_map.values():
        if isinstance(raw, int):
            kinds.add("cuda")
        elif isinstance(raw, torch.device):
            kinds.add(raw.type)
        elif isinstance(raw, str):
            text = raw.strip()
            if text.isdigit():
                kinds.add("cuda")
            elif text in ("cpu", "disk"):
                kinds.add(text)
            else:
                try:
                    kinds.add(runtime.device_type(text))
                except runtime.DeviceResolutionError:
                    continue
    accelerators = kinds & {"cuda", "npu"}
    if len(accelerators) > 1:
        raise ValueError(
            "target.device_map mixes CUDA and NPU placements; one training run must use a single "
            f"accelerator backend, got {sorted(accelerators)}."
        )
    if accelerators:
        return next(iter(accelerators))
    if "cpu" in kinds:
        return "cpu"
    return None


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
    configured_device = device or target_cfg.get("device") or (cfg.get("train") or {}).get("device") or "cuda:0"
    backend_hint = runtime.device_type(configured_device)
    model_path = contract["target"]["model_path"]
    local_only = bool(target_cfg.get("local_files_only", True))
    # ``target.device_map`` shards a target that does not fit next to the
    # drafter on one GPU across devices (accelerate); ``target.max_memory``
    # caps each device so the drafter keeps room on its own.
    device_map = target_cfg.get("device_map")
    if device_map is None and target_cfg.get("max_memory") is not None:
        raise ValueError("target.max_memory only applies with target.device_map; set both or neither.")
    if device_map is not None:
        backend_hint = _device_map_backend_hint(device_map) or backend_hint
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
        backend_hint=backend_hint,
        **load_kwargs,
    )
    if device_map is not None:
        # accelerate silently offloads what does not fit on the GPUs (or under
        # max_memory) to cpu or disk (meta tensors); a frozen teacher that is
        # mapped onto GPUs must sit there whole. An all-cpu map is left alone.
        devices = {str(param.device) for param in model.parameters()}
        offloaded = sorted(name for name in devices if name == "meta" or (name == "cpu" and devices != {"cpu"}))
        if offloaded:
            mapped = getattr(model, "hf_device_map", None) or {}
            modules = [name for name, where in mapped.items() if where in ("cpu", "disk")]
            shown = (", ".join(modules[:4]) + (" ..." if len(modules) > 4 else "")) if modules else "unknown"
            raise ValueError(
                f"target.device_map left target parameters on {offloaded} (offloaded modules: {shown}); give the "
                "mapped GPUs room for the whole target (raise target.max_memory or map more devices)."
            )
    processor = AutoProcessor.from_pretrained(
        model_path,
        **adapter.processor_kwargs(contract),
        local_files_only=local_only,
    )
    adapter.finalize_processor(processor, contract)
    if device_map is None:
        # Training/regen resolve this device again with backend availability
        # checks. Here we allow a CUDA device object without probing CUDA so the
        # model contract can be exercised on CPU CI with lightweight fakes; an
        # explicit NPU request still imports torch_npu and fails loudly when it
        # is unavailable.
        require_available = backend_hint == "npu"
        resolved = runtime.resolve_device(
            configured_device, context="target.device", require_available=require_available
        )
        model.to(str(resolved))
    else:
        setattr(model, "_vlm_dflash_device_map_backend", backend_hint)
    model.eval()
    model.requires_grad_(False)
    return model, processor
