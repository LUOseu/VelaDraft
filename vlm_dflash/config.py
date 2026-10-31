"""YAML config loading with dotted-path lookup and ``--set`` overrides."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _replace_env(match: re.Match) -> str:
    name, default = match.group(1), match.group(2)
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    raise ValueError(f"Config references environment variable {name} with no default and it is unset.")


def expand_env_text(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references in a single string.

    Same syntax as YAML configs (unlike ``os.path.expandvars``, which has no
    default-value form); used for manifest prompt paths.
    """
    return _ENV_PATTERN.sub(_replace_env, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_text(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse into a mapping.")
    return _expand_env(cfg)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def validate_runtime_config(cfg: dict, *, require_devices: bool = False) -> None:
    """Validate device/backend knobs shared by every training entry point.

    Config parsing stays probe-free for unit tests; the real ``train`` command
    calls this again with ``require_devices=True`` before creating a run dir.
    An explicit NPU request is always checked because silently substituting CPU
    or CUDA would invalidate the run's evidence.
    """
    from .dflash.factory import model_attention_implementation
    from .runtime import device_type, ensure_runtime_device, npu_available

    train = cfg.get("train") or {}
    target = cfg.get("target") or {}
    drafter = cfg.get("drafter") or {}
    ce_chunk_size = int(drafter.get("ce_chunk_size", 0))
    ce_backward = str(drafter.get("ce_backward", "checkpoint"))
    if ce_backward not in {"checkpoint", "precompute"}:
        raise ValueError("drafter.ce_backward must be 'checkpoint' or 'precompute'.")
    if ce_backward == "precompute" and not ce_chunk_size:
        raise ValueError("drafter.ce_backward=precompute requires drafter.ce_chunk_size > 0.")
    if ce_chunk_size < 0:
        raise ValueError("drafter.ce_chunk_size must be nonnegative (0 keeps dense scoring).")
    if ce_chunk_size:
        from .algorithms.registry import algorithm_name
        from .objectives import resolve_recipe

        name = algorithm_name(cfg.get("algorithm"))
        recipe = resolve_recipe(cfg, name)
        if name not in {"dflash", "dflare", "dfly"} or (recipe is not None and recipe.name != "controlled_ce"):
            raise ValueError("drafter.ce_chunk_size supports only controlled CE on DFlash/DFlare/DFly.")
    if bool(drafter.get("teacher_projection_dedup", False)):
        from .algorithms.registry import algorithm_name
        from .objectives import resolve_recipe

        name = algorithm_name(cfg.get("algorithm"))
        recipe = resolve_recipe(cfg, name)
        if name not in {"dflash", "dflare", "dfly", "dspark"} or recipe is None or not recipe.needs_teacher:
            raise ValueError(
                "drafter.teacher_projection_dedup requires upstream_two_stage on "
                "DFlash/DFlare/DFly or upstream_dspark on DSpark."
            )
    train_device = train.get("device", "cuda:0")
    target_device = target.get("device")
    for label, value in (("train.device", train_device), ("target.device", target_device)):
        if value is None:
            continue
        kind = device_type(value)
        if kind not in ("cpu", "cuda", "npu"):
            raise ValueError(
                f"{label}={value!r} uses unsupported device type {kind!r}; supported types are cpu, cuda and npu."
            )
        if kind == "npu" and not npu_available():
            raise ValueError(
                f"{label}={value!r} requests an Ascend NPU, but torch_npu/CANN is unavailable. "
                "Install the matching TorchNPU wheel and source the CANN environment; no silent "
                "CPU/CUDA fallback is performed."
            )
        if require_devices:
            ensure_runtime_device(value, context=label)

    # Raises for an unknown backend before any expensive stage starts.
    model_attention_implementation(drafter.get("attention_backend", "flex_attention"))
    requested = str(drafter.get("attention_backend", "flex_attention"))
    if requested == "flex_attention" and device_type(train_device) == "npu":
        raise ValueError(
            "drafter.attention_backend=flex_attention is not enabled on Ascend NPU; "
            "use sdpa_chunked (the reference NPU path) or opt into npu_fusion explicitly."
        )
    if requested == "npu_fusion" and device_type(train_device) != "npu":
        raise ValueError(
            "drafter.attention_backend=npu_fusion requires train.device to be an NPU; "
            "use sdpa_chunked on CPU/CUDA."
        )
    if requested in ("sdpa_chunked", "npu_fusion"):
        chunk = int(drafter.get("attention_chunk_size", 128))
        if chunk <= 0:
            raise ValueError("drafter.attention_chunk_size must be a positive integer.")
    if bool(train.get("optimizer_fused", False)) and device_type(train_device) != "cuda":
        raise ValueError("train.optimizer_fused=true is supported only on CUDA.")
    if bool(train.get("pin_memory", False)) and device_type(train_device) != "cuda":
        raise ValueError("train.pin_memory=true requires a CUDA training device.")
    if bool(train.get("non_blocking", False)) and device_type(train_device) not in ("cuda", "npu"):
        raise ValueError("train.non_blocking=true requires a CUDA or NPU training device.")


def parse_scalar(text: str) -> Any:
    """Parse a ``--set key=value`` value as YAML scalar (falls back to string)."""
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        set_dotted(cfg, key, parse_scalar(value))
    return cfg


def resolve_run_dir(cfg: dict, override: str | None = None) -> Path:
    """Run directory: CLI override > ``run.dir`` (relative to CWD)."""
    if override:
        return Path(override).resolve()
    run_dir = get(cfg, "run.dir")
    if not run_dir:
        raise ValueError("Config is missing run.dir and no --run-dir was given.")
    return (Path.cwd() / run_dir).resolve()


def dump_effective_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2, default=str)
