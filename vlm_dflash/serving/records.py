"""RunRecord assembly helpers shared by the correctness and HTTP harnesses."""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    INCLUDE_IN_PAPER,
    PROJECT_SCOPE,
    SCHEMA_VERSION,
    protocol_hash,
    sha256_file,
    validate_run_record,
)


def iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _hash_if_file(path: Path) -> Optional[str]:
    return sha256_file(path) if path.is_file() else None


def model_identity(model_path: str, precision: str = "bfloat16") -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    tokenizer_hash = _hash_if_file(root / "tokenizer.json")
    processor_hash = _hash_if_file(root / "preprocessor_config.json")
    template_hash = _hash_if_file(root / "chat_template.json")
    missing = []
    if not tokenizer_hash:
        missing.append("tokenizer.json not found; tokenizer identity incomplete")
    if not processor_hash:
        missing.append("preprocessor_config.json not found; processor identity incomplete")
    if not template_hash:
        missing.append("chat_template.json not found; template identity incomplete")
    return {
        "model": str(root),
        "revision": _hash_if_file(root / "model.safetensors.index.json") or _hash_if_file(config_path),
        "tokenizer": {"path": str(root / "tokenizer.json"), "sha256": tokenizer_hash, "class": "Qwen2TokenizerFast"},
        "processor": {
            "path": str(root / "preprocessor_config.json"),
            "sha256": processor_hash,
            "class": config.get("model_type", "qwen2_5_vl"),
        },
        "template": {"path": str(root / "chat_template.json"), "sha256": template_hash, "mode": "tokenizer_chat_template"},
        "precision": precision,
        "config_sha256": _hash_if_file(config_path),
        "architecture": (config.get("architectures") or [None])[0],
        "missing_reasons": missing,
    }


def ngram_draft_identity(vllm_version: str, target: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "cpu_ngram",
        "model": "[ngram]",
        "revision": vllm_version,
        "tokenizer": target.get("tokenizer"),
        "precision": "not_applicable",
        "params": dict(params),
        "missing_reasons": [
            "N-Gram has no draft model weights; it uses the target tokenizer and token-id suffix lookup",
        ],
    }


def backend_identity(
    *,
    vllm_version: str,
    code_commit: str,
    devices: list[dict[str, Any]],
    tensor_parallel_size: int = 1,
    attention_backend: Optional[str] = None,
    cudagraph_mode: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "backend": "vllm",
        "version": vllm_version,
        "code_commit": code_commit,
        "device_topology": {
            "gpu_count": len(devices),
            "devices": devices,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": 1,
        },
        "attention_backend": attention_backend,
        "cudagraph_mode": cudagraph_mode,
        "missing_reasons": [],
    }


def resource_usage(
    *,
    wall_s: float,
    gpu_count: int,
    devices: list[dict[str, Any]],
    peak_memory_mib: dict[str, int],
    gpu_hours: Optional[float] = None,
) -> dict[str, Any]:
    if gpu_hours is None:
        gpu_hours = max(0.0, float(wall_s)) * max(1, int(gpu_count)) / 3600.0
    return {
        "gpu_hours": round(float(gpu_hours), 6),
        "gpu_count": int(gpu_count),
        "devices": devices,
        "peak_memory_mib": peak_memory_mib,
        "wall_s": float(wall_s),
    }


def make_run_record(
    *,
    run_id: str,
    request_id: str,
    group_id: str,
    target_identity: dict[str, Any],
    draft_identity: Optional[dict[str, Any]],
    backend_identity: dict[str, Any],
    protocol_payload: dict[str, Any],
    correctness_status: str,
    measurement_scope: dict[str, Any],
    raw_counts: dict[str, Any],
    latency: dict[str, Any],
    resource_usage_value: dict[str, Any],
    missing_reasons: Optional[dict[str, str]] = None,
    **extras: Any,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "project_scope": PROJECT_SCOPE,
        "include_in_paper": INCLUDE_IN_PAPER,
        "run_id": run_id,
        "request_id": request_id,
        "group_id": group_id,
        "target_identity": target_identity,
        "draft_identity": draft_identity,
        "backend_identity": backend_identity,
        "protocol_hash": protocol_hash(protocol_payload),
        "correctness_status": correctness_status,
        "measurement_scope": measurement_scope,
        "raw_counts": raw_counts,
        "latency": latency,
        "resource_usage": resource_usage_value,
        "missing_reasons": missing_reasons or {},
    }
    record.update(extras)
    return validate_run_record(record)


def protocol_payload(
    *,
    model_path: str,
    mode: str,
    spec_config: Optional[dict[str, Any]],
    sampling: dict[str, Any],
    pixel_bounds: dict[str, int],
    max_model_len: int,
    prefix_caching: bool,
    backend_version: str,
    timing_boundary: str,
    stream: bool = False,
    async_scheduling: Optional[bool] = None,
) -> dict[str, Any]:
    return {
        "model": str(Path(model_path).resolve()),
        "mode": mode,
        "speculative_config": spec_config,
        "sampling": sampling,
        "pixel_bounds": pixel_bounds,
        "max_model_len": max_model_len,
        "prefix_caching": prefix_caching,
        "backend": {"name": "vllm", "version": backend_version},
        "stream": stream,
        "async_scheduling": async_scheduling,
        "timing_boundary": timing_boundary,
    }
