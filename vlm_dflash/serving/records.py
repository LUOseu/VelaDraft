"""RunRecord assembly helpers shared by the correctness and HTTP harnesses."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    INCLUDE_IN_PAPER,
    PROJECT_SCOPE,
    SCHEMA_VERSION,
    canonical_json,
    protocol_hash,
    sha256_file,
    sha256_text,
    validate_run_record,
)


def iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _hash_if_file(path: Path) -> Optional[str]:
    return sha256_file(path) if path.is_file() else None


_SNAPSHOT_RE = re.compile(r"[/\\]snapshots[/\\]([0-9a-fA-F]{7,64})(?:[/\\]|$)")


def snapshot_identity(model_path: str | Path) -> dict[str, Any]:
    """Hash the actual weight files, not only the index/config metadata."""

    root = Path(model_path)
    match = _SNAPSHOT_RE.search(str(root.resolve()))
    weight_suffixes = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
    files: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in weight_suffixes:
                continue
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    content_sha256 = hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest() if files else None
    reasons = []
    if not root.is_dir():
        reasons.append("model path is not a directory; weight content identity unavailable")
    elif not files:
        reasons.append("no weight files found; weight content identity unavailable")
    return {
        "snapshot_commit": match.group(1) if match else None,
        "weight_files": files,
        "content_sha256": content_sha256,
        "content_verified": bool(files),
        "missing_reasons": reasons,
    }


def _template_identity(root: Path) -> dict[str, Any]:
    standalone = root / "chat_template.json"
    tokenizer_config = root / "tokenizer_config.json"
    if standalone.is_file():
        return {
            "path": str(standalone),
            "sha256": sha256_file(standalone),
            "mode": "standalone_chat_template_json",
            "effective_source": "chat_template.json",
        }
    if tokenizer_config.is_file():
        try:
            config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        template = config.get("chat_template")
        if isinstance(template, str) and template:
            return {
                "path": str(tokenizer_config),
                "sha256": sha256_text(template),
                "mode": "tokenizer_config.chat_template",
                "effective_source": "tokenizer_config.json#chat_template",
            }
    return {
        "path": str(standalone),
        "sha256": None,
        "mode": "unavailable",
        "effective_source": None,
    }


def model_identity(model_path: str, precision: str = "bfloat16") -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    tokenizer_hash = _hash_if_file(root / "tokenizer.json")
    processor_hash = _hash_if_file(root / "preprocessor_config.json")
    template = _template_identity(root)
    snapshot = snapshot_identity(root)
    missing = []
    if not tokenizer_hash:
        missing.append("tokenizer.json not found; tokenizer identity incomplete")
    if not processor_hash:
        missing.append("preprocessor_config.json not found; processor identity incomplete")
    if not template.get("sha256"):
        missing.append("no effective chat template found; template identity incomplete")
    missing.extend(snapshot["missing_reasons"])
    return {
        "model": str(root),
        "revision": _hash_if_file(root / "model.safetensors.index.json") or _hash_if_file(config_path),
        "weight_content_sha256": snapshot["content_sha256"],
        "snapshot": snapshot,
        "tokenizer": {"path": str(root / "tokenizer.json"), "sha256": tokenizer_hash, "class": "Qwen2TokenizerFast"},
        "processor": {
            "path": str(root / "preprocessor_config.json"),
            "sha256": processor_hash,
            "class": config.get("model_type", "qwen2_5_vl"),
        },
        "template": template,
        "precision": precision,
        "config_sha256": _hash_if_file(config_path),
        "architecture": (config.get("architectures") or [None])[0],
        "identity_complete": not missing,
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
    labels_source: str = "cli_declared_unverified",
    applied_evidence: bool = False,
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
        "labels_source": labels_source,
        "applied_evidence": bool(applied_evidence),
        "missing_reasons": [],
    }


def resource_usage(
    *,
    wall_s: float,
    gpu_count: int,
    devices: list[dict[str, Any]],
    peak_memory_mib: dict[str, int],
    gpu_hours: Optional[float] = None,
    scope: str = "allocation",
    additive: bool = True,
    reason: Optional[str] = None,
    peak_scope: Optional[str] = None,
) -> dict[str, Any]:
    if gpu_hours is None:
        if scope == "allocation":
            gpu_hours = max(0.0, float(wall_s)) * max(1, int(gpu_count)) / 3600.0
        elif not reason:
            raise ValueError("non-allocation resource usage with null gpu_hours needs a reason")
    payload = {
        "gpu_hours": round(float(gpu_hours), 6) if gpu_hours is not None else None,
        "gpu_hours_scope": scope,
        "additive": bool(additive),
        "gpu_count": int(gpu_count),
        "devices": devices,
        "peak_memory_mib": peak_memory_mib,
        "wall_s": float(wall_s),
    }
    if reason:
        payload["gpu_hours_reason"] = reason
    if peak_scope:
        payload["peak_memory_scope"] = peak_scope
    return payload


def exposure_usage(
    *,
    wall_s: float,
    devices: list[dict[str, Any]],
    peak_memory_mib: dict[str, int],
    reason: str,
    scope: str = "request_exposure",
) -> dict[str, Any]:
    """Request/window occupancy that must not be summed into GPU-hours."""

    return resource_usage(
        wall_s=wall_s,
        gpu_count=1,
        devices=devices,
        peak_memory_mib=peak_memory_mib,
        gpu_hours=None,
        scope=scope,
        additive=False,
        reason=reason,
        peak_scope="process_sampler_not_%s" % ("request" if scope == "request_exposure" else "window"),
    )


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
