"""RunRecord construction, validation and legacy aggregate import.

Records exchange as JSONL in the ``veladraft-platform-v1`` wire format. The
consumer layer is torch-free so offline reports never need CUDA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    COUNT_DEFINITION_VERSION,
    INCLUDE_IN_PAPER,
    PROJECT_SCOPE,
    SCHEMA_VERSION,
    canonical_json,
    is_sha256,
    sha256_json,
    validate_record,
)

# Old harness format is imported as aggregate-only evidence.
LEGACY_BENCHMARK_FORMAT = "vlm-dflash-benchmark-v1"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,160}$")
_SAFE_REASON = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,120}$")
_HASH_KEYS = {
    "sha256",
    "hash",
    "weights_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "template_sha256",
    "input_sha256",
    "reference_hash",
    "candidate_hash",
}


def _reason(record: dict, key: str, value: str) -> None:
    record.setdefault("missing_reasons", {})[key] = value


def build_raw_counts(
    *,
    emitted_tokens: int | None,
    proposed_draft_tokens: int | None = None,
    accepted_draft_tokens: int | None = None,
    verify_rounds: int | None = None,
    target_forward_calls: int | None = None,
    draft_forward_calls: int | None = None,
    method: str = "sd",
) -> dict:
    counts = {
        "count_definition_version": COUNT_DEFINITION_VERSION,
        "emitted_tokens": emitted_tokens,
        "proposed_draft_tokens": proposed_draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "verify_rounds": verify_rounds,
        "target_forward_calls": target_forward_calls,
        "draft_forward_calls": draft_forward_calls,
        "acceptance_length_draft": None,
        "acceptance_length_progress": None,
    }
    if method == "ar":
        counts["proposed_draft_tokens"] = None
        counts["accepted_draft_tokens"] = None
        counts["verify_rounds"] = None
    if verify_rounds and verify_rounds > 0:
        if accepted_draft_tokens is not None:
            counts["acceptance_length_draft"] = accepted_draft_tokens / verify_rounds
        if emitted_tokens is not None:
            counts["acceptance_length_progress"] = emitted_tokens / verify_rounds
    return counts


def build_latency(
    *,
    e2e_s: float | None,
    ttft_s: float | None = None,
    itl_s: dict | None = None,
    tpot_s: float | None = None,
    chunk_gap_s: dict | None = None,
    client_chunk_tpot_proxy_s: float | None = None,
    n_samples: int = 1,
    ttft_reason: str = "no_streaming",
    itl_reason: str = "no_token_visibility",
) -> tuple[dict, dict]:
    latency = {
        "e2e_s": e2e_s,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "itl_s": itl_s,
        "chunk_gap_s": chunk_gap_s,
        "client_chunk_tpot_proxy_s": client_chunk_tpot_proxy_s,
        "n_samples": n_samples,
    }
    reasons: dict[str, str] = {}
    if e2e_s is None:
        reasons["latency.e2e_s"] = "not_measured"
    if ttft_s is None:
        reasons["latency.ttft_s"] = ttft_reason
    if itl_s is None:
        reasons["latency.itl_s"] = itl_reason
    return latency, reasons


def build_resource_usage(
    *,
    all_allocated_gpu_hours: float | None,
    measured_run_gpu_hours: float | None,
    devices: list[dict] | None = None,
    allocation_intervals: list[dict] | None = None,
    gpu_memory_fit: bool | None = None,
    note: str | None = None,
) -> dict:
    return {
        "all_allocated_gpu_hours": all_allocated_gpu_hours,
        "measured_run_gpu_hours": measured_run_gpu_hours,
        "devices": devices or [],
        "allocation_intervals": allocation_intervals or [],
        "gpu_memory_fit": gpu_memory_fit,
        "note": note,
    }


def make_record(
    *,
    run_id: str,
    request_id: str,
    group_id: str,
    task: str,
    split: str,
    method: str,
    candidate_id: str,
    session_id: str,
    repeat_index: int,
    execution_order: int,
    target_identity: dict,
    draft_identity: dict | None,
    backend_identity: dict,
    protocol_hash: str,
    correctness_status: str,
    measurement_scope: str,
    raw_counts: dict,
    latency: dict,
    resource_usage: dict,
    protocol_scope_hash: str | None = None,
    correctness: dict | None = None,
    quality: dict | None = None,
    output_tokens_R: int | None = None,
    visual_tokens: dict | None = None,
    acceptance_policy: str | None = None,
    requested_method: str | None = None,
    effective_method: str | None = None,
    fallback: dict | None = None,
    error: dict | None = None,
    artifacts: dict | None = None,
    synthetic: bool = False,
    missing_reasons: dict | None = None,
    measurement: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble a record and attach the reasons required by the protocol."""
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project_scope": PROJECT_SCOPE,
        "include_in_paper": INCLUDE_IN_PAPER,
        "run_id": run_id,
        "request_id": request_id,
        "group_id": group_id,
        "task": task,
        "split": split,
        "session_id": session_id,
        "repeat_index": repeat_index,
        "execution_order": execution_order,
        "method": method,
        "candidate_id": candidate_id,
        "requested_method": requested_method or method,
        "effective_method": effective_method or method,
        "fallback": fallback,
        "synthetic": synthetic,
        "target_identity": target_identity,
        "draft_identity": draft_identity,
        "backend_identity": backend_identity,
        "protocol_hash": protocol_hash,
        "protocol_scope_hash": protocol_scope_hash,
        "correctness_status": correctness_status,
        "acceptance_policy": acceptance_policy
        or ("strict_greedy_token_exact" if correctness_status == "verified_greedy_exact" else "not_applicable"),
        "correctness": correctness,
        "quality": quality,
        "measurement_scope": measurement_scope,
        "measurement": measurement
        or {
            "timing_boundary": "request_ready_to_committed_output",
            "includes": ["preprocess", "encode", "h2d", "generate"],
            "excludes": ["model_load", "media_disk_read"],
        },
        "raw_counts": raw_counts,
        "latency": latency,
        "resource_usage": resource_usage,
        "output_tokens_R": output_tokens_R,
        "visual_tokens": visual_tokens or {"target_V": None, "draft_V": None},
        "error": error,
        "artifacts": artifacts or {},
        "missing_reasons": dict(missing_reasons or {}),
    }
    if method == "ar":
        _reason(record, "draft_identity", "not_applicable_ar")
        _reason(record, "raw_counts.accepted_draft_tokens", "not_applicable_ar")
        _reason(record, "raw_counts.verify_rounds", "not_applicable_ar")
    if measurement_scope == "aggregate_only":
        _reason(record, "latency.e2e_s", "aggregate_only_no_per_sample")
        _reason(record, "resource_usage.all_allocated_gpu_hours", "aggregate_only_no_ledger")
    if quality and quality.get("evaluated") is False:
        _reason(record, "quality", quality.get("reason", "quality_not_evaluated"))
    if extra:
        for key, value in extra.items():
            record.setdefault(key, value)
    return record


def _safe_token(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("/", "\\", "~")) or "\\" in value or ".." in value or "//" in value:
        return None
    return value if _SAFE_TOKEN.fullmatch(value) else None


def _share_id(value: Any) -> str | None:
    """Preserve public token IDs; pseudonymize private paths/free text.

    Use the same mapping everywhere so AR/candidate joins and references in
    the summary and decision remain intact without exporting local paths.
    """
    if not isinstance(value, str) or not value:
        return None
    return _safe_token(value) or f"redacted-{sha256_json(value)}"


def _safe_hash(value: Any) -> str | None:
    return value.lower() if is_sha256(value) else None


def _safe_reason(value: Any) -> str:
    if isinstance(value, str) and _SAFE_REASON.fullmatch(value):
        return value
    return "redacted"


def _safe_scalar(value: Any, *, allow_string: bool = False) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_token(value) if allow_string else None


def _safe_identity(identity: Any) -> dict:
    if not isinstance(identity, dict):
        return {}
    allowed = {
        "source",
        "kind",
        "model_id",
        "revision",
        "dtype",
        "structure",
        "algorithm",
        "recipe",
        "name",
        "version",
        "code_commit",
        "architecture",
        "weights_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "processor_sha256",
        "template_sha256",
        "model_sha256",
    }
    result = {}
    for key, value in identity.items():
        if key not in allowed:
            continue
        if key.endswith("_sha256") or key.endswith("_hash"):
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                result[key] = value
        elif isinstance(value, str):
            token = _safe_token(value)
            if token is not None:
                result[key] = token
        elif isinstance(value, (int, float, bool)):
            result[key] = value
    topology = identity.get("device_topology")
    if isinstance(topology, dict):
        devices = []
        for device in topology.get("devices") or []:
            if not isinstance(device, dict):
                continue
            uuid = _safe_token(device.get("uuid"))
            if uuid:
                devices.append({"uuid": uuid})
        clean_topology = {}
        if isinstance(topology.get("gpu_count"), int):
            clean_topology["gpu_count"] = topology["gpu_count"]
        if devices:
            clean_topology["devices"] = devices
        if clean_topology:
            result["device_topology"] = clean_topology
    return result


def _safe_quality(quality: Any) -> dict | None:
    if not isinstance(quality, dict):
        return None
    result = {}
    for key in ("evaluated", "score", "threshold", "json_schema_valid"):
        if key in quality and (quality[key] is None or isinstance(quality[key], (bool, int, float))):
            result[key] = quality[key]
    status = _safe_token(quality.get("status"))
    if status:
        result["status"] = status
    scorer = quality.get("scorer")
    if isinstance(scorer, dict):
        clean_scorer = {}
        for key in ("name", "version"):
            token = _safe_token(scorer.get(key))
            if token:
                clean_scorer[key] = token
        if clean_scorer:
            result["scorer"] = clean_scorer
    coverage = quality.get("coverage")
    if isinstance(coverage, dict):
        clean_coverage = {}
        for axis in ("requests", "groups", "tasks"):
            values = coverage.get(axis)
            if isinstance(values, dict):
                clean_coverage[axis] = {
                    key: value
                    for key, value in values.items()
                    if key in {"evaluated", "denominator"} and isinstance(value, int)
                }
        if clean_coverage:
            result["coverage"] = clean_coverage
    return result


def _safe_correctness(correctness: Any) -> dict | None:
    if not isinstance(correctness, dict):
        return None
    numeric = {
        "n_requests",
        "compared_tokens",
        "failures",
        "failed_records",
        "denominator_records",
        "denominator_requests",
        "exact_records",
        "exact_pairs",
        "non_exact_pairs",
        "n_records",
        "n_paired_requests",
        "n_paired_groups",
        "n_paired_tasks",
        "run_error_count",
        "run_error_rate",
    }
    result = {key: correctness[key] for key in numeric if isinstance(correctness.get(key), (int, float, bool))}
    for key in ("input_sha256", "reference_hash", "candidate_hash"):
        value = correctness.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            result[key] = value
    for key in ("failed_request_ids", "run_error_request_ids"):
        if isinstance(correctness.get(key), list):
            result[key] = [token for token in (_share_id(value) for value in correctness[key]) if token]
    for key in ("status_counts",):
        if isinstance(correctness.get(key), dict):
            result[key] = {
                _safe_token(status) or "unknown": count
                for status, count in correctness[key].items()
                if isinstance(count, int)
            }
    paired = correctness.get("paired_coverage")
    if isinstance(paired, dict):
        result["paired_coverage"] = {key: value for key, value in paired.items() if isinstance(value, int)}
    denominators = correctness.get("denominators")
    if isinstance(denominators, dict):
        result["denominators"] = {key: value for key, value in denominators.items() if isinstance(value, int)}
    if isinstance(correctness.get("gate"), bool):
        result["gate"] = correctness["gate"]
    return result


def _safe_latency(latency: Any) -> dict:
    if not isinstance(latency, dict):
        return {}
    result = {}
    for key, value in latency.items():
        if key in {"itl_s", "chunk_gap_s"} and isinstance(value, dict):
            clean = {}
            for part in ("mean", "p50", "p95"):
                if isinstance(value.get(part), (int, float)) and not isinstance(value.get(part), bool):
                    clean[part] = value[part]
            if isinstance(value.get("n"), int):
                clean["n"] = value["n"]
            if clean:
                result[key] = clean
        elif isinstance(value, (int, float, bool)) or value is None:
            result[key] = value
    return result


def _safe_counts(counts: Any) -> dict:
    if not isinstance(counts, dict):
        return {}
    allowed = {
        "emitted_tokens",
        "proposed_draft_tokens",
        "accepted_draft_tokens",
        "verify_rounds",
        "target_forward_calls",
        "draft_forward_calls",
        "acceptance_length_draft",
        "acceptance_length_progress",
    }
    result = {
        key: counts[key]
        for key in allowed
        if key in counts and (counts[key] is None or isinstance(counts[key], (int, float)))
    }
    version = _safe_token(counts.get("count_definition_version"))
    if version:
        result["count_definition_version"] = version
    source = _safe_reason(counts.get("count_source"))
    if source != "redacted":
        result["count_source"] = source
    return result


def _safe_resource_usage(resources: Any) -> dict:
    if not isinstance(resources, dict):
        return {}
    result = {}
    for key in ("all_allocated_gpu_hours", "measured_run_gpu_hours", "gpu_memory_fit"):
        if key in resources and (resources[key] is None or isinstance(resources[key], (int, float, bool))):
            result[key] = resources[key]
    devices = []
    for device in resources.get("devices") or []:
        if not isinstance(device, dict):
            continue
        clean = {}
        uuid = _safe_token(device.get("uuid"))
        if uuid:
            clean["uuid"] = uuid
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if isinstance(device.get(key), int):
                clean[key] = device[key]
        scope = _safe_token(device.get("peak_scope"))
        if scope:
            clean["peak_scope"] = scope
        if clean:
            devices.append(clean)
    if devices:
        result["devices"] = devices
    intervals = []
    for interval in resources.get("allocation_intervals") or []:
        if not isinstance(interval, dict):
            continue
        clean = {
            key: interval[key]
            for key in ("start_s", "end_s", "duration_s")
            if isinstance(interval.get(key), (int, float))
        }
        phase = _safe_reason(interval.get("phase"))
        if phase != "redacted":
            clean["phase"] = phase
        if clean:
            intervals.append(clean)
    if intervals:
        result["allocation_intervals"] = intervals
    return result


def redact_record(record: dict) -> dict:
    """Structural share allowlist for one record.

    Unknown keys are dropped rather than recursively scrubbed.  This is the
    share boundary: identities expose public identifiers/hashes, quality and
    correctness expose aggregates, and errors expose codes only.
    """
    allowed = {
        "schema_version",
        "project_scope",
        "include_in_paper",
        "run_id",
        "request_id",
        "group_id",
        "task",
        "split",
        "session_id",
        "repeat_index",
        "execution_order",
        "method",
        "candidate_id",
        "requested_method",
        "effective_method",
        "fallback",
        "synthetic",
        "target_identity",
        "draft_identity",
        "backend_identity",
        "protocol_hash",
        "protocol_scope_hash",
        "correctness_status",
        "acceptance_policy",
        "correctness",
        "quality",
        "measurement_scope",
        "measurement",
        "raw_counts",
        "latency",
        "resource_usage",
        "output_tokens_R",
        "visual_tokens",
        "missing_reasons",
    }
    redacted = {key: record.get(key) for key in allowed if key in record}
    for key in ("run_id", "request_id", "group_id", "session_id", "candidate_id", "task"):
        if key in redacted:
            redacted[key] = _share_id(redacted[key])
    for key in (
        "schema_version", "project_scope", "split", "method", "requested_method",
        "effective_method", "correctness_status", "acceptance_policy", "measurement_scope",
    ):
        if key in redacted:
            redacted[key] = _safe_token(redacted[key])
    for key in ("protocol_hash", "protocol_scope_hash"):
        if key in redacted:
            redacted[key] = _safe_hash(redacted[key])
    for key in ("repeat_index", "execution_order", "output_tokens_R"):
        if key in redacted and not isinstance(redacted[key], int):
            redacted[key] = None
    for key in ("include_in_paper", "synthetic"):
        if key in redacted and not isinstance(redacted[key], bool):
            redacted[key] = None
    for key in ("target_identity", "draft_identity"):
        redacted[key] = _safe_identity(redacted.get(key))
    redacted["backend_identity"] = _safe_identity(redacted.get("backend_identity"))
    redacted["quality"] = _safe_quality(redacted.get("quality"))
    redacted["correctness"] = _safe_correctness(redacted.get("correctness"))
    redacted["latency"] = _safe_latency(redacted.get("latency"))
    redacted["raw_counts"] = _safe_counts(redacted.get("raw_counts"))
    redacted["resource_usage"] = _safe_resource_usage(redacted.get("resource_usage"))
    redacted["visual_tokens"] = {
        key: value
        for key, value in (redacted.get("visual_tokens") or {}).items()
        if key in {"target_V", "draft_V"} and (isinstance(value, (int, float, bool)) or value is None)
    }
    measurement = redacted.get("measurement")
    if isinstance(measurement, dict):
        clean_measurement = {}
        boundary = _safe_token(measurement.get("timing_boundary"))
        if boundary:
            clean_measurement["timing_boundary"] = boundary
        for key in ("includes", "excludes"):
            values = measurement.get(key)
            if isinstance(values, list):
                cleaned = [token for token in (_safe_reason(value) for value in values) if token != "redacted"]
                if cleaned:
                    clean_measurement[key] = cleaned
        redacted["measurement"] = clean_measurement
    else:
        redacted["measurement"] = {}
    redacted["missing_reasons"] = {
        _safe_token(key) or "unknown": _safe_reason(value)
        for key, value in (redacted.get("missing_reasons") or {}).items()
    }
    fallback = redacted.get("fallback")
    if fallback:
        if isinstance(fallback, dict):
            redacted["fallback"] = {
                key: _safe_reason(fallback[key])
                for key in ("code", "reason", "type", "stage")
                if fallback.get(key)
            }
        else:
            redacted["fallback"] = {"code": "fallback"}
    redacted["artifacts"] = {
        key: _safe_hash(value)
        for key, value in (record.get("artifacts") or {}).items()
        if _safe_token(key) and key.endswith("_sha256") and _safe_hash(value) is not None
    }
    if record.get("error"):
        redacted["error"] = {
            key: _safe_reason((record["error"] or {}).get(key))
            for key in ("code", "type", "stage")
            if (record["error"] or {}).get(key)
        }
        if not redacted["error"]:
            redacted["error"] = {"code": "error"}
    return redacted


SERVING_PROFILE = "veladraft-serving-v1"
_SERVING_E2E_BOUNDARIES = {
    "client_request_start_to_last_nonempty_sse_content_chunk",
    "session_start_to_last_request_completion",
}
_SERVING_SURFACES = {
    "openai_chat_completions": "server_endpoint",
    "http": "server_endpoint",
    "offline_llm_chat": "local_client",
}


def detect_external_profile(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    if (
        record.get("schema_version") == SCHEMA_VERSION
        and isinstance(record.get("measurement_scope"), dict)
        and isinstance(record.get("raw_counts"), dict)
        and "emitted" in record["raw_counts"]
        and "proposed" in record["raw_counts"]
        and "accepted" in record["raw_counts"]
    ):
        return SERVING_PROFILE
    return None


def normalize_external_profile(record: dict, *, profile: str = "auto") -> dict:
    """Map a known external record shape into the advisor contract.

    The only supported foreign profile is the PR #17 serving window/request
    record.  Mapping is deliberately narrow: no identity, split, assignment or
    label is invented.  Ambiguous timing or GPU-hour semantics raise instead of
    being relabelled.
    """
    detected = detect_external_profile(record) if profile == "auto" else profile
    if detected != SERVING_PROFILE:
        raise ValueError(f"unsupported external record profile: {profile!r}")
    scope_source = record.get("measurement_scope")
    if not isinstance(scope_source, dict):
        raise ValueError("serving profile measurement_scope must be an object")
    surface = str(scope_source.get("surface") or "")
    if surface not in _SERVING_SURFACES:
        raise ValueError(f"unsupported serving measurement surface: {surface!r}")
    boundary = scope_source.get("timing_boundary")
    if not boundary:
        raise ValueError("serving profile has no timing_boundary; total_s semantics are ambiguous")
    latency_source = record.get("latency")
    if not isinstance(latency_source, dict):
        raise ValueError("serving profile latency must be an object")
    total_s = latency_source.get("total_s")
    explicit_e2e = latency_source.get("e2e_s")
    if total_s is not None and not isinstance(total_s, (int, float)):
        raise ValueError("serving latency.total_s must be numeric")
    if explicit_e2e is not None and not isinstance(explicit_e2e, (int, float)):
        raise ValueError("serving latency.e2e_s must be numeric")
    e2e_s = None
    e2e_reason = None
    if boundary in _SERVING_E2E_BOUNDARIES:
        e2e_s = explicit_e2e if explicit_e2e is not None else total_s
        if e2e_s is None:
            e2e_reason = "serving_boundary_e2e_but_no_timing_value"
    else:
        e2e_reason = "serving_boundary_is_not_end_to_end"

    counts_source = record.get("raw_counts") or {}
    emitted = counts_source.get("emitted")
    proposed = counts_source.get("proposed")
    accepted = counts_source.get("accepted")
    verify_rounds = counts_source.get("verify_rounds")
    counts = build_raw_counts(
        emitted_tokens=emitted,
        proposed_draft_tokens=proposed,
        accepted_draft_tokens=accepted,
        verify_rounds=verify_rounds,
        target_forward_calls=counts_source.get("target_forward_calls"),
        draft_forward_calls=counts_source.get("draft_forward_calls"),
        method="ar" if record.get("draft_identity") is None else "sd",
    )
    counts["count_source"] = "serving_raw_counts"

    # Serving GPU-hours are request/window exposure unless the source explicitly
    # labels an allocation interval.  Concurrent request exposure is never
    # promoted to allocated GPU-hours.
    resources_source = record.get("resource_usage") or {}
    gpu_hours = resources_source.get("gpu_hours")
    gpu_hours_scope = resources_source.get("gpu_hours_scope") or scope_source.get("gpu_hours_scope")
    if gpu_hours_scope is None:
        gpu_hours_scope = "allocation" if record.get("allocation_gpu_hours") is not None else "request_or_window_exposure"
    if gpu_hours is not None and (not isinstance(gpu_hours, (int, float)) or isinstance(gpu_hours, bool) or gpu_hours < 0):
        raise ValueError("serving resource_usage.gpu_hours must be a non-negative number")
    if gpu_hours_scope == "allocation":
        allocated_gpu_hours = gpu_hours
        measured_gpu_hours = gpu_hours
    else:
        allocated_gpu_hours = None
        measured_gpu_hours = None
    devices = []
    for device in resources_source.get("devices") or []:
        if isinstance(device, dict):
            devices.append({"uuid": device.get("uuid"), "peak_allocated_bytes": None})

    mode = str(record.get("mode") or scope_source.get("mode") or "")
    if record.get("draft_identity") is None:
        method = "ar"
        candidate_id = record.get("candidate_id") or mode or "ar"
        draft_identity = None
    else:
        method = "sd"
        candidate_id = record.get("candidate_id") or mode or "external-sd"
        source_draft = record.get("draft_identity") or {}
        draft_identity = {
            "source_profile": SERVING_PROFILE,
            "kind": source_draft.get("kind") if isinstance(source_draft, dict) else None,
            "revision": source_draft.get("revision") if isinstance(source_draft, dict) else None,
            "identity_status": "source_fields_preserved_not_verified",
        }
        draft_identity = {key: value for key, value in draft_identity.items() if value is not None}

    source_target = record.get("target_identity") or {}
    target_identity = {
        "source": SERVING_PROFILE,
        "identity_status": "source_profile_does_not_expose_advisor_identity",
    }
    if isinstance(source_target, dict) and isinstance(source_target.get("revision"), str):
        target_identity["revision"] = source_target["revision"]
    source_backend = record.get("backend_identity") or {}
    backend_identity = {
        "name": (source_backend.get("backend") if isinstance(source_backend, dict) else None) or "vllm",
        "version": (source_backend.get("version") if isinstance(source_backend, dict) else None) or "unknown",
        "source_profile": SERVING_PROFILE,
    }

    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("serving record request_id is required")
    group_id = record.get("group_id") if isinstance(record.get("group_id"), str) and record.get("group_id") else request_id
    session_index = scope_source.get("session_index", scope_source.get("repeat_index"))
    session_id = record.get("session_id") or (f"session-{session_index}" if session_index is not None else "unassigned")
    repeat_index = record.get("repeat_index")
    if not isinstance(repeat_index, int) or isinstance(repeat_index, bool):
        repeat_index = 0
    task = record.get("task") or record.get("category") or "unassigned"
    latency, reasons = build_latency(
        e2e_s=e2e_s,
        ttft_s=latency_source.get("ttft_s"),
        itl_s=None,
        tpot_s=latency_source.get("tpot_s"),
        chunk_gap_s=None,
        ttft_reason="serving_distribution_or_unavailable",
        itl_reason="no_token_visibility",
    )
    if e2e_reason:
        reasons["latency.e2e_s"] = e2e_reason
    reasons.update(
        {
            "split": "missing_holdout_assignment",
            "target_identity": "source_profile_does_not_expose_advisor_identity",
            "quality": "external_profile_has_no_advisor_labels",
            "resource_usage.all_allocated_gpu_hours": "serving_request_or_window_exposure_not_allocation",
            "resource_usage.measured_run_gpu_hours": "serving_request_or_window_exposure_not_run_allocation",
        }
    )
    if draft_identity is None:
        reasons["draft_identity"] = "not_applicable_ar"
    if latency.get("ttft_s") is None:
        reasons["latency.ttft_s"] = "serving_distribution_or_unavailable"
    error = record.get("error")
    if error and not isinstance(error, dict):
        error = {"code": "serving_record_error"}
    correctness_status = record.get("correctness_status") or ("failed" if error else "not_checked")
    normalized = make_record(
        run_id=str(record.get("run_id") or "serving-import"),
        request_id=request_id,
        group_id=group_id,
        task=str(task),
        split=record.get("split") if record.get("split") in ("tuning", "holdout") else None,
        method=method,
        candidate_id=str(candidate_id),
        session_id=str(session_id),
        repeat_index=repeat_index,
        execution_order=repeat_index,
        target_identity=target_identity,
        draft_identity=draft_identity,
        backend_identity=backend_identity,
        protocol_hash=record.get("protocol_hash") or "",
        protocol_scope_hash=record.get("protocol_scope_hash"),
        correctness_status=correctness_status,
        acceptance_policy=record.get("acceptance_policy"),
        measurement_scope=_SERVING_SURFACES[surface],
        raw_counts=counts,
        latency=latency,
        resource_usage=build_resource_usage(
            all_allocated_gpu_hours=allocated_gpu_hours,
            measured_run_gpu_hours=measured_gpu_hours,
            devices=devices,
            gpu_memory_fit=resources_source.get("gpu_memory_fit"),
        ),
        output_tokens_R=emitted if isinstance(emitted, int) else None,
        visual_tokens=None,
        quality={"evaluated": False, "reason": "external_profile_has_no_advisor_labels"},
        error=error,
        missing_reasons=reasons,
        extra={
            "source_profile": SERVING_PROFILE,
            "source_measurement_scope": {
                "concurrency": scope_source.get("concurrency"),
                "queue": scope_source.get("queue"),
                "queue_depth": scope_source.get("queue_depth"),
                "session_index": scope_source.get("session_index"),
                "window": scope_source.get("window"),
                "requests": scope_source.get("requests"),
                "timing_boundary": boundary,
            },
            "serving_gpu_hours": {
                "value": gpu_hours,
                "kind": gpu_hours_scope,
                "source": "resource_usage.gpu_hours",
                "not_allocated_unless_kind_allocation": True,
            },
        },
        measurement={
            "timing_boundary": boundary,
            "includes": ["client_request_start", "generation", "client_visible_commit"] if boundary in _SERVING_E2E_BOUNDARIES else ["generate_only"],
            "excludes": ["model_load", "media_disk_read"],
        },
    )
    errors = validate_record(normalized, expect_request_hash=True)
    if errors:
        raise ValueError(f"serving profile normalization failed validation: {errors[0]['message']}")
    return normalized


def load_records(path: str | Path) -> tuple[list[dict], list[dict]]:
    path = Path(path)
    records: list[dict] = []
    issues: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                issues.append({"line": line_number, "code": "bad_json", "message": exc.msg})
                continue
            record_errors = validate_record(record)
            if record_errors:
                profile = detect_external_profile(record)
                if profile is not None:
                    try:
                        record = normalize_external_profile(record, profile=profile)
                    except (ValueError, TypeError) as exc:
                        issues.append(
                            {
                                "line": line_number,
                                "code": "profile_normalization_failed",
                                "message": str(exc),
                            }
                        )
                        continue
                    record_errors = validate_record(record, expect_request_hash=True)
                if record_errors:
                    issues.extend({"line": line_number, **issue} for issue in record_errors)
                    continue
            records.append(record)
    return records, issues


def write_records(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
            count += 1
    return count


def records_digest(records: Iterable[dict]) -> str:
    return sha256_json(sorted((canonical_json(record) for record in records), key=lambda text: text))


def ar_reference_reuse_error(record: dict, reference: dict) -> str | None:
    """AR token references are reusable only under an identical protocol."""
    if record.get("protocol_hash") != reference.get("protocol_hash"):
        return (
            "AR reference protocol_hash mismatch: "
            f"{reference.get('protocol_hash')} != {record.get('protocol_hash')}"
        )
    if record.get("protocol_scope_hash") != reference.get("protocol_scope_hash"):
        return (
            "AR reference protocol_scope_hash mismatch: "
            f"{reference.get('protocol_scope_hash')} != {record.get('protocol_scope_hash')}"
        )
    left = (record.get("correctness") or {}).get("input_sha256")
    right = (reference.get("correctness") or {}).get("input_sha256")
    if left is not None and right is not None and left != right:
        return f"AR reference input_sha256 mismatch: {right} != {left}"
    return None


def aggregate_record_from_legacy(
    report: dict,
    workload_name: str,
    *,
    run_id: str,
    protocol_scope_hash: str,
    target_identity: dict,
    backend_identity: dict,
    candidate_id: str = "legacy",
) -> dict:
    """Import one old ``vlm-dflash-benchmark-v1`` workload entry.

    Old reports do not contain per-request data, TTFT/ITL or a GPU ledger, so
    the record is marked ``aggregate_only`` and its correctness is
    ``not_checked`` even when the legacy aggregate says token_exact=true.
    """
    entry = (report.get("workloads") or {}).get(workload_name)
    if not isinstance(entry, dict) or entry.get("status") != "measured":
        raise ValueError(f"legacy report has no measured workload {workload_name!r}")
    protocol = report.get("protocol") or {}
    checkpoint = protocol.get("checkpoint")
    method = "ar" if entry.get("target_only_wall_seconds_per_sample") == entry.get("wall_seconds_per_sample") else "sd"
    latency, reasons = build_latency(e2e_s=entry.get("wall_seconds_per_sample"), n_samples=int(entry.get("num_prompts") or 0))
    reasons["latency.e2e_s"] = "aggregate_wall_from_legacy_report"
    counts = {
        "count_definition_version": COUNT_DEFINITION_VERSION,
        "emitted_tokens": None,
        "proposed_draft_tokens": None,
        "accepted_draft_tokens": None,
        "verify_rounds": None,
        "target_forward_calls": None,
        "draft_forward_calls": None,
        "acceptance_length_draft": entry.get("al"),
        "acceptance_length_progress": None,
    }
    missing = {
        **reasons,
        "raw_counts.accepted_draft_tokens": "aggregate_only",
        "raw_counts.verify_rounds": "aggregate_only",
        "resource_usage.all_allocated_gpu_hours": "aggregate_only_no_ledger",
        "ttft_s": "aggregate_only",
    }
    record = make_record(
        run_id=run_id,
        request_id=f"{workload_name}:aggregate",
        group_id=f"{workload_name}:aggregate",
        task=workload_name,
        split="holdout",
        method=method,
        candidate_id=candidate_id,
        session_id="legacy-aggregate",
        repeat_index=0,
        execution_order=0,
        target_identity=target_identity,
        draft_identity=None if method == "ar" else {"checkpoint": checkpoint, "source": "legacy_report"},
        backend_identity=backend_identity,
        protocol_hash=sha256_json({"legacy_protocol": protocol, "workload": workload_name}),
        protocol_scope_hash=protocol_scope_hash,
        correctness_status="not_checked",
        measurement_scope="aggregate_only",
        raw_counts=counts,
        latency=latency,
        resource_usage=build_resource_usage(all_allocated_gpu_hours=None, measured_run_gpu_hours=None),
        output_tokens_R=None,
        missing_reasons=missing,
        extra={
            "legacy": {
                "format": report.get("format"),
                "token_exact": entry.get("token_exact"),
                "al": entry.get("al"),
                "acceptance_rate": entry.get("acceptance_rate"),
                "num_prompts": entry.get("num_prompts"),
                "speedup_vs_target_only": entry.get("speedup_vs_target_only"),
                "verification": protocol.get("verification"),
            }
        },
    )
    return record
