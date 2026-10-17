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
    sha256_json,
    validate_record,
)

# Old harness format is imported as aggregate-only evidence.
LEGACY_BENCHMARK_FORMAT = "vlm-dflash-benchmark-v1"
_ABS_PATH = re.compile(r"(?:(?:/Users|/home|/private|/tmp|[A-Za-z]:\\\\)[^\s\"']+)")


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


def _scrub_abs_paths(value: Any) -> Any:
    if isinstance(value, str):
        return _ABS_PATH.sub("<redacted-path>", value)
    if isinstance(value, list):
        return [_scrub_abs_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _scrub_abs_paths(item) for key, item in value.items()}
    return value


def redact_record(record: dict) -> dict:
    """Share-safe record: hashes and aggregates only, no prompts or paths."""
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
    for key in ("target_identity", "draft_identity"):
        identity = redacted.get(key)
        if isinstance(identity, dict):
            redacted[key] = {
                item_key: item_value
                for item_key, item_value in identity.items()
                if item_key not in {"model_path", "checkpoint", "local_path", "checkpoint_path"}
            }
    redacted["artifacts"] = {
        key: value for key, value in (record.get("artifacts") or {}).items() if key.endswith("_sha256")
    }
    if record.get("error"):
        redacted["error"] = {
            "type": (record["error"] or {}).get("type"),
            "stage": (record["error"] or {}).get("stage"),
            "message": _scrub_abs_paths((record["error"] or {}).get("message")),
        }
    return redacted


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
