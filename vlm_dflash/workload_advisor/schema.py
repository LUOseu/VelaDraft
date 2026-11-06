"""Validation + canonical hashing for the ``veladraft-platform-v1`` protocol.

The protocol is intentionally self-contained: this module never imports the
training stack (or torch), so ``validate`` / ``report`` / ``recommend`` run on
CPU-only machines and in minimal CI images.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "veladraft-platform-v1"
PROJECT_SCOPE = "platform_extension"
INCLUDE_IN_PAPER = False
COUNT_DEFINITION_VERSION = "veladraft-counts-v1"

CORRECTNESS_STATUSES = (
    "not_checked",
    "verified_greedy_exact",
    "verified_semantic_acceptance",
    "experimental_numeric_divergence",
    "failed",
    "unsupported",
)

EVIDENCE_STATES = (
    "implemented",
    "cpu_tested",
    "gpu_smoke",
    "validated",
    "blocked",
)

MEASUREMENT_SCOPES = (
    "local_client",
    "storage",
    "server_endpoint",
    "aggregate_only",
)

RECOMMENDED_ACTIONS = (
    "accelerate",
    "conditional_accelerate",
    "retain_ar",
    "insufficient_evidence",
)

# Required by the task book. ``schema_version``/``project_scope``/
# ``include_in_paper`` are additionally required by section 5.
REQUIRED_RECORD_KEYS = (
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
    "method",
    "candidate_id",
    "target_identity",
    "draft_identity",
    "backend_identity",
    "protocol_hash",
    "correctness_status",
    "measurement_scope",
    "raw_counts",
    "latency",
    "resource_usage",
)

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "kind",
    "project_scope",
    "include_in_paper",
    "workload_id",
    "media_root",
    "rows_path",
    "target",
    "draft_candidates",
    "decode",
    "backend",
    "measurement_scope",
    "concurrency",
    "bucket_rules",
)

_HEX = set("0123456789abcdef")


def canonical_json(value: Any) -> str:
    """Stable JSON used for every recorded hash (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX


def _issue(code: str, message: str, path: str = "$") -> dict:
    return {"code": code, "message": message, "path": path}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_count(value: Any, path: str, errors: list[dict]) -> None:
    if value is None:
        return
    if not _is_int(value) or value < 0:
        errors.append(_issue("invalid_count", f"{path} must be a non-negative integer or null.", path))


def _check_seconds(value: Any, path: str, errors: list[dict]) -> None:
    if value is None:
        return
    if not _is_number(value) or value < 0:
        errors.append(_issue("invalid_seconds", f"{path} must be a non-negative number or null.", path))


def _has_reason(record: dict, *keys: str) -> bool:
    reasons = record.get("missing_reasons")
    if not isinstance(reasons, dict):
        return False
    return any(key in reasons and reasons[key] not in (None, "") for key in keys)


def validate_record(record: Any, *, expect_request_hash: bool = False) -> list[dict]:
    """Validate one RunRecord. Returns a list of structured issues (empty = ok)."""
    errors: list[dict] = []
    if not isinstance(record, dict):
        return [_issue("not_object", "RunRecord must be a JSON object.")]

    for key in REQUIRED_RECORD_KEYS:
        if key not in record:
            errors.append(_issue("missing_key", f"RunRecord is missing required key {key!r}.", key))
    if errors:
        return errors

    if record["schema_version"] != SCHEMA_VERSION:
        errors.append(
            _issue(
                "unknown_schema_version",
                f"schema_version must be {SCHEMA_VERSION!r}, got {record['schema_version']!r}.",
                "schema_version",
            )
        )
    if record["project_scope"] != PROJECT_SCOPE:
        errors.append(
            _issue("wrong_project_scope", f"project_scope must be {PROJECT_SCOPE!r}.", "project_scope")
        )
    if record["include_in_paper"] is not False:
        errors.append(
            _issue("paper_scope", "include_in_paper must be false for this platform extension.", "include_in_paper")
        )

    for key in ("run_id", "request_id", "group_id"):
        if not isinstance(record[key], str) or not record[key]:
            errors.append(_issue("invalid_id", f"{key} must be a non-empty string.", key))

    if not isinstance(record.get("session_id"), str) or not record.get("session_id"):
        errors.append(_issue("invalid_session", "session_id must be a non-empty string.", "session_id"))
    if not _is_int(record.get("repeat_index")) or record.get("repeat_index") < 0:
        errors.append(_issue("invalid_repeat", "repeat_index must be a non-negative integer.", "repeat_index"))
    split = record.get("split")
    if split is not None and split not in ("tuning", "holdout"):
        errors.append(_issue("invalid_split", "split must be 'tuning', 'holdout' or null.", "split"))
    elif split is None and not _has_reason(record, "split", "holdout_assignment"):
        errors.append(
            _issue(
                "missing_reason",
                "split=null needs missing_reasons['split'] (for example, missing_holdout_assignment).",
                "missing_reasons.split",
            )
        )
    for key in ("method", "candidate_id"):
        if not isinstance(record.get(key), str) or not record.get(key):
            errors.append(_issue("invalid_method", f"{key} must be a non-empty string.", key))

    method = record.get("method") or record.get("candidate_id")
    if method is not None and not isinstance(method, str):
        errors.append(_issue("invalid_method", "method/candidate_id must be a string when present.", "method"))

    for key in ("target_identity", "backend_identity"):
        if not isinstance(record[key], dict) or not record[key]:
            errors.append(_issue("invalid_identity", f"{key} must be a non-empty object.", key))

    draft_identity = record["draft_identity"]
    if method == "ar":
        if draft_identity is not None:
            errors.append(_issue("ar_has_draft", "AR records must set draft_identity=null.", "draft_identity"))
        elif not _has_reason(record, "draft_identity", "draft_identity_not_applicable"):
            errors.append(
                _issue(
                    "missing_reason",
                    "AR draft_identity=null needs missing_reasons['draft_identity'] (e.g. not_applicable_ar).",
                    "missing_reasons.draft_identity",
                )
            )
    else:
        if not isinstance(draft_identity, dict) or not draft_identity:
            errors.append(
                _issue("missing_draft_identity", "Speculative records need a non-null draft_identity.", "draft_identity")
            )

    if not is_sha256(record["protocol_hash"]):
        errors.append(_issue("invalid_protocol_hash", "protocol_hash must be a 64-char sha256.", "protocol_hash"))
    if expect_request_hash and record.get("protocol_scope_hash") is not None:
        if not is_sha256(record["protocol_scope_hash"]):
            errors.append(
                _issue("invalid_protocol_scope_hash", "protocol_scope_hash must be a 64-char sha256.", "protocol_scope_hash")
            )

    status = record["correctness_status"]
    if status not in CORRECTNESS_STATUSES:
        errors.append(
            _issue(
                "invalid_correctness_status",
                f"correctness_status must be one of {list(CORRECTNESS_STATUSES)}, got {status!r}.",
                "correctness_status",
            )
        )
    if status == "failed" and not record.get("error"):
        errors.append(_issue("missing_error", "correctness_status=failed requires an error object.", "error"))
    if record.get("acceptance_policy") == "relaxed" and status == "verified_greedy_exact":
        errors.append(
            _issue(
                "relaxed_not_exact",
                "acceptance_policy=relaxed cannot be reported as verified_greedy_exact.",
                "correctness_status",
            )
        )

    scope = record["measurement_scope"]
    if scope not in MEASUREMENT_SCOPES:
        errors.append(
            _issue(
                "invalid_measurement_scope",
                f"measurement_scope must be one of {list(MEASUREMENT_SCOPES)}, got {scope!r}.",
                "measurement_scope",
            )
        )

    counts = record["raw_counts"]
    if not isinstance(counts, dict):
        errors.append(_issue("invalid_raw_counts", "raw_counts must be an object.", "raw_counts"))
    else:
        if counts.get("count_definition_version") != COUNT_DEFINITION_VERSION:
            errors.append(
                _issue(
                    "count_version",
                    f"raw_counts.count_definition_version must be {COUNT_DEFINITION_VERSION!r}.",
                    "raw_counts.count_definition_version",
                )
            )
        for key in (
            "emitted_tokens",
            "proposed_draft_tokens",
            "accepted_draft_tokens",
            "verify_rounds",
            "target_forward_calls",
            "draft_forward_calls",
        ):
            if key not in counts:
                errors.append(_issue("missing_count", f"raw_counts is missing {key!r} (null when unknown).", f"raw_counts.{key}"))
            else:
                _check_count(counts[key], f"raw_counts.{key}", errors)

        rounds = counts.get("verify_rounds")
        accepted = counts.get("accepted_draft_tokens")
        emitted = counts.get("emitted_tokens")
        if _is_int(rounds) and rounds > 0 and _is_int(accepted):
            expected = accepted / rounds
            recorded = counts.get("acceptance_length_draft")
            if recorded is not None and (not _is_number(recorded) or abs(recorded - expected) > 1e-9):
                errors.append(
                    _issue(
                        "al_draft_mismatch",
                        f"acceptance_length_draft={recorded!r} but accepted/rounds={expected!r}.",
                        "raw_counts.acceptance_length_draft",
                    )
                )
        if _is_int(rounds) and rounds > 0 and _is_int(emitted):
            expected = emitted / rounds
            recorded = counts.get("acceptance_length_progress")
            if recorded is not None and (not _is_number(recorded) or abs(recorded - expected) > 1e-9):
                errors.append(
                    _issue(
                        "al_progress_mismatch",
                        f"acceptance_length_progress={recorded!r} but emitted/rounds={expected!r}.",
                        "raw_counts.acceptance_length_progress",
                    )
                )

        if method == "ar":
            for key in ("proposed_draft_tokens", "accepted_draft_tokens", "verify_rounds", "acceptance_length_draft"):
                if counts.get(key) is not None:
                    errors.append(
                        _issue(
                            "ar_acceptance_not_null",
                            f"AR raw_counts.{key} must be null + not_applicable (not 0).",
                            f"raw_counts.{key}",
                        )
                    )

    latency = record["latency"]
    if not isinstance(latency, dict):
        errors.append(_issue("invalid_latency", "latency must be an object.", "latency"))
    else:
        if "e2e_s" not in latency:
            errors.append(_issue("missing_e2e", "latency.e2e_s must be present (null + reason if unknown).", "latency.e2e_s"))
        for key, value in latency.items():
            if key in {"itl_s", "chunk_gap_s"}:
                if value is None:
                    continue
                if isinstance(value, dict):
                    unknown = set(value) - {"mean", "p50", "p95", "n"}
                    if unknown:
                        errors.append(
                            _issue(
                                "invalid_latency_distribution",
                                f"latency.{key} has unsupported keys {sorted(unknown)}.",
                                f"latency.{key}",
                            )
                        )
                    if not any(part in value and value[part] is not None for part in ("mean", "p50", "p95")):
                        errors.append(
                            _issue(
                                "invalid_latency_distribution",
                                f"latency.{key} needs at least one of mean/p50/p95.",
                                f"latency.{key}",
                            )
                        )
                    for part in ("mean", "p50", "p95"):
                        if part in value:
                            _check_seconds(value[part], f"latency.{key}.{part}", errors)
                    if "n" in value and (not _is_int(value["n"]) or value["n"] < 0):
                        errors.append(
                            _issue(
                                "invalid_latency_distribution",
                                f"latency.{key}.n must be a non-negative integer.",
                                f"latency.{key}.n",
                            )
                        )
                else:
                    _check_seconds(value, f"latency.{key}", errors)
            elif key.endswith("_s") or key.endswith("_seconds"):
                if key.endswith("_seconds"):
                    errors.append(
                        _issue("seconds_suffix", f"latency key {key!r} must use the '_s' suffix.", f"latency.{key}")
                    )
                _check_seconds(value, f"latency.{key}", errors)
        if latency.get("ttft_s") is None and not _has_reason(
            record, "latency.ttft_s", "ttft_s", "ttft_unavailable"
        ):
            errors.append(
                _issue(
                    "missing_reason",
                    "latency.ttft_s=null needs missing_reasons['latency.ttft_s'] (e.g. no_streaming).",
                    "missing_reasons.latency.ttft_s",
                )
            )
        if latency.get("itl_s") is None and not _has_reason(record, "latency.itl_s", "itl_s", "no_token_visibility"):
            errors.append(
                _issue(
                    "missing_reason",
                    "latency.itl_s=null needs missing_reasons['latency.itl_s'] (e.g. no_token_visibility).",
                    "missing_reasons.latency.itl_s",
                )
            )
        if scope != "aggregate_only" and latency.get("e2e_s") is None and not _has_reason(record, "latency.e2e_s", "e2e_s"):
            errors.append(
                _issue(
                    "missing_reason",
                    "latency.e2e_s=null needs missing_reasons['latency.e2e_s'].",
                    "missing_reasons.latency.e2e_s",
                )
            )

    resources = record["resource_usage"]
    if not isinstance(resources, dict):
        errors.append(_issue("invalid_resource_usage", "resource_usage must be an object.", "resource_usage"))
    else:
        for key in ("all_allocated_gpu_hours", "measured_run_gpu_hours"):
            if key not in resources:
                errors.append(
                    _issue("missing_resource", f"resource_usage is missing {key!r} (null + reason if unknown).", f"resource_usage.{key}")
                )
            else:
                _check_seconds(resources[key], f"resource_usage.{key}", errors)
        if resources.get("all_allocated_gpu_hours") is None and not _has_reason(
            record, "resource_usage.all_allocated_gpu_hours", "all_allocated_gpu_hours"
        ):
            errors.append(
                _issue(
                    "missing_reason",
                    "resource_usage.all_allocated_gpu_hours=null needs a missing reason.",
                    "missing_reasons.resource_usage.all_allocated_gpu_hours",
                )
            )
        devices = resources.get("devices")
        if devices is not None and not isinstance(devices, list):
            errors.append(_issue("invalid_devices", "resource_usage.devices must be a list.", "resource_usage.devices"))
        memory_fit = resources.get("gpu_memory_fit")
        if memory_fit is not None and not isinstance(memory_fit, bool):
            errors.append(
                _issue("invalid_memory_fit", "resource_usage.gpu_memory_fit must be boolean or null.",
                       "resource_usage.gpu_memory_fit")
            )

    correctness = record.get("correctness")
    if status == "verified_greedy_exact":
        if not isinstance(correctness, dict):
            errors.append(
                _issue("missing_correctness", "verified_greedy_exact needs correctness details.", "correctness")
            )
        else:
            n_requests = correctness.get("n_requests")
            if not _is_int(n_requests) or n_requests < 1:
                errors.append(
                    _issue("missing_denominator", "correctness.n_requests must be >= 1 (keep the denominator).", "correctness.n_requests")
                )
            failed = correctness.get("failed_request_ids")
            if not isinstance(failed, list):
                errors.append(
                    _issue("missing_failed_ids", "correctness.failed_request_ids must be a list.", "correctness.failed_request_ids")
                )
            elif failed:
                errors.append(
                    _issue(
                        "exact_with_failures",
                        "verified_greedy_exact cannot list failed_request_ids.",
                        "correctness.failed_request_ids",
                    )
                )
            for key in ("reference_hash", "candidate_hash"):
                if not is_sha256(correctness.get(key)):
                    errors.append(
                        _issue("missing_correctness_hash", f"correctness.{key} must be a sha256.", f"correctness.{key}")
                    )

    quality = record.get("quality")
    if isinstance(quality, dict) and "evaluated" in quality and not isinstance(quality["evaluated"], bool):
        errors.append(_issue("invalid_quality_evaluated", "quality.evaluated must be boolean.", "quality.evaluated"))
    if isinstance(quality, dict) and quality.get("evaluated") is False:
        if not _has_reason(record, "quality", "quality_not_evaluated"):
            errors.append(
                _issue(
                    "missing_quality_reason",
                    "quality.evaluated=false needs missing_reasons['quality'].",
                    "missing_reasons.quality",
                )
            )

    if record.get("synthetic") is not None and not isinstance(record["synthetic"], bool):
        errors.append(_issue("invalid_synthetic", "synthetic must be boolean when present.", "synthetic"))

    return errors


def validate_manifest_shape(manifest: Any) -> list[dict]:
    """Structural manifest checks that do not touch the filesystem."""
    errors: list[dict] = []
    if not isinstance(manifest, dict):
        return [_issue("not_object", "workload manifest must be a JSON object.")]
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(_issue("missing_key", f"manifest is missing required key {key!r}.", key))
    if errors:
        return errors

    if manifest["schema_version"] != SCHEMA_VERSION:
        errors.append(
            _issue(
                "unknown_schema_version",
                f"schema_version must be {SCHEMA_VERSION!r}, got {manifest['schema_version']!r}.",
                "schema_version",
            )
        )
    if manifest.get("kind") != "workload_manifest":
        errors.append(_issue("wrong_kind", "kind must be 'workload_manifest'.", "kind"))
    if manifest.get("project_scope") != PROJECT_SCOPE:
        errors.append(_issue("wrong_project_scope", f"project_scope must be {PROJECT_SCOPE!r}.", "project_scope"))
    if manifest.get("include_in_paper") is not False:
        errors.append(_issue("paper_scope", "include_in_paper must be false.", "include_in_paper"))
    for key in ("workload_id", "media_root", "rows_path"):
        if not isinstance(manifest.get(key), str) or not manifest.get(key):
            errors.append(_issue("invalid_string", f"{key} must be a non-empty string.", key))

    target = manifest.get("target")
    if not isinstance(target, dict) or not target:
        errors.append(_issue("invalid_target", "target must be a non-empty object.", "target"))
    else:
        for key in ("model_id", "revision", "dtype"):
            if not target.get(key):
                errors.append(_issue("missing_target_identity", f"target.{key} is required.", f"target.{key}"))
        if not target.get("processor") and not target.get("processor_revision"):
            errors.append(
                _issue("missing_processor", "target.processor (or processor_revision) is required.", "target.processor")
            )

    candidates = manifest.get("draft_candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append(_issue("invalid_candidates", "draft_candidates must be a non-empty list.", "draft_candidates"))
    else:
        seen = set()
        for index, candidate in enumerate(candidates):
            path = f"draft_candidates[{index}]"
            if not isinstance(candidate, dict):
                errors.append(_issue("invalid_candidate", "candidate must be an object.", path))
                continue
            candidate_id = candidate.get("candidate_id")
            if not candidate_id or candidate_id in seen:
                errors.append(_issue("duplicate_candidate", "candidate_id must be unique and non-empty.", f"{path}.candidate_id"))
            seen.add(candidate_id)
            method = candidate.get("method")
            if method not in ("ar", "sd"):
                errors.append(_issue("invalid_method", "method must be 'ar' or 'sd'.", f"{path}.method"))
            if method == "ar":
                if candidate.get("draft_identity") is not None:
                    errors.append(_issue("ar_has_draft", "AR candidate must have draft_identity=null.", f"{path}.draft_identity"))
            elif method == "sd":
                if not candidate.get("algorithm"):
                    errors.append(_issue("missing_algorithm", "SD candidate needs algorithm.", f"{path}.algorithm"))
                if not candidate.get("checkpoint"):
                    errors.append(_issue("missing_checkpoint", "SD candidate needs checkpoint.", f"{path}.checkpoint"))
                if not candidate.get("draft_identity"):
                    errors.append(_issue("missing_draft_identity", "SD candidate needs draft_identity.", f"{path}.draft_identity"))
        if not any(isinstance(item, dict) and item.get("method") == "ar" for item in candidates):
            errors.append(_issue("missing_ar", "draft_candidates must include the AR baseline.", "draft_candidates"))

    decode = manifest.get("decode")
    if not isinstance(decode, dict):
        errors.append(_issue("invalid_decode", "decode must be an object.", "decode"))
    else:
        if decode.get("temperature") not in (0, 0.0):
            errors.append(_issue("not_greedy", "first-round decode.temperature must be 0.", "decode.temperature"))
        if not _is_int(decode.get("max_new_tokens")) or decode["max_new_tokens"] < 1:
            errors.append(_issue("invalid_max_new_tokens", "decode.max_new_tokens must be >= 1.", "decode.max_new_tokens"))
        stop_ids = decode.get("stop_ids")
        if not isinstance(stop_ids, list) or not stop_ids:
            errors.append(_issue("invalid_stop_ids", "decode.stop_ids must be a non-empty list.", "decode.stop_ids"))
        elif not all(_is_int(value) and value >= 0 for value in stop_ids):
            errors.append(_issue("invalid_stop_ids", "decode.stop_ids must contain non-negative integers.", "decode.stop_ids"))
        if not decode.get("eos_policy"):
            errors.append(_issue("missing_eos_policy", "decode.eos_policy is required.", "decode.eos_policy"))

    backend = manifest.get("backend")
    if not isinstance(backend, dict) or not backend.get("name"):
        errors.append(_issue("invalid_backend", "backend.name is required.", "backend"))

    if manifest.get("measurement_scope") not in MEASUREMENT_SCOPES:
        errors.append(_issue("invalid_measurement_scope", "measurement_scope is invalid.", "measurement_scope"))
    if not _is_int(manifest.get("concurrency")) or manifest["concurrency"] < 1:
        errors.append(_issue("invalid_concurrency", "concurrency must be an integer >= 1.", "concurrency"))

    bucket_rules = manifest.get("bucket_rules")
    if not isinstance(bucket_rules, dict):
        errors.append(_issue("invalid_bucket_rules", "bucket_rules must be an object.", "bucket_rules"))
    else:
        for axis in ("target_V", "ar_R"):
            edges = bucket_rules.get(axis)
            if not isinstance(edges, list) or not edges or edges != sorted(edges) or len(set(edges)) != len(edges):
                errors.append(_issue("invalid_bucket_edges", f"bucket_rules.{axis} must be sorted unique edges.", f"bucket_rules.{axis}"))
        min_groups = bucket_rules.get("per_bucket_min_groups", 20)
        if not _is_int(min_groups) or min_groups < 1:
            errors.append(_issue("invalid_min_groups", "per_bucket_min_groups must be >= 1.", "bucket_rules.per_bucket_min_groups"))

    return errors


def protocol_payload(manifest: dict, rows_digest: str | None, labels_digest: str | None) -> dict:
    """The frozen protocol hashed for every run in this workload.

    Includes actual input identity (row/label digests), generation and stop
    rules, model/processor identity, precision, backend, vision config and the
    timing boundary. Excludes mutable bookkeeping such as output paths.
    """
    target = manifest.get("target") or {}
    decode = manifest.get("decode") or {}
    backend = manifest.get("backend") or {}
    measurement = manifest.get("measurement_scope")
    return {
        "schema_version": SCHEMA_VERSION,
        "workload_id": manifest.get("workload_id"),
        "input_identity": {
            "rows_sha256": rows_digest,
            "labels_sha256": labels_digest,
            "media_root_rule": "manifest_relative_no_remote_fetch",
        },
        "target_identity": target,
        "draft_candidates": manifest.get("draft_candidates"),
        "decode": decode,
        "backend_identity": backend,
        "measurement_scope": measurement,
        "timing_boundary": (manifest.get("timing") or {}).get(
            "boundary", "request_ready_to_committed_output"
        ),
        "concurrency": manifest.get("concurrency"),
        "seed": manifest.get("seed"),
        "resolution": manifest.get("resolution"),
        "bucket_rules": manifest.get("bucket_rules"),
        "scorers": manifest.get("scorers"),
        "quality_thresholds": manifest.get("quality_thresholds"),
    }


def protocol_scope_hash(payload: dict) -> str:
    return sha256_json(payload)


def request_protocol_hash(payload: dict, request_identity: dict) -> str:
    return sha256_json({**payload, "request_identity": request_identity})


def clip(value: Any, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
