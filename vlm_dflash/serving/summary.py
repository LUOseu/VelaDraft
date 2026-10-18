"""Protocol-aware summary for serving correctness and HTTP measurements.

The summary is deliberately conservative.  Throughput ratios are emitted only
for diagnostic inspection unless the AR and N-Gram documents share a verified
common protocol and a complete per-request pairing.  Mode and speculative
configuration are the only expected differences.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .metrics import summarize_latencies

SUMMARY_SCHEMA = "veladraft-serving-summary-v2"
SUCCESS_FINISH_REASONS = {"stop", "length"}
NATURAL_CATEGORIES = {"ocr", "chart_doc", "vqa", "caption"}
_MODE_PREFIX_RE = re.compile(r"^(?:ar|ngram)-")


def _first_record(doc: dict[str, Any]) -> dict[str, Any]:
    records = doc.get("records") or []
    return records[0] if records else {}


def _identity_value(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _target_signature(record: dict[str, Any]) -> dict[str, Any]:
    target = record.get("target_identity") or {}
    return {
        "model": target.get("model"),
        "precision": target.get("precision"),
        "revision": target.get("revision"),
        "config_sha256": target.get("config_sha256"),
        "architecture": target.get("architecture"),
        "tokenizer_sha256": _identity_value(target, ("tokenizer", "sha256")),
        "processor_sha256": _identity_value(target, ("processor", "sha256")),
        "template_sha256": _identity_value(target, ("template", "sha256")),
        "template_effective_source": _identity_value(target, ("template", "effective_source")),
        "weight_content_sha256": target.get("weight_content_sha256"),
        "identity_complete": target.get("identity_complete"),
    }


def _backend_signature(record: dict[str, Any]) -> dict[str, Any]:
    backend = record.get("backend_identity") or {}
    topology = backend.get("device_topology") or {}
    return {
        "backend": backend.get("backend"),
        "version": backend.get("version"),
        "code_commit": backend.get("code_commit"),
        "attention_backend": backend.get("attention_backend"),
        "cudagraph_mode": backend.get("cudagraph_mode"),
        "tensor_parallel_size": topology.get("tensor_parallel_size"),
        "device_uuids": [device.get("uuid") for device in topology.get("devices", [])],
        "labels_source": backend.get("labels_source"),
        "applied_evidence": backend.get("applied_evidence"),
    }


def _scope_signature(record: dict[str, Any]) -> dict[str, Any]:
    scope = record.get("measurement_scope") or {}
    return {
        "surface": scope.get("surface"),
        "stream": scope.get("stream"),
        "prefix_caching": scope.get("prefix_caching"),
        "timing_boundary": scope.get("timing_boundary"),
    }


def _signature_variants(records: list[dict[str, Any]], signature) -> int:
    return len({json.dumps(signature(record), sort_keys=True) for record in records})


def _record_success(record: dict[str, Any]) -> bool:
    if record.get("cancelled"):
        return False
    if record.get("error"):
        return False
    terminal = record.get("terminal_status")
    if terminal is not None and terminal != "completed":
        return False
    return record.get("finish_reason") in SUCCESS_FINISH_REASONS


def _window_key(window: dict[str, Any]) -> tuple[Any, Any]:
    scope = window.get("measurement_scope") or {}
    return scope.get("concurrency"), scope.get("session_index")


def _index_windows(doc: dict[str, Any], side: str) -> tuple[dict[tuple[Any, Any], dict[str, Any]], list[str]]:
    indexed: dict[tuple[Any, Any], dict[str, Any]] = {}
    errors: list[str] = []
    for window in doc.get("windows", []):
        key = _window_key(window)
        if key in indexed:
            errors.append(f"{side}: duplicate window {key}")
        else:
            indexed[key] = window
    return indexed, errors


def _index_records(doc: dict[str, Any], side: str) -> tuple[dict[tuple[Any, Any, str], dict[str, Any]], list[str]]:
    indexed: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    errors: list[str] = []
    for record in doc.get("records", []):
        scope = record.get("measurement_scope") or {}
        request_id = str(record.get("request_id", ""))
        key = (scope.get("concurrency"), scope.get("session_index"), _MODE_PREFIX_RE.sub("", request_id))
        if key in indexed:
            errors.append(f"{side}: duplicate request record {key}")
        else:
            indexed[key] = record
    return indexed, errors


def _expected_window_keys(doc: dict[str, Any]) -> set[tuple[int, int]]:
    meta = doc.get("meta") or {}
    try:
        concurrency = [int(value) for value in str(meta.get("concurrency", "")).split(",") if value.strip()]
        sessions = int(meta.get("sessions", 0))
    except (TypeError, ValueError):
        return set()
    return {(c, session) for c in concurrency for session in range(1, sessions + 1)}


def _protocol_check(ar_doc: dict[str, Any], sd_doc: dict[str, Any]) -> dict[str, Any]:
    ar_record = _first_record(ar_doc)
    sd_record = _first_record(sd_doc)
    reasons: list[str] = []
    ar_meta = ar_doc.get("meta") or {}
    sd_meta = sd_doc.get("meta") or {}
    ar_manifest = ar_meta.get("manifest_sha256")
    sd_manifest = sd_meta.get("manifest_sha256")
    if not ar_manifest or not sd_manifest:
        reasons.append("fixture_manifest_sha256_missing")
    elif ar_manifest != sd_manifest:
        reasons.append("fixture_manifest_sha256_mismatch")

    for side, doc in (("ar", ar_doc), ("sd", sd_doc)):
        records = list(doc.get("records") or [])
        if not records:
            reasons.append(f"{side}_records_missing")
            continue
        if _signature_variants(records, _target_signature) != 1:
            reasons.append(f"{side}_target_identity_inconsistent_across_records")
        if _signature_variants(records, _backend_signature) != 1:
            reasons.append(f"{side}_backend_identity_inconsistent_across_records")
        if _signature_variants(records, _scope_signature) != 1:
            reasons.append(f"{side}_measurement_scope_inconsistent_across_records")
        protocol_hashes = [record.get("protocol_hash") for record in records]
        if not all(protocol_hashes):
            reasons.append(f"{side}_protocol_hash_missing")
        elif len(set(protocol_hashes)) != 1:
            reasons.append(f"{side}_protocol_hash_inconsistent_across_records")

    ar_target = _target_signature(ar_record)
    sd_target = _target_signature(sd_record)
    for field in (
        "model",
        "precision",
        "config_sha256",
        "architecture",
        "tokenizer_sha256",
        "processor_sha256",
        "template_sha256",
    ):
        if not ar_target.get(field) or not sd_target.get(field):
            reasons.append(f"target_{field}_missing")
        elif ar_target[field] != sd_target[field]:
            reasons.append(f"target_{field}_mismatch")
    for field in ("revision", "weight_content_sha256"):
        if not ar_target.get(field) or not sd_target.get(field):
            reasons.append(f"target_{field}_missing")
        elif ar_target[field] != sd_target[field]:
            reasons.append(f"target_{field}_mismatch")
    if not ar_target.get("template_effective_source") or not sd_target.get("template_effective_source"):
        reasons.append("target_template_effective_source_missing")
    elif ar_target["template_effective_source"] != sd_target["template_effective_source"]:
        reasons.append("target_template_effective_source_mismatch")
    if ar_target.get("identity_complete") is not True or sd_target.get("identity_complete") is not True:
        reasons.append("target_content_identity_incomplete")

    ar_backend = _backend_signature(ar_record)
    sd_backend = _backend_signature(sd_record)
    for field in (
        "backend",
        "version",
        "code_commit",
        "attention_backend",
        "cudagraph_mode",
        "tensor_parallel_size",
        "device_uuids",
    ):
        if not ar_backend.get(field) or not sd_backend.get(field):
            reasons.append(f"backend_{field}_missing")
        elif ar_backend[field] != sd_backend[field]:
            reasons.append(f"backend_{field}_mismatch")
    if ar_backend.get("applied_evidence") is not True or sd_backend.get("applied_evidence") is not True:
        reasons.append("backend_labels_not_verified_from_engine_or_runtime_evidence")
    if ar_backend.get("labels_source") in {None, "", "cli_declared_unverified"}:
        reasons.append("ar_backend_labels_source_untrusted")
    if sd_backend.get("labels_source") in {None, "", "cli_declared_unverified"}:
        reasons.append("sd_backend_labels_source_untrusted")

    for field in ("stream", "warmup", "max_new_tokens"):
        if ar_meta.get(field) is None or sd_meta.get(field) is None:
            reasons.append(f"measurement_{field}_missing")
        elif ar_meta.get(field) != sd_meta.get(field):
            reasons.append(f"measurement_{field}_mismatch")
    ar_scope = _first_record(ar_doc).get("measurement_scope") or {}
    sd_scope = _first_record(sd_doc).get("measurement_scope") or {}
    for field in ("prefix_caching", "timing_boundary"):
        if ar_scope.get(field) is None or sd_scope.get(field) is None:
            reasons.append(f"measurement_scope_{field}_missing")
        elif ar_scope.get(field) != sd_scope.get(field):
            reasons.append(f"measurement_scope_{field}_mismatch")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "allowed_differences": ["mode", "speculative_config"],
        "fixture_manifest_sha256": ar_manifest if ar_manifest == sd_manifest else None,
        "target": {"ar": ar_target, "ngram": sd_target},
        "backend": {"ar": ar_backend, "ngram": sd_backend},
        "timing_scope": {
            "ar": ar_scope.get("timing_boundary"),
            "ngram": sd_scope.get("timing_boundary"),
        },
        "request_budget": {
            "ar": ar_meta.get("max_new_tokens"),
            "ngram": sd_meta.get("max_new_tokens"),
        },
    }


def _record_control(record: dict[str, Any]) -> tuple[Optional[int], bool, bool]:
    control = record.get("request_control")
    if isinstance(control, dict) and control.get("source") == "actual_request_body":
        return control.get("max_tokens"), bool(control.get("ignore_eos")), True
    scope = record.get("measurement_scope") or {}
    return scope.get("max_new_tokens"), False, False


def _pair_check(
    ar_doc: dict[str, Any],
    sd_doc: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[tuple[int, int], dict[str, Any]]]:
    ar_windows, ar_window_errors = _index_windows(ar_doc, "ar")
    sd_windows, sd_window_errors = _index_windows(sd_doc, "sd")
    errors = ar_window_errors + sd_window_errors
    expected_keys = _expected_window_keys(ar_doc) | _expected_window_keys(sd_doc)
    if not expected_keys:
        errors.append("measurement_window_keys_unavailable")
    for key in sorted(expected_keys):
        if key not in ar_windows:
            errors.append(f"ar: missing window {key}")
        if key not in sd_windows:
            errors.append(f"sd: missing window {key}")
    for key in sorted(set(ar_windows) | set(sd_windows)):
        if key not in expected_keys:
            errors.append(f"extra window outside metadata: {key}")

    ar_records, ar_record_errors = _index_records(ar_doc, "ar")
    sd_records, sd_record_errors = _index_records(sd_doc, "sd")
    errors.extend(ar_record_errors)
    errors.extend(sd_record_errors)
    pairs: list[dict[str, Any]] = []
    by_window: dict[tuple[int, int], dict[str, Any]] = {}
    for window_key in sorted(expected_keys):
        ar_items = {key[2]: record for key, record in ar_records.items() if key[:2] == window_key}
        sd_items = {key[2]: record for key, record in sd_records.items() if key[:2] == window_key}
        missing_sd = sorted(set(ar_items) - set(sd_items))
        missing_ar = sorted(set(sd_items) - set(ar_items))
        if missing_sd:
            errors.append(f"{window_key}: sd missing request records {missing_sd[:5]}")
        if missing_ar:
            errors.append(f"{window_key}: ar missing request records {missing_ar[:5]}")
        ar_window = ar_windows.get(window_key, {})
        sd_window = sd_windows.get(window_key, {})
        budget_verified = True
        failures = 0
        for request_id in sorted(set(ar_items) & set(sd_items)):
            left = ar_items[request_id]
            right = sd_items[request_id]
            left_budget, left_ignore, left_verified = _record_control(left)
            right_budget, right_ignore, right_verified = _record_control(right)
            budget_verified = budget_verified and left_verified and right_verified
            if left_budget != right_budget:
                errors.append(f"{window_key}:{request_id}: max_tokens mismatch")
            if left_ignore != right_ignore:
                errors.append(f"{window_key}:{request_id}: ignore_eos mismatch")
            for field in ("prompt_sha256", "image_sha256", "category"):
                if left.get(field) != right.get(field):
                    errors.append(f"{window_key}:{request_id}: {field} mismatch")
            failures += int(not _record_success(left)) + int(not _record_success(right))
            pairs.append(
                {
                    "concurrency": window_key[0],
                    "session": window_key[1],
                    "request_id": request_id,
                    "prompt_sha256": left.get("prompt_sha256"),
                    "image_sha256": left.get("image_sha256"),
                    "max_tokens": left_budget,
                    "budget_verified": left_verified and right_verified,
                }
            )
        ar_requests = ar_window.get("num_requests")
        sd_requests = sd_window.get("num_requests")
        if ar_requests != len(ar_items) or sd_requests != len(sd_items):
            errors.append(f"{window_key}: window/record count mismatch")
        paired_count = len(set(ar_items) & set(sd_items))
        if paired_count == 0:
            errors.append(f"{window_key}: no paired request records")
        budget_verified = bool(paired_count) and budget_verified
        by_window[window_key] = {
            "requests": len(ar_items),
            "paired_requests": paired_count,
            "missing_ar": missing_ar,
            "missing_sd": missing_sd,
            "budget_verified": budget_verified,
            "failures": failures,
            "ar_wall_s": (ar_window.get("latency") or {}).get("total_s"),
            "sd_wall_s": (sd_window.get("latency") or {}).get("total_s"),
        }
    return pairs, errors, by_window


def _latency_summary(doc: dict[str, Any]) -> dict[str, Any]:
    records = []
    for record in doc.get("records", []):
        scope = record.get("measurement_scope") or {}
        if record.get("cancelled"):
            continue
        if scope.get("fixture_category") not in NATURAL_CATEGORIES:
            continue
        if "-nat-" not in str(record.get("request_id", "")):
            continue
        if _record_success(record):
            records.append(record)
    ttft = [record["latency"].get("ttft_s") for record in records if record.get("latency", {}).get("ttft_s") is not None]
    e2e = [record["latency"].get("e2e_s") for record in records if record.get("latency", {}).get("e2e_s") is not None]
    tpot = []
    tpot_unavailable = 0
    for record in records:
        latency = record.get("latency") or {}
        value = latency.get("tpot_s")
        if value is not None and latency.get("tpot_unavailable_reason") is None and record.get("token_ids_complete") is True:
            tpot.append(value)
        else:
            tpot_unavailable += 1
    gap = [
        record["latency"].get("client_chunk_gap_mean_s")
        for record in records
        if record.get("latency", {}).get("client_chunk_gap_mean_s") is not None
    ]
    return {
        "requests": len(records),
        "ttft": summarize_latencies(ttft),
        "e2e": summarize_latencies(e2e),
        "tpot": summarize_latencies(tpot),
        "tpot_available_requests": len(tpot),
        "tpot_unavailable_requests": tpot_unavailable,
        "client_chunk_gap_mean": summarize_latencies(gap),
        "tpot_definition": "visible token_ids only; chunk-gap mean is reported separately",
    }


def measurement_lifecycle_checks(doc: dict[str, Any], side: str = "ar") -> dict[str, Any]:
    """Checks supported by one measure document.

    A socket close plus a global active-gauge return is reported separately
    from a concurrent, request-correlated engine abort.  Only the latter can
    support a full lifecycle claim.
    """

    expected = _expected_window_keys(doc)
    windows, window_errors = _index_windows(doc, side)
    records = doc.get("records", [])
    summary = doc.get("summary") or {}
    cancels = summary.get("cancel_probes") or []
    boundaries = summary.get("boundary_results") or []
    nonstream = summary.get("nonstream_probes") or []
    pair_checks = summary.get("pair_checks") or []
    failures = sum(1 for record in records if not _record_success(record) and not record.get("cancelled"))
    unexpected_cancelled = [
        str(record.get("request_id") or "<missing request_id>")
        for record in records
        if record.get("cancelled") is True
        and not (record.get("measurement_scope") or {}).get("cancelled_probe")
    ]
    window_error_counts = [int(window.get("num_errors") or 0) for window in doc.get("windows", [])]
    window_cancelled_counts = [int(window.get("num_cancelled") or 0) for window in doc.get("windows", [])]
    window_request_count = sum(int(window.get("num_requests") or 0) for window in doc.get("windows", []))
    record_count_ok = bool(records) and window_request_count == len(records)
    cancel_keys = {(item.get("concurrency"), item.get("session")) for item in cancels}
    cancel_reclaimed = bool(
        cancels
        and len(cancels) == len(expected)
        and cancel_keys == expected
        and all(
            item.get("cancelled") is True
            and item.get("observed_in_flight") is True
            and item.get("cancel_reclaimed_s") is not None
            and not item.get("error")
            for item in cancels
        )
    )
    concurrent_cancel = bool(
        cancels
        and all(item.get("concurrent_with_healthy_request") is True for item in cancels)
    )
    internal_abort = bool(
        cancels
        and all(item.get("internal_abort_evidence") is True for item in cancels)
    )
    pair_keys = {(item.get("concurrency"), item.get("session")) for item in pair_checks}
    pair_count_ok = len(pair_checks) == len(expected) and pair_keys == expected
    pair_input_ok = bool(pair_count_ok and all(item.get("input_submission_ok") for item in pair_checks))
    pair_output_verified = bool(
        pair_count_ok and all(item.get("output_attribution_verified") for item in pair_checks)
    )
    checks = {
        "expected_windows": sorted(expected),
        "observed_windows": sorted(windows),
        "window_errors": window_errors,
        "all_windows_present": bool(expected) and expected.issubset(set(windows)),
        "record_errors": failures,
        "unexpected_cancelled_records": unexpected_cancelled,
        "window_error_counts": window_error_counts,
        "window_cancelled_counts": window_cancelled_counts,
        "window_errors_total": sum(window_error_counts),
        "window_cancelled_total": sum(window_cancelled_counts),
        "records": len(records),
        "window_request_count": window_request_count,
        "record_count_ok": record_count_ok,
        "boundary_ok": len(boundaries) == 8 and all(item.get("ok") for item in boundaries),
        "nonstream_ok": len(nonstream) >= 1 and all(item.get("ok") for item in nonstream),
        "cancel_probes": len(cancels),
        "cancel_keys": sorted(cancel_keys),
        "socket_close_reclaimed": cancel_reclaimed,
        "concurrent_with_healthy_request": concurrent_cancel,
        "internal_abort_correlated": internal_abort,
        "cancel_claim_scope": "client_socket_close_and_global_gauge_only"
        if cancel_reclaimed and not internal_abort
        else "full_concurrent_cancel",
        "pair_input_ok": pair_input_ok,
        "pair_output_attribution_verified": pair_output_verified,
        "pair_count_ok": pair_count_ok,
    }
    checks["api_ok"] = bool(
        checks["all_windows_present"]
        and not window_errors
        and record_count_ok
        and failures == 0
        and not unexpected_cancelled
        and not any(window_error_counts)
        and not any(window_cancelled_counts)
        and checks["boundary_ok"]
        and checks["nonstream_ok"]
    )
    checks["full_lifecycle_ok"] = bool(
        checks["api_ok"]
        and cancel_reclaimed
        and concurrent_cancel
        and internal_abort
        and pair_input_ok
        and pair_output_verified
    )
    return checks


def _lifecycle(ar_doc: dict[str, Any], sd_doc: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "ar": measurement_lifecycle_checks(ar_doc, "ar"),
        "ngram": measurement_lifecycle_checks(sd_doc, "ngram"),
    }
    reasons: list[str] = []
    for side in ("ar", "ngram"):
        if not checks[side]["api_ok"]:
            reasons.append(f"{side}_api_checks_failed")
        if not checks[side]["socket_close_reclaimed"]:
            reasons.append(f"{side}_cancel_reclamation_not_observed")
        if not checks[side]["concurrent_with_healthy_request"]:
            reasons.append(f"{side}_cancel_probe_was_idle_not_concurrent")
        if not checks[side]["internal_abort_correlated"]:
            reasons.append(f"{side}_no_request_correlated_engine_abort_evidence")
        if not checks[side]["pair_output_attribution_verified"]:
            reasons.append(f"{side}_same_text_different_image_output_attribution_unverified")
    lifecycle_pass = not reasons
    return {
        "status": "pass" if lifecycle_pass else "pending_internal_evidence",
        "ok": lifecycle_pass,
        "reasons": reasons,
        "ar": checks["ar"],
        "ngram": checks["ngram"],
        "ar_api_pass": bool(checks["ar"]["api_ok"]),
        "cancel_validated": False,
        "cross_session_claim_supported": False,
    }


def build_summary(
    correctness: dict[str, Any],
    ar_doc: dict[str, Any],
    sd_doc: dict[str, Any],
    *,
    manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    protocol = _protocol_check(ar_doc, sd_doc)
    pairs, pair_errors, by_window = _pair_check(ar_doc, sd_doc)
    lifecycle = _lifecycle(ar_doc, sd_doc)
    claim_reasons = list(protocol["reasons"]) + pair_errors
    if not pairs:
        claim_reasons.append("no_paired_request_records")
    if any(item["failures"] for item in by_window.values()):
        claim_reasons.append("paired_windows_contain_failed_or_cancelled_requests")
    if not all(item["budget_verified"] for item in by_window.values()):
        claim_reasons.append("actual_request_budget_not_verified_for_every_pair")
    claim_eligible = not claim_reasons

    ratios = []
    for key, item in sorted(by_window.items()):
        ar_wall = item.get("ar_wall_s")
        sd_wall = item.get("sd_wall_s")
        if ar_wall and sd_wall:
            ratios.append(
                {
                    "concurrency": key[0],
                    "session": key[1],
                    "ar_wall_s": ar_wall,
                    "ngram_wall_s": sd_wall,
                    "ngram_over_ar": ar_wall / sd_wall,
                }
            )
    ratios_by_c: dict[str, dict[str, Any]] = {}
    for concurrency in sorted({item["concurrency"] for item in ratios}):
        items = [item for item in ratios if item["concurrency"] == concurrency]
        ar_total = sum(item["ar_wall_s"] for item in items)
        ngram_total = sum(item["ngram_wall_s"] for item in items)
        ratios_by_c[str(concurrency)] = {
            "pairs": len(items),
            "ar_total_s": ar_total,
            "ngram_total_s": ngram_total,
            "ngram_over_ar": (ar_total / ngram_total) if ngram_total else None,
            "claim_eligible": claim_eligible,
            "raw": items,
        }

    levels = dict(correctness.get("completion_levels", {}))
    levels["AR_API_PASS"] = lifecycle["ar_api_pass"]
    levels["SERVING_LIFECYCLE_PASS"] = lifecycle["ok"]
    levels["serving_lifecycle_status"] = lifecycle["status"]
    levels["SD_CORRECTNESS_PASS"] = bool(levels.get("SD_CORRECTNESS_PASS"))
    levels["SD_EXECUTED"] = bool(levels.get("SD_EXECUTED"))
    levels["AR_OFFLINE_PASS"] = bool(levels.get("AR_OFFLINE_PASS"))

    return {
        "schema_version": SUMMARY_SCHEMA,
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "completion_levels": levels,
        "correctness": {
            "coverage": correctness.get("coverage"),
            "ar_ar": correctness.get("ar_ar"),
            "sd_ar": correctness.get("sd_ar"),
            "categories": correctness.get("categories"),
            "diagnostic": correctness.get("diagnostic"),
            "natural": correctness.get("natural"),
            "failed_ids": correctness.get("failed_ids"),
        },
        "protocol": protocol,
        "pairing": {
            "claim_eligible": claim_eligible,
            "reasons": claim_reasons,
            "pairs": len(pairs),
            "windows": len(by_window),
            "by_window": {
                f"c{key[0]}-s{key[1]}": value for key, value in sorted(by_window.items())
            },
            "request_pairs": pairs,
        },
        "latency": {
            "ar": _latency_summary(ar_doc),
            "ngram": _latency_summary(sd_doc),
        },
        "speedup": {
            "claim_eligible": claim_eligible,
            "reasons": claim_reasons,
            "definition": "ngram_over_ar = AR total wall / N-Gram total wall; below 1 means N-Gram slower",
            "by_concurrency": ratios_by_c,
        },
        "resource": {
            "allocation_gpu_hours_additive": True,
            "ar_gpu_hours": (ar_doc.get("summary") or {}).get("gpu_hours"),
            "ngram_gpu_hours": (sd_doc.get("summary") or {}).get("gpu_hours"),
            "ar_allocation_wall_s": (ar_doc.get("summary") or {}).get("allocation_wall_s"),
            "ngram_allocation_wall_s": (sd_doc.get("summary") or {}).get("allocation_wall_s"),
            "request_gpu_hours_additive": False,
            "legacy_request_gpu_hours_sum": {
                "ar": round(
                    sum(float((record.get("resource_usage") or {}).get("gpu_hours") or 0.0) for record in ar_doc.get("records", [])),
                    6,
                ),
                "ngram": round(
                    sum(float((record.get("resource_usage") or {}).get("gpu_hours") or 0.0) for record in sd_doc.get("records", [])),
                    6,
                ),
            },
            "legacy_request_gpu_hours_note": (
                "request intervals overlap under concurrency; this sum is retained for audit only and is not additive"
            ),
        },
        "lifecycle": lifecycle,
    }
