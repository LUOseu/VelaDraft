"""Strict paired evidence, denominators and frozen-holdout statistics.

The recommendation path never aggregates AR and speculative records in
separate buckets.  Records are first keyed by
``(candidate_id, request_id, session_id, repeat_index)``, checked for duplicate
keys and joined with the corresponding AR record.  A pair contributes timing
only when the request identity, reference/protocol identity and execution
contract all match.  Group bootstrap samples whole groups, preserving every
valid session/repeat inside the sampled group.

Only records whose request is in the frozen ``holdout`` assignment can enter a
recommendation.  Tuning is summarized separately for audit/diagnosis and never
changes a holdout CI or coverage denominator.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from .schema import canonical_json, sha256_json


SPLITS = ("tuning", "holdout")
FAILED_STATUSES = {"failed", "unsupported"}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def bootstrap_ratio(
    group_pairs: list[tuple[float, float]],
    *,
    seed: int,
    n_boot: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """Paired group bootstrap for the total-time ratio (AR / candidate)."""
    if not group_pairs:
        return {
            "point_ratio": None,
            "ci95_low": None,
            "ci95_high": None,
            "mean_of_ratios": None,
            "mean_of_ratios_ci95_low": None,
            "mean_of_ratios_ci95_high": None,
            "n_groups": 0,
            "n_boot": n_boot,
            "bootstrap_seed": seed,
        }
    point = sum(ar for ar, _ in group_pairs) / sum(candidate for _, candidate in group_pairs)
    mean_ratio = sum(ar / candidate for ar, candidate in group_pairs) / len(group_pairs)
    rng = random.Random(seed)
    boot_total: list[float] = []
    boot_mean: list[float] = []
    size = len(group_pairs)
    for _ in range(n_boot):
        picks = [rng.randrange(size) for _ in range(size)]
        ar_sum = sum(group_pairs[index][0] for index in picks)
        candidate_sum = sum(group_pairs[index][1] for index in picks)
        if candidate_sum > 0:
            boot_total.append(ar_sum / candidate_sum)
            boot_mean.append(
                sum(group_pairs[index][0] / group_pairs[index][1] for index in picks) / len(picks)
            )
    alpha = (1.0 - confidence) / 2.0
    return {
        "point_ratio": point,
        "ci95_low": percentile(boot_total, alpha),
        "ci95_high": percentile(boot_total, 1.0 - alpha),
        "mean_of_ratios": mean_ratio,
        "mean_of_ratios_ci95_low": percentile(boot_mean, alpha),
        "mean_of_ratios_ci95_high": percentile(boot_mean, 1.0 - alpha),
        "n_groups": size,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
    }


def bucket_label(value: float | None, edges: list[float]) -> str:
    if value is None:
        return "unknown"
    if not edges:
        return "all"
    if value < edges[0]:
        return f"<{edges[0]:g}"
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return f"{low:g}-{high:g}"
    return f">={edges[-1]:g}"


def _finite_non_negative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _positive_seconds(value: Any) -> bool:
    return _finite_non_negative(value) and value > 0


def _record_key(record: dict) -> tuple[str, str, str, str]:
    repeat = record.get("repeat_index")
    return (
        str(record.get("candidate_id")),
        str(record.get("request_id")),
        str(record.get("session_id")),
        str(repeat),
    )


def _measurement_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("request_id")),
        str(record.get("session_id")),
        str(record.get("repeat_index")),
    )


def _key_payload(key: tuple[str, str, str, str]) -> dict:
    return {
        "candidate_id": key[0],
        "request_id": key[1],
        "session_id": key[2],
        "repeat_index": key[3],
    }


def _request_key_payload(record: dict) -> dict:
    return {
        "request_id": record.get("request_id"),
        "session_id": record.get("session_id"),
        "repeat_index": record.get("repeat_index"),
        "group_id": record.get("group_id"),
    }


def _record_failure(record: dict) -> str | None:
    error = record.get("error")
    if error:
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or error.get("stage") or "error")
        return "error"
    status = record.get("correctness_status")
    if status in FAILED_STATUSES:
        return str(status)
    return None


def _is_fallback(record: dict) -> bool:
    return bool(record.get("fallback")) or record.get("effective_method") != record.get("requested_method")


def _is_ar(record: dict) -> bool:
    return record.get("method") == "ar" or record.get("candidate_id") == "ar"


def _correctness_hash(record: dict, key: str) -> str | None:
    value = (record.get("correctness") or {}).get(key)
    return value if isinstance(value, str) and value else None


def _timing_boundary(record: dict) -> Any:
    return (record.get("measurement") or {}).get("timing_boundary")


def _pair_mismatch(ar: dict, candidate: dict) -> str | None:
    for field in ("request_id", "group_id", "task"):
        if ar.get(field) != candidate.get(field):
            return f"{field}_mismatch"
    if ar.get("protocol_hash") != candidate.get("protocol_hash"):
        return "protocol_hash_mismatch"
    if ar.get("protocol_scope_hash") != candidate.get("protocol_scope_hash"):
        return "protocol_scope_hash_mismatch"
    for field in ("target_identity", "backend_identity"):
        if canonical_json(ar.get(field) or {}) != canonical_json(candidate.get(field) or {}):
            return f"{field}_mismatch"
    if ar.get("measurement_scope") != candidate.get("measurement_scope"):
        return "measurement_scope_mismatch"
    if _timing_boundary(ar) != _timing_boundary(candidate):
        return "timing_boundary_mismatch"
    if ar.get("concurrency") != candidate.get("concurrency"):
        return "concurrency_mismatch"
    if ar.get("output_tokens_R") != candidate.get("output_tokens_R"):
        return "output_tokens_mismatch"
    for field in ("input_sha256", "reference_hash"):
        left = _correctness_hash(ar, field)
        right = _correctness_hash(candidate, field)
        if not left or not right:
            return f"{field}_missing"
        if left != right:
            return f"{field}_mismatch"
    left_visual = (ar.get("visual_tokens") or {}).get("target_V")
    right_visual = (candidate.get("visual_tokens") or {}).get("target_V")
    if left_visual is not None and right_visual is not None and left_visual != right_visual:
        return "target_visual_tokens_mismatch"
    return None


def _assignment_entry(assignment: dict, request_id: str) -> dict:
    raw = assignment.get(request_id)
    if isinstance(raw, str):
        return {"split": raw}
    return raw if isinstance(raw, dict) else {}


def _membership_issues(records: list[dict], assignment: dict | None) -> list[dict]:
    if assignment is None:
        return []
    issues: list[dict] = []
    for record in records:
        request_id = str(record.get("request_id"))
        expected = _assignment_entry(assignment, request_id)
        if not expected:
            issues.append(
                {
                    "code": "record_not_in_assignment",
                    "request_id": request_id,
                    "group_id": record.get("group_id"),
                    "session_id": record.get("session_id"),
                }
            )
            continue
        for field in ("group_id", "task", "split"):
            if expected.get(field) is not None and record.get(field) != expected.get(field):
                issues.append(
                    {
                        "code": f"assignment_{field}_mismatch",
                        "request_id": request_id,
                        "record": field,
                        "observed": record.get(field),
                        "expected": expected.get(field),
                    }
                )
    return issues


def _group_split_issues(records: list[dict]) -> list[dict]:
    seen: dict[str, str] = {}
    issues: list[dict] = []
    for record in records:
        group_id = str(record.get("group_id"))
        split = record.get("split")
        previous = seen.get(group_id)
        if previous is not None and previous != split:
            issues.append(
                {
                    "code": "group_split_mismatch",
                    "group_id": group_id,
                    "observed": split,
                    "previous": previous,
                }
            )
        else:
            seen[group_id] = str(split)
    return issues


def _duplicates(records: list[dict]) -> dict[tuple[str, str, str, str], list[dict]]:
    by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        by_key[_record_key(record)].append(record)
    return {key: values for key, values in by_key.items() if len(values) > 1}


def _empty_pairing() -> dict:
    return {
        "duplicate_records": 0,
        "duplicate_ids": [],
        "missing_ar": [],
        "missing_candidate": [],
        "failed_ar": [],
        "failed_candidate": [],
        "fallback_ar": [],
        "fallback_candidate": [],
        "protocol_mismatch": [],
    }


def _pairing_counts(pairing: dict) -> dict:
    return {key: len(value) for key, value in pairing.items() if isinstance(value, list)}


def _pair_candidate(
    records: list[dict],
    candidate_id: str,
    *,
    duplicate_keys: set[tuple[str, str, str, str]],
) -> tuple[list[dict], dict]:
    """Return complete pairs and the exact exclusion ledger for one candidate."""
    pairing = _empty_pairing()
    ar_by_key: dict[tuple[str, str, str], dict] = {}
    candidate_by_key: dict[tuple[str, str, str], dict] = {}
    for record in records:
        key = _record_key(record)
        if key in duplicate_keys:
            pairing["duplicate_records"] += 1
            pairing["duplicate_ids"].append(_key_payload(key))
            continue
        if _is_ar(record):
            ar_by_key[_measurement_key(record)] = record
        elif record.get("candidate_id") == candidate_id:
            candidate_by_key[_measurement_key(record)] = record

    pairs: list[dict] = []
    for key in sorted(set(ar_by_key) | set(candidate_by_key)):
        ar = ar_by_key.get(key)
        candidate = candidate_by_key.get(key)
        if ar is None:
            pairing["missing_ar"].append(_request_key_payload(candidate))
            continue
        if candidate is None:
            pairing["missing_candidate"].append(_request_key_payload(ar))
            continue
        ar_failure = _record_failure(ar)
        candidate_failure = _record_failure(candidate)
        if ar_failure:
            pairing["failed_ar"].append({**_request_key_payload(ar), "reason": ar_failure})
        if candidate_failure:
            pairing["failed_candidate"].append({**_request_key_payload(candidate), "reason": candidate_failure})
        if ar_failure or candidate_failure:
            continue
        if _is_fallback(ar):
            pairing["fallback_ar"].append(_request_key_payload(ar))
        if _is_fallback(candidate):
            pairing["fallback_candidate"].append(_request_key_payload(candidate))
        if _is_fallback(ar) or _is_fallback(candidate):
            continue
        mismatch = _pair_mismatch(ar, candidate)
        if mismatch:
            pairing["protocol_mismatch"].append({**_request_key_payload(candidate), "reason": mismatch})
            continue
        ar_seconds = (ar.get("latency") or {}).get("e2e_s")
        candidate_seconds = (candidate.get("latency") or {}).get("e2e_s")
        if not _positive_seconds(ar_seconds) or not _positive_seconds(candidate_seconds):
            pairing["failed_ar" if not _positive_seconds(ar_seconds) else "failed_candidate"].append(
                {**_request_key_payload(candidate if _positive_seconds(ar_seconds) else ar), "reason": "incomplete_timing"}
            )
            continue
        pairs.append(
            {
                "ar": ar,
                "candidate": candidate,
                "ar_s": float(ar_seconds),
                "candidate_s": float(candidate_seconds),
                "key": key,
            }
        )
    return pairs, pairing


def _group_pairs(pairs: list[dict]) -> tuple[list[tuple[float, float]], list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_group[str(pair["candidate"].get("group_id"))].append(pair)
    totals: list[tuple[float, float]] = []
    metadata: list[dict] = []
    for group_id in sorted(by_group):
        group = by_group[group_id]
        first_ar = group[0]["ar"]
        first_candidate = group[0]["candidate"]
        ar_total = sum(pair["ar_s"] for pair in group)
        candidate_total = sum(pair["candidate_s"] for pair in group)
        totals.append((ar_total, candidate_total))
        metadata.append(
            {
                "group_id": group_id,
                "ar_s": ar_total,
                "candidate_s": candidate_total,
                "task": first_candidate.get("task"),
                "target_V": (first_candidate.get("visual_tokens") or {}).get("target_V"),
                "ar_R": first_ar.get("output_tokens_R"),
                "pairs": len(group),
            }
        )
    return totals, metadata


def _bucket_evidence(
    group_totals: list[dict],
    bucket_rules: dict,
    *,
    seed: int,
    n_boot: int,
) -> dict:
    min_groups = int(bucket_rules.get("per_bucket_min_groups", 20))
    axes = {
        "task": bucket_rules.get("task_values") or None,
        "target_V": bucket_rules.get("target_V") or [],
        "ar_R": bucket_rules.get("ar_R") or [],
    }
    result: dict[str, dict] = {}
    for axis, edges in axes.items():
        buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for group in group_totals:
            if axis == "task":
                label = str(group.get("task") or "unknown")
            elif axis == "target_V":
                label = bucket_label(group.get("target_V"), edges)
            else:
                label = bucket_label(group.get("ar_R"), edges)
            buckets[label].append((group["ar_s"], group["candidate_s"]))
        axis_result: dict[str, dict] = {}
        for label, pairs in sorted(buckets.items()):
            if len(pairs) < min_groups:
                axis_result[label] = {
                    "n_groups": len(pairs),
                    "status": "insufficient_evidence",
                    "min_groups": min_groups,
                    "point_ratio": None,
                    "ci95_low": None,
                    "ci95_high": None,
                }
            else:
                stats = bootstrap_ratio(pairs, seed=seed, n_boot=n_boot)
                stats["status"] = "ok"
                axis_result[label] = stats
        result[axis] = axis_result
    return result


def _latency_summary(records: list[dict]) -> dict:
    e2e = [record["latency"]["e2e_s"] for record in records if (record.get("latency") or {}).get("e2e_s") is not None]
    ttft = [record["latency"]["ttft_s"] for record in records if (record.get("latency") or {}).get("ttft_s") is not None]
    tpot = [record["latency"]["tpot_s"] for record in records if (record.get("latency") or {}).get("tpot_s") is not None]
    paired = [
        (record["latency"]["e2e_s"], record["output_tokens_R"])
        for record in records
        if (record.get("latency") or {}).get("e2e_s") is not None and record.get("output_tokens_R") is not None
    ]
    serial_tps = (
        (sum(tokens for _, tokens in paired) / len(paired))
        / (sum(seconds for seconds, _ in paired) / len(paired))
        if paired
        else None
    )
    return {
        "e2e_s": {
            "mean": sum(e2e) / len(e2e) if e2e else None,
            "p50": percentile(e2e, 0.5) if e2e else None,
            "p95": percentile(e2e, 0.95) if e2e else None,
            "n": len(e2e),
            "null": len(records) - len(e2e),
        },
        "ttft_s": {
            "mean": sum(ttft) / len(ttft) if ttft else None,
            "p50": percentile(ttft, 0.5) if ttft else None,
            "p95": percentile(ttft, 0.95) if ttft else None,
            "n": len(ttft),
            "null": len(records) - len(ttft),
        },
        "tpot_s": {
            "mean": sum(tpot) / len(tpot) if tpot else None,
            "p50": percentile(tpot, 0.5) if tpot else None,
            "p95": percentile(tpot, 0.95) if tpot else None,
            "n": len(tpot),
            "null": len(records) - len(tpot),
        },
        "itl": "null_no_token_visibility",
        "serial_output_tokens_per_second": serial_tps,
    }


def _resource_summary(records: list[dict]) -> dict:
    allocated = [record["resource_usage"].get("all_allocated_gpu_hours") for record in records]
    measured = [record["resource_usage"].get("measured_run_gpu_hours") for record in records]
    allocated = [value for value in allocated if value is not None]
    measured = [value for value in measured if value is not None]
    peaks: dict[str, int] = {}
    fit_values = []
    for record in records:
        for device in record["resource_usage"].get("devices") or []:
            uuid = device.get("uuid") or device.get("name") or "unknown"
            peak = device.get("peak_allocated_bytes")
            if isinstance(peak, int):
                peaks[uuid] = max(peaks.get(uuid, 0), peak)
        if record["resource_usage"].get("gpu_memory_fit") is not None:
            fit_values.append(bool(record["resource_usage"]["gpu_memory_fit"]))
    return {
        "all_allocated_gpu_hours_max_record": max(allocated) if allocated else None,
        "measured_run_gpu_hours_max_record": max(measured) if measured else None,
        "peak_allocated_bytes": peaks,
        "memory_fit": all(fit_values) if fit_values else None,
    }


def _unique_records_by_request(records: list[dict], expected_requests: set[str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        request_id = str(record.get("request_id"))
        if expected_requests and request_id not in expected_requests:
            continue
        result[request_id].append(record)
    return result


def _correctness_summary(
    records: list[dict],
    *,
    expected_requests: set[str],
    expected_groups: set[str],
    expected_tasks: set[str],
    paired_pairs: list[dict],
    pairing_complete: bool = True,
) -> dict:
    statuses: dict[str, int] = defaultdict(int)
    failed_request_ids: set[str] = set()
    run_error_request_ids: set[str] = set()
    request_ids: set[str] = set()
    for record in records:
        request_id = str(record.get("request_id"))
        request_ids.add(request_id)
        statuses[str(record.get("correctness_status"))] += 1
        if record.get("correctness_status") in FAILED_STATUSES:
            failed_request_ids.add(request_id)
        if record.get("error"):
            run_error_request_ids.add(request_id)
            failed_request_ids.add(request_id)
    paired_requests = {str(pair["candidate"].get("request_id")) for pair in paired_pairs}
    paired_groups = {str(pair["candidate"].get("group_id")) for pair in paired_pairs}
    paired_tasks = {str(pair["candidate"].get("task")) for pair in paired_pairs}
    failed_groups = {
        str(record.get("group_id"))
        for record in records
        if record.get("correctness_status") in FAILED_STATUSES or record.get("error")
    }
    failed_tasks = {
        str(record.get("task"))
        for record in records
        if record.get("correctness_status") in FAILED_STATUSES or record.get("error")
    }
    exact_pairs = 0
    non_exact_pairs = 0
    for pair in paired_pairs:
        ar = pair["ar"]
        candidate = pair["candidate"]
        if ar.get("correctness_status") == "verified_greedy_exact" and candidate.get("correctness_status") == "verified_greedy_exact":
            exact_pairs += 1
        else:
            non_exact_pairs += 1
    denominator_requests = len(expected_requests) if expected_requests else len(request_ids)
    denominator_groups = len(expected_groups)
    denominator_tasks = len(expected_tasks)
    complete_denominators = bool(expected_requests and expected_groups and expected_tasks)
    gate = bool(
        complete_denominators
        and paired_requests == expected_requests
        and not failed_request_ids
        and not run_error_request_ids
        and exact_pairs == len(paired_pairs)
        and len(paired_pairs) > 0
        and pairing_complete
    )
    return {
        "status_counts": dict(sorted(statuses.items())),
        "exact_records": statuses.get("verified_greedy_exact", 0),
        "exact_pairs": exact_pairs,
        "non_exact_pairs": non_exact_pairs,
        "n_records": len(records),
        "n_requests": len(request_ids),
        "n_paired_requests": len(paired_requests),
        "n_paired_groups": len(paired_groups),
        "n_paired_tasks": len(paired_tasks),
        "failed_request_ids": sorted(failed_request_ids),
        "failed_group_ids": sorted(failed_groups),
        "failed_task_ids": sorted(failed_tasks),
        "run_error_request_ids": sorted(run_error_request_ids),
        "run_error_count": len(run_error_request_ids),
        "run_error_rate": (len(run_error_request_ids) / denominator_requests) if denominator_requests else None,
        "denominators": {
            "requests": denominator_requests,
            "groups": denominator_groups,
            "tasks": denominator_tasks,
        },
        "paired_coverage": {
            "requests": len(paired_requests),
            "groups": len(paired_groups),
            "tasks": len(paired_tasks),
        },
        "gate": gate,
        "pairing_complete": pairing_complete,
        "note": "Exactness is supported only for complete AR/candidate pairs; zero over N keeps N.",
    }


def _quality_summary(
    records: list[dict],
    *,
    expected_requests: set[str],
    expected_groups_by_request: dict[str, str],
    expected_tasks: set[str],
    expected_labels: set[str] | None,
) -> dict:
    by_request = _unique_records_by_request(records, expected_requests)
    evaluated_requests: dict[str, dict] = {}
    for request_id, request_records in by_request.items():
        if expected_labels is not None and request_id not in expected_labels:
            continue
        evaluated = [record for record in request_records if (record.get("quality") or {}).get("evaluated")]
        if not evaluated:
            continue
        statuses = [str((record.get("quality") or {}).get("status", "unknown")) for record in evaluated]
        scores = [
            float(record["quality"]["score"])
            for record in evaluated
            if (record.get("quality") or {}).get("score") is not None
        ]
        thresholds = [
            float(record["quality"]["threshold"])
            for record in evaluated
            if (record.get("quality") or {}).get("threshold") is not None
        ]
        schema_flags = [
            bool(record["quality"]["json_schema_valid"])
            for record in evaluated
            if (record.get("quality") or {}).get("json_schema_valid") is not None
        ]
        score = min(scores) if scores else None
        threshold = max(thresholds) if thresholds else None
        failed = "fail" in statuses or any(status == "fail" for status in statuses)
        if score is not None and threshold is not None and score < threshold:
            failed = True
        evaluated_requests[request_id] = {
            "status": "fail" if failed else ("conditional" if "conditional" in statuses else "pass"),
            "score": score,
            "threshold": threshold,
            "json_schema_valid": all(schema_flags) if schema_flags else None,
            "group_id": str(request_records[0].get("group_id")),
            "task": str(request_records[0].get("task")),
        }
    evaluated_group_ids = {
        item["group_id"] for item in evaluated_requests.values() if item["group_id"] in expected_groups_by_request.values()
    }
    evaluated_task_ids = {item["task"] for item in evaluated_requests.values() if item["task"] in expected_tasks}
    denominator_requests = len(expected_requests) if expected_requests else len(by_request)
    denominator_groups = len(set(expected_groups_by_request.values()))
    denominator_tasks = len(expected_tasks)
    labels_verified = expected_labels is not None
    complete = bool(
        labels_verified
        and denominator_requests
        and len(evaluated_requests) == denominator_requests
        and len(evaluated_group_ids) == denominator_groups
        and len(evaluated_task_ids) == denominator_tasks
    )
    statuses = [item["status"] for item in evaluated_requests.values()]
    if not evaluated_requests:
        status = "not_evaluated"
        reason = "no_labels_or_scorer_output"
    elif "fail" in statuses:
        status = "fail"
        reason = "one_or_more_holdout_requests_failed_quality"
    elif complete and "conditional" not in statuses:
        status = "pass"
        reason = None
    else:
        status = "conditional"
        reason = "partial_label_or_scorer_coverage"
    scores = [item["score"] for item in evaluated_requests.values() if item["score"] is not None]
    schema_flags = [
        item["json_schema_valid"] for item in evaluated_requests.values() if item["json_schema_valid"] is not None
    ]
    thresholds = {}
    for item in evaluated_requests.values():
        if item["threshold"] is not None:
            thresholds[item["task"]] = item["threshold"]
    result = {
        "evaluated": bool(evaluated_requests),
        "status": status,
        "score_mean": sum(scores) / len(scores) if scores else None,
        "thresholds": thresholds,
        "n": len(evaluated_requests),
        "n_records": len(records),
        "labels_verified": labels_verified,
        "coverage": {
            "requests": {"evaluated": len(evaluated_requests), "denominator": denominator_requests},
            "groups": {"evaluated": len(evaluated_group_ids), "denominator": denominator_groups},
            "tasks": {"evaluated": len(evaluated_task_ids), "denominator": denominator_tasks},
        },
        "json_schema_valid_rate": (sum(bool(flag) for flag in schema_flags) / len(schema_flags)) if schema_flags else None,
        "json_schema_n": len(schema_flags),
    }
    if reason:
        result["reason"] = reason
    return result


def _expected_assignment_parts(assignment: dict | None) -> tuple[set[str], set[str], set[str], dict[str, str]]:
    requests: set[str] = set()
    groups: set[str] = set()
    tasks: set[str] = set()
    groups_by_request: dict[str, str] = {}
    for request_id, raw in (assignment or {}).items():
        entry = _assignment_entry(assignment or {}, request_id)
        if entry.get("split") != "holdout":
            continue
        requests.add(str(request_id))
        group_id = str(entry.get("group_id"))
        groups_by_request[str(request_id)] = group_id
        if entry.get("group_id") is not None:
            groups.add(group_id)
        if entry.get("task") is not None:
            tasks.add(str(entry.get("task")))
    return requests, groups, tasks, groups_by_request


def _tuning_speed(records: list[dict], candidate_id: str, duplicate_keys: set[tuple[str, str, str, str]]) -> dict:
    pairs, pairing = _pair_candidate(records, candidate_id, duplicate_keys=duplicate_keys)
    totals, _ = _group_pairs(pairs)
    return {
        "n_records": sum(1 for record in records if record.get("candidate_id") == candidate_id),
        "n_pairs": len(pairs),
        "n_groups": len(totals),
        "point_ratio": (sum(ar for ar, _ in totals) / sum(candidate for _, candidate in totals)) if totals else None,
        "pairing": {"counts": _pairing_counts(pairing)},
    }


def summarize(
    records: Iterable[dict],
    *,
    bucket_rules: dict | None = None,
    threshold: float = 1.05,
    bootstrap_seed: int = 7,
    n_boot: int = 2000,
    min_groups: int = 20,
    include_synthetic: bool = False,
    assignment: dict[str, dict] | None = None,
    expected_labels: Iterable[str] | None = None,
    manifest_scope_hash: str | None = None,
    manifest: dict | None = None,
    hard_issues: list[dict] | None = None,
) -> dict:
    records = list(records)
    synthetic = [record for record in records if record.get("synthetic")]
    real = [record for record in records if not record.get("synthetic")]
    selected = records if include_synthetic else real
    hard_issues = list(hard_issues or [])
    assignment_verified = isinstance(assignment, dict) and bool(assignment)
    expected_requests, expected_groups, expected_tasks, groups_by_request = _expected_assignment_parts(assignment)
    labels = set(expected_labels) if expected_labels is not None else None

    membership_issues = _membership_issues(selected, assignment)
    split_issues = _group_split_issues(selected)
    hard_issues.extend(membership_issues)
    hard_issues.extend(split_issues)

    holdout_records = [record for record in selected if record.get("split") == "holdout"] if assignment_verified else []
    tuning_records = [record for record in selected if record.get("split") == "tuning"] if assignment_verified else []
    holdout_duplicates = _duplicates(holdout_records)
    tuning_duplicates = _duplicates(tuning_records)

    observed_scopes = sorted(
        {record.get("protocol_scope_hash") for record in selected if record.get("protocol_scope_hash")}
    )
    protocol_mismatch = len(observed_scopes) > 1
    if manifest_scope_hash is not None and observed_scopes != [manifest_scope_hash]:
        protocol_mismatch = True
        hard_issues.append(
            {
                "code": "protocol_mismatch",
                "message": f"records protocol_scope_hash {observed_scopes} != manifest {manifest_scope_hash}",
            }
        )

    fallback_records = [record for record in selected if _is_fallback(record)]
    candidate_ids: set[str] = set()
    for record in selected:
        if not _is_ar(record) and record.get("candidate_id"):
            candidate_ids.add(str(record["candidate_id"]))
    for candidate in (manifest or {}).get("draft_candidates") or []:
        if isinstance(candidate, dict) and candidate.get("method") != "ar" and candidate.get("candidate_id"):
            candidate_ids.add(str(candidate["candidate_id"]))

    candidates: dict[str, dict] = {}
    total_pairing = _empty_pairing()
    for candidate_id in sorted(candidate_ids):
        candidate_records = [record for record in holdout_records if record.get("candidate_id") == candidate_id]
        pairs, pairing = _pair_candidate(
            holdout_records, candidate_id, duplicate_keys=set(holdout_duplicates)
        )
        for key, value in pairing.items():
            if isinstance(value, list):
                total_pairing[key].extend(value)
            else:
                total_pairing[key] += value
        group_totals, group_metadata = _group_pairs(pairs)
        speed = bootstrap_ratio(group_totals, seed=bootstrap_seed, n_boot=n_boot)
        speed["method"] = "paired_total_ratio"
        speed["n_pairs"] = len(pairs)
        correctness = _correctness_summary(
            candidate_records,
            expected_requests=expected_requests,
            expected_groups=expected_groups,
            expected_tasks=expected_tasks,
            paired_pairs=pairs,
            pairing_complete=not any(
                pairing[key]
                for key in (
                    "missing_ar",
                    "missing_candidate",
                    "failed_ar",
                    "failed_candidate",
                    "fallback_ar",
                    "fallback_candidate",
                    "protocol_mismatch",
                    "duplicate_ids",
                )
            ),
        )
        quality = _quality_summary(
            candidate_records,
            expected_requests=expected_requests,
            expected_groups_by_request=groups_by_request,
            expected_tasks=expected_tasks,
            expected_labels=labels,
        )
        resources = _resource_summary(candidate_records)
        real_measurement = bool(candidate_records) and all(
            record.get("measurement_scope") != "aggregate_only" for record in candidate_records
        )
        gates = {
            "assignment_verified": assignment_verified,
            "correctness": correctness["gate"],
            "run_errors": correctness["run_error_count"] == 0 and bool(candidate_records),
            "quality": quality["status"] not in ("fail",),
            "quality_pass": quality["status"] == "pass",
            "quality_conditional": quality["status"] in ("conditional", "not_evaluated"),
            "memory": resources["memory_fit"],
            "real_measurement": real_measurement,
            "coverage": speed["n_groups"] >= min_groups,
            "speed_ci_lower_above_threshold": bool(
                speed["ci95_low"] is not None and speed["ci95_low"] > threshold
            ),
            "threshold": threshold,
        }
        candidate = {
            "candidate_id": candidate_id,
            "method": candidate_records[0].get("method") if candidate_records else None,
            "n_records": len(candidate_records),
            "n_requests": len({record.get("request_id") for record in candidate_records}),
            "n_groups": speed["n_groups"],
            "n_pairs": len(pairs),
            "n_sessions": len({str(pair["candidate"].get("session_id")) for pair in pairs}),
            "missing_pairs": sorted(
                {
                    str(item.get("request_id"))
                    for item in pairing["missing_ar"] + pairing["missing_candidate"]
                }
            ),
            "speed": speed,
            "correctness": correctness,
            "quality": quality,
            "latency": _latency_summary(candidate_records),
            "resources": resources,
            "gates": gates,
            "pairing": {
                **_pairing_counts(pairing),
                "missing_ar": pairing["missing_ar"],
                "missing_candidate": pairing["missing_candidate"],
                "failed_ar": pairing["failed_ar"],
                "failed_candidate": pairing["failed_candidate"],
                "fallback_ar": pairing["fallback_ar"],
                "fallback_candidate": pairing["fallback_candidate"],
                "protocol_mismatch": pairing["protocol_mismatch"],
                "duplicate_ids": pairing["duplicate_ids"],
            },
            "scope_hashes": sorted(
                {record.get("protocol_scope_hash") for record in candidate_records if record.get("protocol_scope_hash")}
            ),
            "tasks": sorted({str(record.get("task")) for record in candidate_records}),
            "acceptance_policies": sorted(
                {str(record.get("acceptance_policy")) for record in candidate_records if record.get("acceptance_policy")}
            ),
            "measurement_scopes": sorted(
                {str(record.get("measurement_scope")) for record in candidate_records if record.get("measurement_scope")}
            ),
            "tuning": _tuning_speed(tuning_records, candidate_id, duplicate_keys=set(tuning_duplicates)),
        }
        if bucket_rules:
            candidate["buckets"] = _bucket_evidence(
                group_metadata, bucket_rules, seed=bootstrap_seed, n_boot=n_boot
            )
        candidates[candidate_id] = candidate

    ar_records = [record for record in holdout_records if _is_ar(record)]
    ar_groups = {str(record.get("group_id")) for record in ar_records}
    ar_speed = {
        "point_ratio": 1.0 if ar_groups else None,
        "ci95_low": 1.0 if ar_groups else None,
        "ci95_high": 1.0 if ar_groups else None,
        "n_groups": len(ar_groups),
        "n_pairs": len(ar_records),
        "n_boot": n_boot,
        "bootstrap_seed": bootstrap_seed,
        "method": "baseline",
    }
    ar_correctness = _correctness_summary(
        ar_records,
        expected_requests=expected_requests,
        expected_groups=expected_groups,
        expected_tasks=expected_tasks,
        paired_pairs=[],
    )
    ar_paired_requests = {str(record.get("request_id")) for record in ar_records}
    ar_correctness["gate"] = bool(
        assignment_verified
        and expected_requests
        and ar_paired_requests == expected_requests
        and not ar_correctness["failed_request_ids"]
        and not ar_correctness["run_error_request_ids"]
        and ar_correctness["exact_records"] == len(ar_records)
        and ar_records
    )
    ar_correctness["n_paired_requests"] = len(ar_paired_requests)
    ar_quality = _quality_summary(
        ar_records,
        expected_requests=expected_requests,
        expected_groups_by_request=groups_by_request,
        expected_tasks=expected_tasks,
        expected_labels=labels,
    )
    ar_resources = _resource_summary(ar_records)
    candidates["ar"] = {
        "candidate_id": "ar",
        "method": "ar",
        "n_records": len(ar_records),
        "n_requests": len(ar_paired_requests),
        "n_groups": len(ar_groups),
        "n_pairs": len(ar_records),
        "n_sessions": len({str(record.get("session_id")) for record in ar_records}),
        "missing_pairs": [],
        "speed": ar_speed,
        "correctness": ar_correctness,
        "quality": ar_quality,
        "latency": _latency_summary(ar_records),
        "resources": ar_resources,
        "gates": {
            "assignment_verified": assignment_verified,
            "correctness": ar_correctness["gate"],
            "run_errors": ar_correctness["run_error_count"] == 0 and bool(ar_records),
            "quality": ar_quality["status"] not in ("fail",),
            "memory": ar_resources["memory_fit"],
            "real_measurement": bool(ar_records)
            and all(record.get("measurement_scope") != "aggregate_only" for record in ar_records),
            "coverage": len(ar_groups) >= min_groups,
            "threshold": threshold,
        },
        "pairing": _pairing_counts(_empty_pairing()),
        "scope_hashes": sorted(
            {record.get("protocol_scope_hash") for record in ar_records if record.get("protocol_scope_hash")}
        ),
        "tasks": sorted({str(record.get("task")) for record in ar_records}),
        "acceptance_policies": sorted(
            {str(record.get("acceptance_policy")) for record in ar_records if record.get("acceptance_policy")}
        ),
        "measurement_scopes": sorted(
            {str(record.get("measurement_scope")) for record in ar_records if record.get("measurement_scope")}
        ),
        "tuning": _tuning_speed(tuning_records, "ar", duplicate_keys=set(tuning_duplicates)),
    }

    task_counts: dict[str, int] = defaultdict(int)
    if assignment_verified:
        for request_id in expected_requests:
            task = _assignment_entry(assignment or {}, request_id).get("task")
            if task is not None:
                task_counts[str(task)] += 1
    else:
        first_by_group: dict[str, dict] = {}
        for record in holdout_records:
            first_by_group.setdefault(str(record.get("group_id")), record)
        for record in first_by_group.values():
            task_counts[str(record.get("task"))] += 1

    pairing_failed = any(
        total_pairing[key] for key in ("failed_ar", "failed_candidate", "fallback_ar", "fallback_candidate",
                                      "protocol_mismatch", "missing_ar", "missing_candidate", "duplicate_ids")
    )
    usable_pairs = sum(candidate["n_pairs"] for candidate_id, candidate in candidates.items() if candidate_id != "ar")
    if hard_issues:
        status = "failed"
    elif not assignment_verified:
        status = "insufficient_evidence"
    elif not expected_requests or not holdout_records:
        status = "insufficient_evidence"
    elif protocol_mismatch:
        status = "insufficient_evidence"
    elif pairing_failed or fallback_records:
        status = "partial"
    elif usable_pairs == 0:
        status = "insufficient_evidence"
    else:
        status = "ok"

    summary = {
        "schema_version": "veladraft-platform-v1",
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "evidence_split": "holdout",
        "assignment_verified": assignment_verified,
        "assignment_sha256": sha256_json(assignment) if assignment_verified else None,
        "protocol_scope_hashes": observed_scopes,
        "protocol_mismatch": protocol_mismatch,
        "input_integrity": {
            "status": "failed" if hard_issues else "ok",
            "issues": hard_issues,
        },
        "overall_status": status,
        "threshold": threshold,
        "bootstrap_seed": bootstrap_seed,
        "n_boot": n_boot,
        "min_groups": min_groups,
        "n_records": len(selected),
        "n_real_records": len(real),
        "n_synthetic_records": len(synthetic),
        "n_holdout_records": len(holdout_records),
        "n_tuning_records": len(tuning_records),
        "n_groups": len(expected_groups) if assignment_verified else len({str(r.get("group_id")) for r in holdout_records}),
        "n_expected_requests": len(expected_requests),
        "n_tasks": len(expected_tasks) if assignment_verified else len(task_counts),
        "task_group_counts": dict(sorted(task_counts.items())),
        "n_sessions": len({str(record.get("session_id")) for record in holdout_records}),
        "fallback_records": len(fallback_records),
        "duplicate_records": len(holdout_duplicates) + len(tuning_duplicates),
        "pairing": {
            **_pairing_counts(total_pairing),
            "duplicate_ids": total_pairing["duplicate_ids"],
            "missing_ar": total_pairing["missing_ar"],
            "missing_candidate": total_pairing["missing_candidate"],
            "failed_ar": total_pairing["failed_ar"],
            "failed_candidate": total_pairing["failed_candidate"],
            "fallback_ar": total_pairing["fallback_ar"],
            "fallback_candidate": total_pairing["fallback_candidate"],
            "protocol_mismatch": total_pairing["protocol_mismatch"],
        },
        "candidate_ids": sorted(candidate for candidate in candidates if candidate != "ar"),
        "frozen_candidates_sha256": sha256_json((manifest or {}).get("draft_candidates") or sorted(candidate_ids)),
        "candidates": candidates,
        "concurrency": sorted({record.get("concurrency") for record in holdout_records if record.get("concurrency") is not None}),
        "records_sha256": sha256_json(sorted((canonical_json(record) for record in selected), key=lambda text: text)),
    }
    return summary


def summarize_synthetic(records: Iterable[dict], **kwargs) -> dict:
    """Fixture-only summary; callers must label it synthetic in reports."""
    summary = summarize(records, include_synthetic=True, **kwargs)
    summary["synthetic"] = True
    summary["overall_status"] = "synthetic_fixture"
    return summary
