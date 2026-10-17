"""Paired group statistics and frozen-bucket evidence.

Speed is measured as a paired total-time ratio: for every independent group,
sum the AR session times and the candidate session times (all repeats kept),
then compare ``sum(AR) / sum(candidate)``. The bootstrap resamples whole groups
and keeps every session inside a resampled group.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from .schema import canonical_json, protocol_scope_hash, sha256_json


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
    """Paired group bootstrap for total-time ratio (AR / candidate)."""
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
    ratios = [ar / cand for ar, cand in group_pairs if cand > 0]
    point = sum(ar for ar, _ in group_pairs) / sum(cand for _, cand in group_pairs)
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    rng = random.Random(seed)
    boot_total: list[float] = []
    boot_mean: list[float] = []
    size = len(group_pairs)
    for _ in range(n_boot):
        picks = [rng.randrange(size) for _ in range(size)]
        ar_sum = sum(group_pairs[index][0] for index in picks)
        cand_sum = sum(group_pairs[index][1] for index in picks)
        if cand_sum > 0:
            boot_total.append(ar_sum / cand_sum)
            boot_mean.append(
                sum(group_pairs[index][0] / group_pairs[index][1] for index in picks if group_pairs[index][1] > 0)
                / max(1, sum(1 for index in picks if group_pairs[index][1] > 0))
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


def _latency_summary(records: list[dict]) -> dict:
    e2e = [record["latency"]["e2e_s"] for record in records if record.get("latency", {}).get("e2e_s") is not None]
    ttft = [record["latency"]["ttft_s"] for record in records if record.get("latency", {}).get("ttft_s") is not None]
    tpot = [record["latency"]["tpot_s"] for record in records if record.get("latency", {}).get("tpot_s") is not None]
    paired = [
        (record["latency"]["e2e_s"], record["output_tokens_R"])
        for record in records
        if record.get("latency", {}).get("e2e_s") is not None and record.get("output_tokens_R") is not None
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


def _correctness_summary(records: list[dict]) -> dict:
    statuses: dict[str, int] = defaultdict(int)
    failed: set[str] = set()
    exact_n = 0
    exact_requests = 0
    failed_records = 0
    for record in records:
        statuses[record["correctness_status"]] += 1
        correctness = record.get("correctness") or {}
        failed.update(correctness.get("failed_request_ids") or [])
        if record["correctness_status"] == "failed":
            failed_records += 1
        if record["correctness_status"] == "verified_greedy_exact":
            exact_n += 1
            exact_requests += int(correctness.get("n_requests") or 0)
    return {
        "status_counts": dict(sorted(statuses.items())),
        "exact_records": exact_n,
        "exact_requests": exact_requests,
        "failed_request_ids": sorted(failed),
        "failures": len(failed),
        "failed_records": failed_records,
        "denominator_records": len(records),
        "denominator_requests": sum(int((record.get("correctness") or {}).get("n_requests") or 0) for record in records),
        "denominator": exact_requests,
        "note": "0/N failures retains N; no inference that the failure probability is zero.",
    }


def _quality_summary(records: list[dict]) -> dict:
    evaluated = [record for record in records if (record.get("quality") or {}).get("evaluated")]
    if not evaluated:
        return {
            "evaluated": False,
            "status": "not_evaluated",
            "score_mean": None,
            "thresholds": {},
            "n": 0,
            "reason": "no_labels_or_scorer_output",
        }
    scores = [float(record["quality"]["score"]) for record in evaluated if record["quality"].get("score") is not None]
    statuses = [str(record["quality"].get("status", "unknown")) for record in evaluated]
    worst = "fail" if "fail" in statuses else ("conditional" if "conditional" in statuses else ("pass" if all(s == "pass" for s in statuses) else "unknown"))
    thresholds = {}
    schema_flags = []
    for record in evaluated:
        task = record.get("task")
        if record["quality"].get("threshold") is not None:
            thresholds[task] = record["quality"]["threshold"]
        if record["quality"].get("json_schema_valid") is not None:
            schema_flags.append(bool(record["quality"]["json_schema_valid"]))
    return {
        "evaluated": True,
        "status": worst,
        "score_mean": sum(scores) / len(scores) if scores else None,
        "thresholds": thresholds,
        "n": len(evaluated),
        "json_schema_valid_rate": (sum(schema_flags) / len(schema_flags)) if schema_flags else None,
        "json_schema_n": len(schema_flags),
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


def _group_records(records: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["group_id"]][record["candidate_id"]].append(record)
    return grouped


def _bucket_evidence(
    groups: dict[str, dict[str, list[dict]]],
    candidate_id: str,
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
        for group_id, by_candidate in groups.items():
            candidate_records = by_candidate.get(candidate_id) or []
            ar_records = by_candidate.get("ar") or []
            if not candidate_records or not ar_records:
                continue
            if axis == "task":
                label = str(candidate_records[0].get("task") or "unknown")
            elif axis == "target_V":
                value = (candidate_records[0].get("visual_tokens") or {}).get("target_V")
                label = bucket_label(value, edges)
            else:
                value = ar_records[0].get("output_tokens_R")
                label = bucket_label(value, edges)
            ar_total = sum(record["latency"]["e2e_s"] for record in ar_records if record["latency"].get("e2e_s") is not None)
            cand_total = sum(record["latency"]["e2e_s"] for record in candidate_records if record["latency"].get("e2e_s") is not None)
            if cand_total > 0:
                buckets[label].append((ar_total, cand_total))
        axis_result = {}
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


def summarize(
    records: Iterable[dict],
    *,
    bucket_rules: dict | None = None,
    threshold: float = 1.05,
    bootstrap_seed: int = 7,
    n_boot: int = 2000,
    min_groups: int = 20,
    include_synthetic: bool = False,
) -> dict:
    records = list(records)
    synthetic = [record for record in records if record.get("synthetic")]
    real = [record for record in records if not record.get("synthetic")]
    selected = records if include_synthetic else real
    scope_hashes = sorted({record.get("protocol_scope_hash") for record in selected if record.get("protocol_scope_hash")})
    fallback_records = [
        record
        for record in selected
        if record.get("fallback") or record.get("effective_method") != record.get("requested_method")
    ]
    usable = [
        record
        for record in selected
        if not record.get("fallback") and record.get("effective_method") == record.get("requested_method")
    ]
    groups = _group_records(usable)
    candidate_ids = sorted({record["candidate_id"] for record in usable if record.get("method") != "ar"})
    if any(record.get("method") == "ar" for record in usable):
        candidate_ids.append("ar")
    candidates: dict[str, dict] = {}
    for candidate_id in candidate_ids:
        is_ar = candidate_id == "ar"
        candidate_records = [
            record
            for record in usable
            if (record.get("method") == "ar" if is_ar else record["candidate_id"] == candidate_id)
        ]
        pairs: list[tuple[float, float]] = []
        missing_groups: list[str] = []
        sessions: set[str] = set()
        for group_id, by_candidate in groups.items():
            cand = by_candidate.get(candidate_id) or []
            ar = by_candidate.get("ar") or []
            if is_ar:
                cand = ar
            if not cand:
                continue
            if not ar and not is_ar:
                missing_groups.append(group_id)
                continue
            ar_total = sum(record["latency"]["e2e_s"] for record in ar if record["latency"].get("e2e_s") is not None)
            cand_total = sum(record["latency"]["e2e_s"] for record in cand if record["latency"].get("e2e_s") is not None)
            if cand_total > 0:
                pairs.append((ar_total if not is_ar else cand_total, cand_total))
            sessions.update(str(record.get("session_id")) for record in cand)
        speed = bootstrap_ratio(pairs, seed=bootstrap_seed, n_boot=n_boot)
        speed["method"] = "paired_total_ratio"
        correctness = _correctness_summary(candidate_records)
        quality = _quality_summary(candidate_records)
        resources = _resource_summary(candidate_records)
        gates = {
            "correctness": correctness["status_counts"].get("verified_greedy_exact", 0) == len(candidate_records)
            and bool(candidate_records),
            "quality": quality["status"] in ("pass", "conditional", "not_evaluated"),
            "quality_conditional": quality["status"] == "not_evaluated",
            "memory": resources["memory_fit"] is True,
            "real_measurement": all(record.get("measurement_scope") != "aggregate_only" for record in candidate_records),
            "coverage": speed["n_groups"] >= min_groups,
            "speed_ci_lower_above_threshold": bool(
                speed["ci95_low"] is not None and speed["ci95_low"] > threshold
            ),
            "threshold": threshold,
        }
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "method": "ar" if is_ar else candidate_records[0].get("method") if candidate_records else None,
            "n_records": len(candidate_records),
            "n_groups": speed["n_groups"],
            "n_sessions": len(sessions),
            "missing_pairs": sorted(missing_groups),
            "speed": speed,
            "correctness": correctness,
            "quality": quality,
            "latency": _latency_summary(candidate_records),
            "resources": resources,
            "gates": gates,
            "scope_hashes": sorted({record.get("protocol_scope_hash") for record in candidate_records if record.get("protocol_scope_hash")}),
            "tasks": sorted({str(record.get("task")) for record in candidate_records}),
            "acceptance_policies": sorted(
                {str(record.get("acceptance_policy")) for record in candidate_records if record.get("acceptance_policy")}
            ),
            "measurement_scopes": sorted(
                {str(record.get("measurement_scope")) for record in candidate_records if record.get("measurement_scope")}
            ),
        }
        if bucket_rules and not is_ar:
            candidates[candidate_id]["buckets"] = _bucket_evidence(
                groups, candidate_id, bucket_rules, seed=bootstrap_seed, n_boot=n_boot
            )

    total_groups = len(groups)
    task_counts: dict[str, int] = defaultdict(int)
    for group_id, by_candidate in groups.items():
        first = next(iter(by_candidate.values()), None)
        if first:
            task_counts[str(first[0].get("task"))] += 1
    status = "ok"
    if fallback_records:
        status = "partial"
    if not usable or total_groups == 0:
        status = "insufficient_evidence"
    if len(scope_hashes) > 1:
        status = "insufficient_evidence"
    return {
        "schema_version": "veladraft-platform-v1",
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "protocol_scope_hashes": scope_hashes,
        "protocol_mismatch": len(scope_hashes) > 1,
        "overall_status": status,
        "threshold": threshold,
        "bootstrap_seed": bootstrap_seed,
        "n_boot": n_boot,
        "min_groups": min_groups,
        "n_records": len(selected),
        "n_real_records": len(real),
        "n_synthetic_records": len(synthetic),
        "n_groups": total_groups,
        "n_tasks": len(task_counts),
        "task_group_counts": dict(sorted(task_counts.items())),
        "n_sessions": len({str(record.get("session_id")) for record in usable}),
        "fallback_records": len(fallback_records),
        "candidates": candidates,
        "concurrency": sorted({record.get("concurrency") for record in usable if record.get("concurrency") is not None}),
        "records_sha256": sha256_json(sorted((canonical_json(record) for record in selected), key=lambda text: text)),
    }


def summarize_synthetic(records: Iterable[dict], **kwargs) -> dict:
    """Fixture-only summary; callers must label it synthetic in reports."""
    summary = summarize(records, include_synthetic=True, **kwargs)
    summary["synthetic"] = True
    summary["overall_status"] = "synthetic_fixture"
    return summary
