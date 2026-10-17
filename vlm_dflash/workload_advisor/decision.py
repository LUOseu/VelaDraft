"""Offline recommendation rules.

The rules are deliberately conservative: a speed point estimate never wins.
A candidate needs a verified correctness status, a quality gate (pass or an
explicit condition), memory fit, real per-request measurement and a paired
group-bootstrap CI lower bound above the frozen business threshold.
"""

from __future__ import annotations

from typing import Any

from .schema import RECOMMENDED_ACTIONS, SCHEMA_VERSION, canonical_json, sha256_json


def _reason(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def _candidate_exclusions(candidate: dict, summary: dict) -> list[dict]:
    reasons: list[dict] = []
    gates = candidate.get("gates") or {}
    correctness = candidate.get("correctness") or {}
    status_counts = correctness.get("status_counts") or {}
    if not gates.get("correctness"):
        if status_counts.get("failed"):
            reasons.append(_reason("correctness_failed", f"{status_counts['failed']} record(s) failed."))
        elif status_counts.get("experimental_numeric_divergence"):
            reasons.append(
                _reason(
                    "experimental_numeric_divergence",
                    "candidate diverges numerically from the AR token reference.",
                )
            )
        elif status_counts.get("verified_semantic_acceptance"):
            reasons.append(
                _reason(
                    "semantic_not_token_exact",
                    "semantic acceptance is not native AR token-exact; excluded from acceleration.",
                )
            )
        else:
            reasons.append(
                _reason("correctness_not_verified", f"status counts: {status_counts or 'none'}.")
            )
    if "relaxed" in (candidate.get("acceptance_policies") or []):
        reasons.append(_reason("relaxed_acceptance", "acceptance_policy=relaxed cannot be exact."))
    if (candidate.get("quality") or {}).get("status") == "fail":
        reasons.append(_reason("quality_failed", "task quality is below the frozen threshold."))
    if gates.get("memory") is False:
        reasons.append(_reason("gpu_memory_not_fit", "candidate does not fit the available device memory."))
    if not gates.get("real_measurement"):
        reasons.append(_reason("aggregate_only", "aggregate-only evidence cannot produce per-request speed CIs."))
    if summary.get("protocol_mismatch"):
        reasons.append(_reason("protocol_mismatch", "records mix protocol scope hashes."))
    if candidate.get("scope_hashes") and summary.get("protocol_scope_hashes"):
        if sorted(candidate["scope_hashes"]) != sorted(summary["protocol_scope_hashes"]):
            reasons.append(_reason("protocol_mismatch", "candidate protocol scope differs from the workload protocol."))
    if summary.get("synthetic"):
        reasons.append(_reason("synthetic_fixture", "synthetic fixtures never enter a real recommendation."))
    return reasons


def _bucket_conditional_scope(candidate: dict, threshold: float) -> list[dict]:
    scopes: list[dict] = []
    for axis, buckets in (candidate.get("buckets") or {}).items():
        for label, stats in buckets.items():
            if stats.get("status") == "ok" and stats.get("ci95_low") is not None and stats["ci95_low"] > threshold:
                scopes.append(
                    {
                        "axis": axis,
                        "bucket": label,
                        "n_groups": stats.get("n_groups"),
                        "ci95_low": stats.get("ci95_low"),
                        "ci95_high": stats.get("ci95_high"),
                    }
                )
    return scopes


def decide(
    summary: dict,
    *,
    threshold: float = 1.05,
    bootstrap_seed: int | None = None,
    min_groups: int = 20,
) -> dict:
    threshold = float(summary.get("threshold", threshold))
    bootstrap_seed = int(summary.get("bootstrap_seed", 7) if bootstrap_seed is None else bootstrap_seed)
    min_groups = int(summary.get("min_groups", min_groups))
    candidates = summary.get("candidates") or {}
    ar = candidates.get("ar")
    ar_valid = bool(
        ar
        and (ar.get("gates") or {}).get("correctness")
        and (ar.get("gates") or {}).get("real_measurement")
        and not (ar.get("correctness") or {}).get("failures")
    )
    excluded: list[dict] = []
    eligible: list[tuple[dict, list[dict]]] = []
    conditional_scopes: list[dict] = []
    for candidate_id, candidate in sorted(candidates.items()):
        if candidate_id == "ar":
            continue
        reasons = _candidate_exclusions(candidate, summary)
        speed = candidate.get("speed") or {}
        gates = candidate.get("gates") or {}
        if gates.get("coverage") is False:
            reasons.append(
                _reason(
                    "insufficient_coverage",
                    f"{speed.get('n_groups', 0)} independent group(s) < required {min_groups}.",
                )
            )
        if speed.get("ci95_low") is None:
            reasons.append(_reason("missing_ci", "no paired bootstrap CI was produced."))
        elif speed["ci95_low"] <= threshold:
            reasons.append(
                _reason(
                    "speedup_ci_not_above_threshold",
                    f"CI lower {speed['ci95_low']:.4f} <= threshold {threshold:.4f}; point ratio "
                    f"{speed.get('point_ratio')}.",
                )
            )
        if reasons:
            excluded.append({"candidate_id": candidate_id, "reasons": reasons})
            scopes = _bucket_conditional_scope(candidate, threshold)
            blocking_codes = {
                "correctness_failed",
                "experimental_numeric_divergence",
                "semantic_not_token_exact",
                "correctness_not_verified",
                "relaxed_acceptance",
                "quality_failed",
                "gpu_memory_not_fit",
                "aggregate_only",
                "protocol_mismatch",
                "synthetic_fixture",
            }
            has_blocking = any(reason["code"] in blocking_codes for reason in reasons)
            if scopes and not has_blocking and gates.get("correctness") and gates.get("memory") is not False:
                conditional_scopes.append({"candidate_id": candidate_id, "buckets": scopes})
            continue
        eligible.append((candidate, []))

    action = "insufficient_evidence"
    recommended_method = None
    reasons: list[str] = []
    scope: dict[str, Any] = {
        "measurement_scope": "local_client",
        "threshold": threshold,
        "min_groups": min_groups,
    }
    if summary.get("synthetic"):
        reasons.append("Synthetic fixture evidence: no real recommendation is made.")
    elif summary.get("protocol_mismatch"):
        reasons.append("Records mix protocol hashes; timing and correctness cannot be paired.")
    elif eligible:
        eligible.sort(
            key=lambda item: (
                item[0]["speed"]["ci95_low"],
                item[0]["speed"]["point_ratio"],
            ),
            reverse=True,
        )
        winner, _ = eligible[0]
        recommended_method = winner["candidate_id"]
        quality_status = (winner.get("quality") or {}).get("status")
        conditional = quality_status in ("conditional", "not_evaluated", "unknown")
        action = "conditional_accelerate" if conditional else "accelerate"
        reasons.append(
            f"{recommended_method}: paired speed ratio CI lower "
            f"{winner['speed']['ci95_low']:.4f} > {threshold:.4f} "
            f"(point {winner['speed']['point_ratio']:.4f}, n_groups={winner['speed']['n_groups']})."
        )
        if conditional:
            reasons.append(
                "Condition: business task quality is not fully evaluated; the candidate keeps the AR token output on the measured set."
            )
        scope.update(
            {
                "tasks": winner.get("tasks") or None,
                "target_V_buckets": sorted((winner.get("buckets") or {}).get("target_V", {}).keys()),
                "ar_R_buckets": sorted((winner.get("buckets") or {}).get("ar_R", {}).keys()),
                "n_groups": winner["speed"]["n_groups"],
                "n_sessions": winner.get("n_sessions"),
            }
        )
        for other, _ in eligible[1:]:
            excluded.append(
                {
                    "candidate_id": other["candidate_id"],
                    "reasons": [_reason("lower_confidence_bound", "another candidate has a stronger CI lower bound.")],
                }
            )
    elif conditional_scopes:
        action = "conditional_accelerate"
        recommended_method = conditional_scopes[0]["candidate_id"]
        reasons.append(
            "Global CI does not clear the threshold, but at least one frozen bucket does; "
            "recommendation is restricted to those buckets."
        )
        scope["conditional_buckets"] = conditional_scopes
    elif ar_valid:
        action = "retain_ar"
        reasons.append(
            "No candidate cleared correctness, quality, memory, coverage and CI gates; retain the AR baseline."
        )
    else:
        action = "insufficient_evidence"
        reasons.append("Neither a compliant candidate nor a verified AR baseline is available.")

    if action not in RECOMMENDED_ACTIONS:
        raise ValueError(f"internal error: action {action!r} is not in {RECOMMENDED_ACTIONS}")
    coverage = {
        "n_groups": summary.get("n_groups", 0),
        "n_sessions": summary.get("n_sessions", 0),
        "n_tasks": summary.get("n_tasks", 0),
        "task_group_counts": summary.get("task_group_counts", {}),
        "min_groups": min_groups,
        "buckets_insufficient": _insufficient_buckets(candidates),
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "recommended_action": action,
        "recommended_method": recommended_method,
        "scope": scope,
        "coverage": coverage,
        "threshold": threshold,
        "bootstrap_seed": bootstrap_seed,
        "reasons": reasons,
        "excluded_candidates": excluded,
        "protocol_scope_hashes": summary.get("protocol_scope_hashes", []),
        "protocol_hash": (summary.get("protocol_scope_hashes") or [None])[0],
        "records_sha256": summary.get("records_sha256"),
        "summary_sha256": summary.get("summary_sha256"),
        "synthetic": bool(summary.get("synthetic")),
    }
    decision["decision_id"] = sha256_json(decision)
    return decision


def _insufficient_buckets(candidates: dict) -> dict:
    result: dict[str, list[str]] = {}
    for candidate_id, candidate in candidates.items():
        if candidate_id == "ar":
            continue
        for axis, buckets in (candidate.get("buckets") or {}).items():
            labels = [label for label, stats in buckets.items() if stats.get("status") == "insufficient_evidence"]
            if labels:
                result[f"{candidate_id}:{axis}"] = sorted(labels)
    return result


def decision_json(decision: dict) -> str:
    return canonical_json(decision)
