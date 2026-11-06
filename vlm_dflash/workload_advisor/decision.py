"""Conservative offline recommendation rules.

A speed point estimate never wins.  A candidate is recommendable only from a
verified frozen holdout with complete AR/candidate pairs, token-exact output,
no run errors, memory fit and a paired group-bootstrap CI lower bound above the
frozen threshold.  Business quality may be ``pass`` or explicitly
``conditional``; missing labels can never be reported as a quality pass.
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
    quality = candidate.get("quality") or {}
    status_counts = correctness.get("status_counts") or {}

    if summary.get("input_integrity", {}).get("status") == "failed":
        reasons.append(_reason("input_integrity_failed", "Input validation or frozen-assignment checks failed."))
    if not summary.get("assignment_verified"):
        reasons.append(_reason("assignment_missing", "The frozen holdout request/split assignment is unavailable."))
    if summary.get("overall_status") == "failed":
        reasons.append(_reason("hard_failure", "The report contains a hard input/protocol failure."))
    if summary.get("protocol_mismatch"):
        reasons.append(_reason("protocol_mismatch", "Records mix protocol scopes or disagree with the manifest."))
    if summary.get("synthetic"):
        reasons.append(_reason("synthetic_fixture", "Synthetic fixtures never enter a real recommendation."))

    if not gates.get("correctness"):
        if correctness.get("failed_request_ids"):
            reasons.append(
                _reason(
                    "correctness_failed",
                    f"{len(correctness['failed_request_ids'])} holdout request(s) failed exactness.",
                )
            )
        elif status_counts.get("experimental_numeric_divergence"):
            reasons.append(
                _reason("experimental_numeric_divergence", "Candidate diverges numerically from the AR token reference.")
            )
        elif status_counts.get("verified_semantic_acceptance"):
            reasons.append(
                _reason("semantic_not_token_exact", "Semantic acceptance is not native AR token-exact.")
            )
        elif (correctness.get("paired_coverage") or {}).get("requests") != (
            (correctness.get("denominators") or {}).get("requests")
        ):
            reasons.append(
                _reason(
                    "correctness_coverage_incomplete",
                    "Complete AR/candidate pairs do not cover every frozen holdout request.",
                )
            )
        else:
            reasons.append(_reason("correctness_not_verified", f"Status counts: {status_counts or 'none'}."))
    if not gates.get("run_errors"):
        reasons.append(
            _reason(
                "run_errors",
                f"{correctness.get('run_error_count', 0)} request(s) have recorded run errors.",
            )
        )
    if "relaxed" in (candidate.get("acceptance_policies") or []):
        reasons.append(_reason("relaxed_acceptance", "acceptance_policy=relaxed cannot establish token exactness."))
    if quality.get("status") == "fail":
        reasons.append(_reason("quality_failed", "Task quality is below the frozen threshold."))
    if gates.get("memory") is False:
        reasons.append(_reason("gpu_memory_not_fit", "Candidate does not fit the available device memory."))
    elif gates.get("memory") is not True:
        reasons.append(_reason("gpu_memory_not_verified", "Memory fit is not verified for every candidate record."))
    if not gates.get("real_measurement"):
        reasons.append(_reason("aggregate_only", "Aggregate-only evidence cannot produce per-request speed CIs."))
    if not gates.get("assignment_verified"):
        reasons.append(_reason("assignment_missing", "Candidate evidence is not tied to a frozen assignment."))
    if candidate.get("scope_hashes") and summary.get("protocol_scope_hashes"):
        if sorted(candidate["scope_hashes"]) != sorted(summary["protocol_scope_hashes"]):
            reasons.append(
                _reason("protocol_mismatch", "Candidate protocol scope differs from the workload protocol.")
            )
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


def _blocking_without_speed(candidate: dict, summary: dict) -> bool:
    gates = candidate.get("gates") or {}
    quality = candidate.get("quality") or {}
    return bool(
        summary.get("input_integrity", {}).get("status") == "failed"
        or not summary.get("assignment_verified")
        or summary.get("overall_status") == "failed"
        or summary.get("protocol_mismatch")
        or summary.get("synthetic")
        or not gates.get("correctness")
        or not gates.get("run_errors")
        or quality.get("status") == "fail"
        or gates.get("memory") is not True
        or not gates.get("real_measurement")
        or "relaxed" in (candidate.get("acceptance_policies") or [])
    )


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
    ar = candidates.get("ar") or {}
    ar_gates = ar.get("gates") or {}
    ar_correctness = ar.get("correctness") or {}
    ar_valid = bool(
        summary.get("assignment_verified")
        and summary.get("input_integrity", {}).get("status") != "failed"
        and not summary.get("protocol_mismatch")
        and not summary.get("synthetic")
        and ar
        and ar_gates.get("correctness")
        and ar_gates.get("run_errors")
        and ar_gates.get("coverage")
        and ar_gates.get("real_measurement")
        and not ar_correctness.get("failed_request_ids")
        and not ar_correctness.get("run_error_request_ids")
    )

    excluded: list[dict] = []
    eligible: list[dict] = []
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
                    f"{speed.get('n_groups', 0)} complete holdout group(s) < required {min_groups}.",
                )
            )
        if speed.get("ci95_low") is None:
            reasons.append(_reason("missing_ci", "No paired holdout bootstrap CI was produced."))
        elif speed["ci95_low"] <= threshold:
            reasons.append(
                _reason(
                    "speedup_ci_not_above_threshold",
                    f"CI lower {speed['ci95_low']:.4f} <= threshold {threshold:.4f}; "
                    f"point ratio {speed.get('point_ratio')}.",
                )
            )
        if reasons:
            excluded.append({"candidate_id": candidate_id, "reasons": reasons})
            scopes = _bucket_conditional_scope(candidate, threshold)
            if scopes and not _blocking_without_speed(candidate, summary) and gates.get("correctness"):
                conditional_scopes.append({"candidate_id": candidate_id, "buckets": scopes})
            continue
        eligible.append(candidate)

    action = "insufficient_evidence"
    recommended_method = None
    reasons: list[str] = []
    scope: dict[str, Any] = {
        "measurement_scope": "local_client",
        "threshold": threshold,
        "min_groups": min_groups,
    }
    if summary.get("input_integrity", {}).get("status") == "failed":
        reasons.append("Hard input/protocol validation failed; no recommendation is produced.")
    elif not summary.get("assignment_verified"):
        reasons.append("No verifiable frozen holdout assignment is available; evidence is insufficient.")
    elif summary.get("synthetic"):
        reasons.append("Synthetic fixture evidence: no real recommendation is made.")
    elif summary.get("protocol_mismatch"):
        reasons.append("Records mix protocol hashes or disagree with the manifest; timing and correctness are not pairable.")
    elif eligible:
        eligible.sort(
            key=lambda candidate: (
                bool((candidate.get("quality") or {}).get("status") == "pass"),
                candidate["speed"]["ci95_low"],
                candidate["speed"]["point_ratio"],
            ),
            reverse=True,
        )
        winner = eligible[0]
        recommended_method = winner["candidate_id"]
        quality_status = (winner.get("quality") or {}).get("status")
        conditional = quality_status in ("conditional", "not_evaluated", "unknown")
        action = "conditional_accelerate" if conditional else "accelerate"
        reasons.append(
            f"{recommended_method}: paired holdout speed ratio CI lower "
            f"{winner['speed']['ci95_low']:.4f} > {threshold:.4f} "
            f"(point {winner['speed']['point_ratio']:.4f}, n_groups={winner['speed']['n_groups']})."
        )
        if conditional:
            reasons.append(
                "Condition: business task quality is not fully evaluated over every frozen holdout "
                "request/group/task; the candidate keeps the AR token output on the measured set."
            )
        scopes = winner.get("measurement_scopes") or []
        scope.update(
            {
                "measurement_scope": scopes[0] if len(scopes) == 1 else "mixed",
                "tasks": winner.get("tasks") or None,
                "target_V_buckets": sorted((winner.get("buckets") or {}).get("target_V", {}).keys()),
                "ar_R_buckets": sorted((winner.get("buckets") or {}).get("ar_R", {}).keys()),
                "n_groups": winner["speed"]["n_groups"],
                "n_sessions": winner.get("n_sessions"),
            }
        )
        for other in eligible[1:]:
            excluded.append(
                {
                    "candidate_id": other["candidate_id"],
                    "reasons": [_reason("lower_confidence_bound", "Another frozen candidate has a stronger CI lower bound.")],
                }
            )
    elif conditional_scopes:
        action = "conditional_accelerate"
        recommended_method = conditional_scopes[0]["candidate_id"]
        reasons.append(
            "Global holdout CI does not clear the threshold, but at least one frozen bucket does; "
            "the recommendation is restricted to those buckets."
        )
        scope["conditional_buckets"] = conditional_scopes
    elif ar_valid:
        action = "retain_ar"
        reasons.append(
            "No candidate cleared complete-pair correctness, run-error, quality, memory, coverage and CI gates; "
            "retain the verified AR baseline."
        )
    else:
        action = "insufficient_evidence"
        reasons.append("Neither a compliant candidate nor a verified AR holdout baseline is available.")

    if action not in RECOMMENDED_ACTIONS:
        raise ValueError(f"internal error: action {action!r} is not in {RECOMMENDED_ACTIONS}")
    coverage = {
        "n_groups": summary.get("n_groups", 0),
        "n_expected_requests": summary.get("n_expected_requests", 0),
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
        "assignment_verified": bool(summary.get("assignment_verified")),
        "assignment_sha256": summary.get("assignment_sha256"),
        "input_integrity_status": (summary.get("input_integrity") or {}).get("status"),
        "reasons": reasons,
        "excluded_candidates": excluded,
        "protocol_scope_hashes": summary.get("protocol_scope_hashes", []),
        "protocol_hash": (summary.get("protocol_scope_hashes") or [None])[0],
        "records_sha256": summary.get("records_sha256"),
        "frozen_candidates_sha256": summary.get("frozen_candidates_sha256"),
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
