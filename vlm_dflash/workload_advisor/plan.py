"""Dry-run planning: capabilities, request budget and calibration needs.

A dry run never predicts a speed ranking. Before a real measurement the speed
estimate and GPU-hour prediction stay null and the required calibration is
listed explicitly.
"""

from __future__ import annotations

from typing import Any

from .schema import SCHEMA_VERSION, protocol_payload, protocol_scope_hash


def build_plan(
    manifest: dict,
    *,
    validation: dict | None = None,
    readiness: dict | None = None,
    budget_gpu_hours: float = 4.0,
    sessions: int = 3,
    warmups_per_candidate_session: int = 4,
) -> dict:
    counts = (validation or {}).get("counts") or {}
    groups_by_split = counts.get("groups_by_split") or {}
    requests_by_split = counts.get("requests_by_split") or {}
    holdout_requests = int(requests_by_split.get("holdout", 0))
    tuning_requests = int(requests_by_split.get("tuning", 0))
    candidates = list(manifest.get("draft_candidates") or [])
    candidate_matrix = []
    blocked_candidates = []
    for candidate in candidates:
        entry = {
            "candidate_id": candidate.get("candidate_id"),
            "method": candidate.get("method"),
            "algorithm": candidate.get("algorithm"),
            "checkpoint": candidate.get("checkpoint"),
            "draft_identity": candidate.get("draft_identity"),
            "status": "ready",
        }
        if readiness and readiness.get("status") == "blocked":
            entry["status"] = "blocked"
            if candidate.get("method") == "sd":
                blocked_candidates.append(candidate.get("candidate_id"))
        elif readiness and candidate.get("candidate_id") in {
            item.get("candidate_id") for item in readiness.get("candidate_blockers") or []
        }:
            entry["status"] = "blocked"
            if candidate.get("method") == "sd":
                blocked_candidates.append(candidate.get("candidate_id"))
        candidate_matrix.append(entry)

    measured_runs = holdout_requests * sessions * len(candidates)
    warmup_runs = warmups_per_candidate_session * sessions * len(candidates)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "workload_id": manifest.get("workload_id"),
        "protocol_scope_hash": protocol_scope_hash(
            protocol_payload(manifest, (validation or {}).get("rows_sha256"), (validation or {}).get("labels_sha256"))
        ),
        "overall_status": (
            "blocked"
            if readiness and readiness.get("status") == "blocked"
            else ("partial" if readiness and readiness.get("status") == "partial" else "ready")
        ),
        "dry_run": True,
        "predicted_speedup": None,
        "predicted_gpu_hours": None,
        "budget_gpu_hours": budget_gpu_hours,
        "budget_note": "4 GPU-h is an execution cap, not a prediction; includes load, calibration, warmup, timing and failures.",
        "timing_scope": {
            "scope": manifest.get("measurement_scope", "local_client"),
            "boundary": "request_ready_to_committed_output",
            "includes": ["preprocess", "encode", "h2d", "generate"],
            "excludes": ["model_load", "media_disk_read"],
            "cold_start_reported_separately": True,
        },
        "counts": {
            "tuning_groups": groups_by_split.get("tuning", 0),
            "holdout_groups": groups_by_split.get("holdout", 0),
            "tuning_requests": tuning_requests,
            "holdout_requests": holdout_requests,
            "sessions": sessions,
            "warmups_per_candidate_session": warmups_per_candidate_session,
            "measured_runs": measured_runs,
            "warmup_runs": warmup_runs,
        },
        "candidate_matrix": candidate_matrix,
        "blocked_candidates": blocked_candidates,
        "missing_dependencies": (readiness or {}).get("blockers") or [],
        "warnings": (readiness or {}).get("warnings") or [],
        "calibration_required": [
            "measure target-only AR e2e and peak memory on the actual device",
            "measure each candidate's load time, memory and speculative acceptance",
            "calibrate the 4 GPU-h cap against the first 8 requests before the holdout pass",
        ],
        "required_gates": [
            "properly licensed GPU allocation with UUID flock",
            "protocol-matched AR token reference",
            "per-request token-exact correctness gate",
            "task quality scorer and threshold (or explicit not_evaluated condition)",
            "paired group bootstrap with >= 20 holdout groups per bucket",
        ],
        "ranking_allowed": False,
        "note": "No measured records exist yet; any ranking or budget estimate would be fabricated.",
    }
    if readiness and readiness.get("status") == "blocked":
        plan["overall_status"] = "blocked"
    return plan
