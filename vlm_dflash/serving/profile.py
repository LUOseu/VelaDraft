"""Explicit wire profile for serving records consumed by downstream code.

Serving and the workload advisor share a package-level schema name but do not
have identical field profiles.  This module fixes the serving profile and
states counter/resource attribution rather than silently coercing missing
engine counters to zero.
"""

from __future__ import annotations

from typing import Any

PROFILE_ID = "veladraft-serving-wire"
PROFILE_VERSION = "1.0"
PROFILE_SCHEMA_VERSION = "veladraft-platform-v1"

PROFILE_SCOPE = {
    "request": "one HTTP/offline completion request",
    "window": "one measured concurrency x session window",
    "global": "process/engine totals across a server lifetime",
}

COUNTER_DEFINITIONS = {
    "emitted": {
        "scope": "request_or_window",
        "definition": "visible output tokens emitted by the completion",
        "missing": "null with raw_counts.unavailable.emitted",
    },
    "proposed": {
        "scope": "request_or_window",
        "definition": "draft tokens proposed by the speculative path",
        "missing": "null when the engine does not expose request/window attribution",
    },
    "accepted": {
        "scope": "request_or_window",
        "definition": "draft tokens accepted after target verification",
        "missing": "null when the engine does not expose request/window attribution",
    },
    "verify_rounds": {
        "scope": "request_or_window",
        "definition": "target verification rounds entered by speculative decoding",
        "missing": "null when the engine does not expose request/window attribution",
    },
    "target_forward_calls": {
        "scope": "global",
        "definition": "target model forward calls; vLLM 0.22.1 does not expose this counter",
        "missing": "null with a mandatory reason",
    },
    "draft_forward_calls": {
        "scope": "request_or_window",
        "definition": "draft model forward calls; native N-Gram has no draft model forward",
        "missing": "null with a mandatory reason",
    },
}

LATENCY_DEFINITIONS = {
    "ttft_s": "client request start to first non-empty content/token chunk",
    "e2e_s": "client request start to terminal SSE completion or stream close",
    "tpot_s": "null unless per-token timestamps exist; SSE chunks are not per-token timestamps",
    "client_visible_token_tpot_proxy_s": "clearly labelled proxy using visible token IDs and content chunk endpoints",
    "client_chunk_gap_mean_s": "mean gap between non-empty content chunks; not ITL/TPOT",
    "cancel_requested_s": "client wall time when a cancellation probe closes the stream",
}

RESOURCE_DEFINITIONS = {
    "allocation": "additive GPU-hours from the unique allocation ledger",
    "request_exposure": "non-additive request wall-time exposure; concurrent requests overlap",
    "window_exposure": "non-additive window wall-time exposure",
    "peak_memory": "process/session sampler peak, not a per-request exclusive maximum",
}


def describe_profile() -> dict[str, Any]:
    return {
        "profile": {"id": PROFILE_ID, "version": PROFILE_VERSION},
        "schema_version": PROFILE_SCHEMA_VERSION,
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "scope": PROFILE_SCOPE,
        "counters": COUNTER_DEFINITIONS,
        "latency": LATENCY_DEFINITIONS,
        "resource": RESOURCE_DEFINITIONS,
        "normalization": {
            "unknown_counter": "null plus reason; never silently 0",
            "request_counters_require_attribution": True,
            "window_counters_are_not_request_counters": True,
            "global_counters_are_not_allocated_to_requests": True,
        },
    }


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map one serving RunRecord into the stable downstream profile."""

    scope = record.get("measurement_scope") or {}
    counts = record.get("raw_counts") or {}
    attribution = scope.get("spec_counter_attribution")
    request_counters = None
    request_counter_reason = None
    if attribution == "per_request_c1":
        request_counters = {
            key: counts.get(key)
            for key in ("emitted", "proposed", "accepted", "verify_rounds", "draft_forward_calls")
        }
    else:
        request_counter_reason = (
            "spec counters have no request tag at concurrency > 1; use the window counters"
            if attribution == "global_window_only"
            else "counter attribution is not recorded for this record"
        )
    window_counters = None
    if scope.get("window") is True:
        window_counters = {
            key: counts.get(key)
            for key in ("emitted", "proposed", "accepted", "verify_rounds", "draft_forward_calls")
        }
    resource = record.get("resource_usage") or {}
    return {
        "profile": {"id": PROFILE_ID, "version": PROFILE_VERSION},
        "schema_version": record.get("schema_version"),
        "identity": {
            "run_id": record.get("run_id"),
            "request_id": record.get("request_id"),
            "group_id": record.get("group_id"),
            "target": record.get("target_identity"),
            "draft": record.get("draft_identity"),
            "backend": record.get("backend_identity"),
            "protocol_hash": record.get("protocol_hash"),
        },
        "scope": {
            "surface": scope.get("surface"),
            "mode": scope.get("mode"),
            "concurrency": scope.get("concurrency"),
            "session_index": scope.get("session_index"),
            "request": {
                "request_id": record.get("request_id"),
                "fixture_category": scope.get("fixture_category"),
                "max_new_tokens": scope.get("max_new_tokens"),
            },
            "window": {key: scope.get(key) for key in ("window", "requests", "timing_boundary")},
            "global": {"not_recorded": True},
        },
        "counters": {
            "request": request_counters,
            "request_unavailable_reason": request_counter_reason,
            "window": window_counters,
            "global": {
                "target_forward_calls": counts.get("target_forward_calls"),
                "reason": (counts.get("unavailable") or {}).get("target_forward_calls"),
            },
        },
        "latency": dict(record.get("latency") or {}),
        "resource": {
            "gpu_hours": resource.get("gpu_hours"),
            "gpu_hours_scope": resource.get("gpu_hours_scope", "allocation"),
            "additive": resource.get("additive", True),
            "gpu_hours_reason": resource.get("gpu_hours_reason"),
            "peak_memory_mib": resource.get("peak_memory_mib"),
            "peak_memory_scope": resource.get("peak_memory_scope"),
        },
        "missing_reasons": dict(record.get("missing_reasons") or {}),
    }
