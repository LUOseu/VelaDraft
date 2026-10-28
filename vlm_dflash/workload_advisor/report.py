"""Markdown/JSON deliverables for validation, plans, summaries and decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .records import _safe_correctness, _safe_hash, _safe_latency, _safe_quality, _safe_reason, _safe_resource_usage, _safe_token, _share_id, redact_record
from .schema import SCHEMA_VERSION, canonical_json, sha256_json


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(speed: dict) -> str:
    low, high = speed.get("ci95_low"), speed.get("ci95_high")
    point = speed.get("point_ratio")
    if low is None or high is None:
        return f"point={_fmt(point)}; CI=null"
    return f"{_fmt(point)} [{_fmt(low)}, {_fmt(high)}]"


def summary_payload(summary: dict, decision: dict) -> dict:
    payload = dict(summary)
    payload["decision"] = decision
    payload["summary_sha256"] = sha256_json({key: value for key, value in payload.items() if key != "summary_sha256"})
    return payload


def render_markdown(
    summary: dict,
    decision: dict,
    *,
    manifest: dict | None = None,
    validation: dict | None = None,
    share: bool = False,
) -> str:
    synthetic = bool(summary.get("synthetic"))
    lines: list[str] = []
    lines.append("# VelaDraft workload advisory report")
    lines.append("")
    if synthetic:
        lines += [
            "> **SYNTHETIC FIXTURE** — these numbers are generated CPU fixtures, not real accelerator measurements.",
            "> They must not be copied into a production or paper results table.",
            "",
        ]
    elif summary.get("n_synthetic_records", 0):
        lines += [
            f"> {summary['n_synthetic_records']} synthetic record(s) were excluded from this report's real tables.",
            "",
        ]
    lines.append(f"- schema: `{SCHEMA_VERSION}`")
    lines.append(f"- project scope: `platform_extension`; include_in_paper: `false`")
    if manifest:
        lines.append(f"- workload: `{manifest.get('workload_id')}`")
        lines.append(f"- target: `{manifest.get('target', {}).get('model_id')}` ({manifest.get('target', {}).get('dtype')})")
        lines.append(f"- measurement scope: `{manifest.get('measurement_scope')}`; concurrency={manifest.get('concurrency')}")
    lines.append(f"- groups: {summary.get('n_groups', 0)}; sessions: {summary.get('n_sessions', 0)}; tasks: {summary.get('n_tasks', 0)}")
    protocol_hashes = summary.get("protocol_scope_hashes") or []
    lines.append(f"- protocol scope hash(es): `{', '.join(protocol_hashes) if protocol_hashes else 'missing'}`")
    lines.append(f"- bootstrap: seed={summary.get('bootstrap_seed')}, n_boot={summary.get('n_boot')}, paired by group")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- action: **{decision.get('recommended_action')}**")
    lines.append(f"- method: `{decision.get('recommended_method')}`")
    lines.append(f"- threshold: paired speed-ratio CI lower > {decision.get('threshold')}")
    lines.append(f"- coverage: {json.dumps(decision.get('coverage'), sort_keys=True)}")
    for reason in decision.get("reasons") or []:
        lines.append(f"- {reason}")
    if decision.get("excluded_candidates"):
        lines.append("")
        lines.append("Excluded candidates:")
        for item in decision["excluded_candidates"]:
            codes = "; ".join(f"{reason['code']}: {reason['detail']}" for reason in item["reasons"])
            lines.append(f"- `{item['candidate_id']}` — {codes}")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    lines.append("| candidate | records | groups | correctness | quality | speed ratio [95% CI] | memory fit | latency mean AR -> candidate (s) |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
    ar = (summary.get("candidates") or {}).get("ar") or {}
    for candidate_id, candidate in sorted((summary.get("candidates") or {}).items()):
        if candidate_id == "ar":
            continue
        correctness = candidate.get("correctness") or {}
        status = ", ".join(f"{key}={value}" for key, value in (correctness.get("status_counts") or {}).items())
        quality = candidate.get("quality") or {}
        ar_mean = ((ar.get("latency") or {}).get("e2e_s") or {}).get("mean")
        cand_mean = (candidate.get("latency") or {}).get("e2e_s", {}).get("mean")
        lines.append(
            "| `{cid}` | {n} | {g} | {status} (failures {f}/{d}) | {q} | {ci} | {mem} | {ar} -> {cand} |".format(
                cid=candidate_id,
                n=candidate.get("n_records"),
                g=candidate.get("n_groups"),
                status=status or "none",
                f=correctness.get("failed_records"),
                d=correctness.get("denominator_records"),
                q=quality.get("status"),
                ci=_ci(candidate.get("speed") or {}),
                mem=(candidate.get("resources") or {}).get("memory_fit"),
                ar=_fmt(ar_mean),
                cand=_fmt(cand_mean),
            )
        )
    lines.append("")
    lines.append("## Latency and resources")
    lines.append("")
    lines.append("| method | e2e mean | p50 | p95 | n | serial output tok/s | TTFT | ITL | peak allocated bytes | GPU-h (max record) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |")
    for candidate_id, candidate in sorted((summary.get("candidates") or {}).items()):
        latency = candidate.get("latency") or {}
        e2e = latency.get("e2e_s") or {}
        resources = candidate.get("resources") or {}
        lines.append(
            f"| `{candidate_id}` | {_fmt(e2e.get('mean'))} | {_fmt(e2e.get('p50'))} | {_fmt(e2e.get('p95'))} | "
            f"{e2e.get('n')} | {_fmt(latency.get('serial_output_tokens_per_second'))} | "
            f"{_fmt((latency.get('ttft_s') or {}).get('mean'))} | "
            f"{latency.get('itl')} | {json.dumps(resources.get('peak_allocated_bytes') or {}, sort_keys=True)} | "
            f"{_fmt(resources.get('all_allocated_gpu_hours_max_record'), 6)} |"
        )
    lines.append("")
    lines.append("TTFT/ITL stay null when the backend does not expose token visibility; no proxy is substituted.")
    lines.append("The throughput column is `serial_output_tokens_per_second`, not a concurrent serving throughput.")
    lines.append("")
    lines.append("## Bucket evidence")
    lines.append("")
    lines.append("Frozen tuning boundaries; buckets with fewer than the required independent groups are `insufficient_evidence`.")
    lines.append("")
    any_bucket = False
    for candidate_id, candidate in sorted((summary.get("candidates") or {}).items()):
        for axis, buckets in (candidate.get("buckets") or {}).items():
            any_bucket = True
            lines.append(f"### `{candidate_id}` by {axis}")
            lines.append("")
            lines.append("| bucket | n groups | status | ratio [95% CI] |")
            lines.append("| --- | ---: | --- | --- |")
            for label, stats in sorted(buckets.items()):
                lines.append(
                    f"| {label} | {stats.get('n_groups')} | {stats.get('status')} | {_ci(stats)} |"
                )
            lines.append("")
    if not any_bucket:
        lines.append("No bucket evidence recorded.")
        lines.append("")
    lines.append("## Correctness and quality")
    lines.append("")
    lines.append("- Correctness statuses follow the protocol enum; `verified_greedy_exact` requires per-request token comparison against a protocol-matched AR reference.")
    lines.append("- A 0/N failure count retains N; it is not evidence that the failure probability is strictly zero.")
    lines.append("- Task quality is reported separately from token exactness and run errors.")
    for candidate_id, candidate in sorted((summary.get("candidates") or {}).items()):
        quality = candidate.get("quality") or {}
        if quality.get("json_schema_valid_rate") is not None:
            lines.append(
                f"- `{candidate_id}` JSON-schema valid rate: {_fmt(quality['json_schema_valid_rate'])} "
                f"(n={quality.get('json_schema_n')}); this measures structure, not field content."
            )
    if manifest:
        scorers = manifest.get("scorers") or {}
        if scorers:
            lines.append(f"- frozen scorers: `{json.dumps(scorers, sort_keys=True)}`")
    lines.append("")
    lines.append("## Gaps and next steps")
    lines.append("")
    if synthetic:
        lines.append("- Replace synthetic fixture timings with a real HF backend run on an allocated GPU (4 GPU-h first-round cap).")
    if summary.get("fallback_records"):
        lines.append(f"- {summary['fallback_records']} fallback record(s) are excluded from candidate speed evidence.")
    if summary.get("overall_status") == "insufficient_evidence":
        lines.append("- Evidence is insufficient for a production claim; add independent groups or a compliant candidate.")
    insuff = (decision.get("coverage") or {}).get("buckets_insufficient") or {}
    if insuff:
        lines.append(f"- Buckets still below the minimum group count: `{json.dumps(insuff, sort_keys=True)}`.")
    lines.append("- No cross-hardware, cross-engine or cross-customer generalization is claimed.")
    lines.append("")
    lines.append("## Replay")
    lines.append("")
    lines.append("```bash")
    if share:
        lines.append("# share package: prompts, images, answers and absolute paths removed")
    lines.append("python scripts/workload_advisor.py recommend --records records.jsonl --out-dir out --bootstrap-seed 7")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _redact_issue(issue: Any) -> dict:
    if not isinstance(issue, dict):
        return {"code": "redacted_issue"}
    result = {"code": _safe_reason(issue.get("code"))}
    if isinstance(issue.get("line"), int):
        result["line"] = issue["line"]
    if isinstance(issue.get("count"), int):
        result["count"] = issue["count"]
    return result


def _redact_readiness(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    result = {}
    status = _safe_token(value.get("status"))
    if status:
        result["status"] = status
    checks = value.get("checks")
    if isinstance(checks, dict):
        clean_checks = {}
        for key, check in checks.items():
            safe_key = _safe_reason(key)
            if safe_key == "redacted":
                continue
            if isinstance(check, dict):
                clean = {
                    item: check[item]
                    for item in ("ok", "status", "available", "required")
                    if item in check and (check[item] is None or isinstance(check[item], (bool, int, float)))
                }
                if clean:
                    clean_checks[safe_key] = clean
            elif isinstance(check, (bool, int, float)) or check is None:
                clean_checks[safe_key] = check
        if clean_checks:
            result["checks"] = clean_checks
    for key in ("blockers", "reasons"):
        if isinstance(value.get(key), list):
            codes = [
                _safe_reason(item.get("code") if isinstance(item, dict) else item)
                for item in value[key]
            ]
            codes = [code for code in codes if code != "redacted"]
            if codes:
                result[key] = codes
    return result


def _redact_run(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    result = {}
    scalar_keys = {
        "schema_version",
        "project_scope",
        "include_in_paper",
        "status",
        "run_id",
        "budget_gpu_hours",
        "budget_used_gpu_hours",
        "budget_remaining_gpu_hours",
        "records_total",
        "records_written",
        "requests_total",
        "requests_succeeded",
        "requests_failed",
        "error_count",
        "elapsed_s",
        "session_count",
        "warmup_count",
    }
    for key in scalar_keys:
        if key in value and (value[key] is None or isinstance(value[key], (bool, int, float))):
            result[key] = value[key]
        elif key in {"status", "run_id", "schema_version", "project_scope"}:
            token = _share_id(value.get(key)) if key == "run_id" else _safe_token(value.get(key))
            if token:
                result[key] = token
    readiness = _redact_readiness(value.get("readiness"))
    if readiness is not None:
        result["readiness"] = readiness
    error = value.get("error")
    if isinstance(error, dict):
        result["error"] = {
            key: _safe_reason(error.get(key))
            for key in ("code", "type", "stage")
            if error.get(key)
        }
    if value.get("status") == "blocked":
        reason = _safe_reason(value.get("reason"))
        if reason != "redacted":
            result["reason"] = reason
    return result


def _redact_candidate(candidate: dict) -> dict:
    result = {}
    for key in ("candidate_id", "method"):
        token = _share_id(candidate.get(key)) if key == "candidate_id" else _safe_token(candidate.get(key))
        if token:
            result[key] = token
    for key in (
        "n_records",
        "n_requests",
        "n_groups",
        "n_pairs",
        "n_sessions",
    ):
        if isinstance(candidate.get(key), int):
            result[key] = candidate[key]
    gates = candidate.get("gates") or {}
    result["gates"] = {
        key: value
        for key, value in gates.items()
        if isinstance(value, (bool, int, float)) and key not in {"threshold"}
    }
    if isinstance(gates.get("threshold"), (int, float)):
        result["gates"]["threshold"] = gates["threshold"]
    speed = candidate.get("speed") or {}
    result["speed"] = {
        key: speed[key]
        for key in (
            "point_ratio",
            "ci95_low",
            "ci95_high",
            "n_groups",
            "n_pairs",
            "n_boot",
            "bootstrap_seed",
        )
        if speed.get(key) is None or isinstance(speed.get(key), (int, float))
    }
    method = _safe_token(speed.get("method"))
    if method:
        result["speed"]["method"] = method
    correctness = _safe_correctness(candidate.get("correctness"))
    if correctness is not None:
        result["correctness"] = correctness
    quality = _safe_quality(candidate.get("quality"))
    if quality is not None:
        result["quality"] = quality
    result["latency"] = _safe_latency(candidate.get("latency"))
    result["resources"] = _safe_resource_usage(candidate.get("resources"))
    pairing = candidate.get("pairing") or {}
    result["pairing_counts"] = {
        key: len(value) if isinstance(value, list) else value
        for key, value in pairing.items()
        if isinstance(value, int) or isinstance(value, list)
    }
    result["tasks"] = [
        token for token in (_share_id(task) for task in candidate.get("tasks") or []) if token
    ]
    result["measurement_scopes"] = [
        token for token in (_safe_token(scope) for scope in candidate.get("measurement_scopes") or []) if token
    ]
    result["acceptance_policies"] = [
        token
        for token in (_safe_token(policy) for policy in candidate.get("acceptance_policies") or [])
        if token
    ]
    buckets = {}
    for axis, values in (candidate.get("buckets") or {}).items():
        safe_axis = _safe_token(axis)
        if not safe_axis or not isinstance(values, dict):
            continue
        clean_buckets = {}
        for label, stats in values.items():
            safe_label = _share_id(label) if axis == "task" else _safe_reason(label)
            if not safe_label or safe_label == "redacted" or not isinstance(stats, dict):
                continue
            clean_buckets[safe_label] = {
                key: stats[key]
                for key in ("point_ratio", "ci95_low", "ci95_high", "n_groups", "min_groups")
                if stats.get(key) is None or isinstance(stats.get(key), (int, float))
            }
            status = _safe_token(stats.get("status"))
            if status:
                clean_buckets[safe_label]["status"] = status
        buckets[safe_axis] = clean_buckets
    if buckets:
        result["buckets"] = buckets
    tuning = candidate.get("tuning") or {}
    result["tuning"] = {
        key: tuning[key]
        for key in ("n_records", "n_pairs", "n_groups", "point_ratio")
        if tuning.get(key) is None or isinstance(tuning.get(key), (int, float))
    }
    return result


def redact_summary(summary: dict) -> dict:
    """Allowlisted summary for sharing; unknown/free-text fields are dropped."""
    result = {}
    for key in (
        "schema_version",
        "project_scope",
        "include_in_paper",
        "evidence_split",
        "assignment_verified",
        "assignment_sha256",
        "protocol_mismatch",
        "overall_status",
        "threshold",
        "bootstrap_seed",
        "n_boot",
        "min_groups",
        "n_records",
        "n_real_records",
        "n_synthetic_records",
        "n_holdout_records",
        "n_tuning_records",
        "n_groups",
        "n_expected_requests",
        "n_tasks",
        "n_sessions",
        "fallback_records",
        "duplicate_records",
        "frozen_candidates_sha256",
        "records_sha256",
        "summary_sha256",
        "synthetic",
    ):
        if key in summary:
            value = summary[key]
            if key.endswith("_sha256"):
                result[key] = _safe_hash(value)
            elif isinstance(value, str):
                result[key] = _safe_token(value)
            elif value is None or isinstance(value, (bool, int, float)):
                result[key] = value
    result["protocol_scope_hashes"] = [
        value for value in (_safe_hash(value) for value in summary.get("protocol_scope_hashes") or []) if value
    ]
    result["task_group_counts"] = {
        _share_id(task) or "unknown": count
        for task, count in (summary.get("task_group_counts") or {}).items()
        if isinstance(count, int)
    }
    result["concurrency"] = [
        value for value in summary.get("concurrency") or [] if isinstance(value, int)
    ]
    integrity = summary.get("input_integrity") or {}
    result["input_integrity"] = {
        "status": _safe_token(integrity.get("status")) or "unknown",
        "issues": [_redact_issue(issue) for issue in integrity.get("issues") or []],
    }
    pairing = summary.get("pairing") or {}
    result["pairing_counts"] = {
        key: value for key, value in pairing.items() if isinstance(value, int)
    }
    for key in ("duplicate_ids", "missing_ar", "missing_candidate", "failed_ar", "failed_candidate",
                "fallback_ar", "fallback_candidate", "protocol_mismatch"):
        if isinstance(pairing.get(key), list):
            result["pairing_counts"][key] = len(pairing[key])
    result["candidate_ids"] = [
        token for token in (_share_id(candidate) for candidate in summary.get("candidate_ids") or []) if token
    ]
    result["candidates"] = {
        _share_id(candidate_id) or "unknown": _redact_candidate(candidate)
        for candidate_id, candidate in (summary.get("candidates") or {}).items()
        if isinstance(candidate, dict)
    }
    run = _redact_run(summary.get("run"))
    if run is not None:
        result["run"] = run
    readiness = _redact_readiness(summary.get("readiness"))
    if readiness is not None:
        result["readiness"] = readiness
    result["issues"] = [_redact_issue(issue) for issue in summary.get("issues") or []]
    return result


def redact_decision(decision: dict) -> dict:
    """Allowlisted decision for sharing; free-text reasons stay local."""
    result = {}
    for key in (
        "schema_version",
        "project_scope",
        "include_in_paper",
        "recommended_action",
        "recommended_method",
        "threshold",
        "bootstrap_seed",
        "assignment_verified",
        "assignment_sha256",
        "input_integrity_status",
        "protocol_hash",
        "records_sha256",
        "frozen_candidates_sha256",
        "summary_sha256",
        "synthetic",
        "decision_id",
    ):
        if key in decision:
            value = decision[key]
            if key.endswith("_sha256") or key in {"protocol_hash", "decision_id"}:
                result[key] = _safe_hash(value)
            elif key == "recommended_method":
                result[key] = _share_id(value)
            elif isinstance(value, str):
                result[key] = _safe_token(value)
            elif value is None or isinstance(value, (bool, int, float)):
                result[key] = value
    result["protocol_scope_hashes"] = [
        value for value in (_safe_hash(value) for value in decision.get("protocol_scope_hashes") or []) if value
    ]
    scope = decision.get("scope") or {}
    clean_scope = {
        key: scope[key]
        for key in ("threshold", "min_groups", "n_groups", "n_sessions")
        if isinstance(scope.get(key), (int, float))
    }
    measurement_scope = _safe_token(scope.get("measurement_scope"))
    if measurement_scope:
        clean_scope["measurement_scope"] = measurement_scope
    for key in ("tasks", "target_V_buckets", "ar_R_buckets"):
        values = scope.get(key)
        if isinstance(values, list):
            clean_scope[key] = [
                token for token in ((_share_id(value) if key == "tasks" else _safe_token(value)) for value in values) if token
            ]
    conditional = []
    for item in scope.get("conditional_buckets") or []:
        if not isinstance(item, dict):
            continue
        clean_item = {"candidate_id": _share_id(item.get("candidate_id"))}
        clean_buckets = []
        for bucket in item.get("buckets") or []:
            if not isinstance(bucket, dict):
                continue
            clean_bucket = {"axis": _safe_token(bucket.get("axis"))}
            label = _share_id(bucket.get("bucket")) if bucket.get("axis") == "task" else _safe_reason(bucket.get("bucket"))
            if label and label != "redacted":
                clean_bucket["bucket"] = label
            for key in ("n_groups", "ci95_low", "ci95_high"):
                if isinstance(bucket.get(key), (int, float)):
                    clean_bucket[key] = bucket[key]
            clean_buckets.append(clean_bucket)
        clean_item["buckets"] = clean_buckets
        conditional.append(clean_item)
    if conditional:
        clean_scope["conditional_buckets"] = conditional
    result["scope"] = clean_scope
    coverage = decision.get("coverage") or {}
    clean_coverage = {
        key: coverage[key]
        for key in ("n_groups", "n_expected_requests", "n_sessions", "n_tasks", "min_groups")
        if isinstance(coverage.get(key), (int, float))
    }
    clean_coverage["task_group_counts"] = {
        _share_id(task) or "unknown": count
        for task, count in (coverage.get("task_group_counts") or {}).items()
        if isinstance(count, int)
    }
    clean_coverage["buckets_insufficient"] = {}
    for key, values in (coverage.get("buckets_insufficient") or {}).items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        candidate_id, separator, axis = key.rpartition(":")
        if not separator or not _safe_token(axis):
            continue
        clean_key = f"{_share_id(candidate_id) or 'unknown'}:{axis}"
        clean_coverage["buckets_insufficient"][clean_key] = [
            label
            for label in ((_share_id(value) if axis == "task" else _safe_reason(value)) for value in values)
            if label and label != "redacted"
        ]
    result["coverage"] = clean_coverage
    result["reason_count"] = len(decision.get("reasons") or [])
    result["reasons"] = ["details_available_in_local_decision"]
    excluded = []
    for item in decision.get("excluded_candidates") or []:
        if not isinstance(item, dict):
            continue
        excluded.append(
            {
                "candidate_id": _share_id(item.get("candidate_id")) or "unknown",
                "reason_codes": [
                    _safe_reason(reason.get("code"))
                    for reason in item.get("reasons") or []
                    if isinstance(reason, dict)
                ],
            }
        )
    result["excluded_candidates"] = excluded
    return result


def render_share_markdown(summary: dict, decision: dict) -> str:
    lines = [
        "# VelaDraft workload advisory report (share package)",
        "",
        "- schema: `veladraft-platform-v1`",
        "- project scope: `platform_extension`; include_in_paper: `false`",
        "- This package contains public IDs, hashes, aggregate scores and reason codes only.",
        "- Prompts, media, answers, predictions, raw error text and local paths stay private.",
        "",
        "## Recommendation",
        "",
        f"- action: **{decision.get('recommended_action')}**",
        f"- method: `{decision.get('recommended_method')}`",
        f"- threshold: `{decision.get('threshold')}`",
        f"- assignment verified: `{decision.get('assignment_verified')}`",
        f"- input integrity: `{decision.get('input_integrity_status')}`",
        f"- coverage: `{json.dumps(decision.get('coverage'), sort_keys=True)}`",
        "",
        "## Candidates",
        "",
        "| candidate | groups | pairs | correctness | quality | speed ratio [95% CI] |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for candidate_id, candidate in sorted((summary.get("candidates") or {}).items()):
        if candidate_id == "ar":
            continue
        speed = candidate.get("speed") or {}
        lines.append(
            "| `{}` | {} | {} | `{}` | `{}` | {} |".format(
                candidate_id,
                candidate.get("n_groups", 0),
                candidate.get("n_pairs", 0),
                (candidate.get("correctness") or {}).get("gate"),
                (candidate.get("quality") or {}).get("status"),
                _ci(speed),
            )
        )
    if decision.get("excluded_candidates"):
        lines += ["", "Excluded candidates:", ""]
        for item in decision["excluded_candidates"]:
            codes = ", ".join(item.get("reason_codes") or [])
            lines.append(f"- `{item.get('candidate_id')}` — {codes}")
    lines += [
        "",
        "## Replay",
        "",
        "Replay the local records and manifest under the documented protocol; no private sample text is included here.",
        "",
    ]
    return "\n".join(lines)


def write_deliverables(
    out_dir: str | Path,
    *,
    validation: dict | None = None,
    plan: dict | None = None,
    records: Iterable[dict] | None = None,
    summary: dict | None = None,
    decision: dict | None = None,
    report_md: str | None = None,
    share: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    if validation is not None:
        write_json(out_dir / "validation.json", validation)
        written["validation"] = str(out_dir / "validation.json")
    if plan is not None:
        write_json(out_dir / "plan.json", plan)
        written["plan"] = str(out_dir / "plan.json")
    if records is not None:
        records = list(records)
        path = out_dir / "records.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
        written["records"] = str(path)
    if summary is not None:
        write_json(out_dir / "summary.json", summary)
        written["summary"] = str(out_dir / "summary.json")
    if decision is not None:
        write_json(out_dir / "decision.json", decision)
        written["decision"] = str(out_dir / "decision.json")
    if report_md is not None:
        write_text(out_dir / "report.md", report_md)
        written["report"] = str(out_dir / "report.md")
    return written


def write_share_package(
    out_dir: str | Path,
    *,
    records: Iterable[dict],
    summary: dict,
    decision: dict,
    report_md: str,
) -> dict:
    """Default share package: hashes/aggregates only; no prompts, media or answers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    share_records = [redact_record(record) for record in records]
    records_path = out_dir / "records.share.jsonl"
    with open(records_path, "w", encoding="utf-8") as handle:
        for record in share_records:
            handle.write(canonical_json(record) + "\n")
    share_summary = redact_summary(summary)
    share_decision = redact_decision(decision)
    write_json(out_dir / "summary.share.json", share_summary)
    write_json(out_dir / "decision.share.json", share_decision)
    write_text(out_dir / "report.share.md", render_share_markdown(share_summary, share_decision))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "share_package": True,
        "redaction_policy": "structural_allowlist_v1",
        "removed": [
            "prompts",
            "images",
            "answers",
            "predictions",
            "quality_references",
            "raw_error_messages",
            "free_text_reasons",
            "absolute_and_local_paths",
            "artifacts_local_paths",
        ],
        "files": ["records.share.jsonl", "summary.share.json", "decision.share.json", "report.share.md"],
        "records_sha256": sha256_json(share_records),
        "decision_id": _safe_hash(decision.get("decision_id")),
    }
    write_json(out_dir / "SHARE_MANIFEST.json", manifest)
    return manifest
