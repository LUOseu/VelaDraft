"""Markdown/JSON deliverables for validation, plans, summaries and decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .records import redact_record
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
    write_json(out_dir / "summary.share.json", summary)
    write_json(out_dir / "decision.share.json", decision)
    write_text(out_dir / "report.share.md", report_md)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "share_package": True,
        "removed": ["prompts", "images", "answers", "absolute_paths", "artifacts_local_paths"],
        "files": ["records.share.jsonl", "summary.share.json", "decision.share.json", "report.share.md"],
        "records_sha256": sha256_json(share_records),
        "decision_id": decision.get("decision_id"),
    }
    write_json(out_dir / "SHARE_MANIFEST.json", manifest)
    return manifest
