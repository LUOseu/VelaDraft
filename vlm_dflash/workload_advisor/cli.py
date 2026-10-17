#!/usr/bin/env python3
"""Enterprise workload advisor CLI (platform extension; CPU-first).

    python scripts/workload_advisor.py validate --manifest examples/workload_advisor/mini/manifest.json
    python scripts/workload_advisor.py dry-run --manifest ... --out-dir out/dry-run
    python scripts/workload_advisor.py run --manifest ... --out-dir out/pilot --budget-gpu-hours 4
    python scripts/workload_advisor.py report --records out/pilot/records.jsonl --out-dir out/report
    python scripts/workload_advisor.py recommend --records ... --out-dir out/recommend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import readiness
from .decision import decide
from .plan import build_plan
from .records import aggregate_record_from_legacy, load_records, write_records
from .report import render_markdown, summary_payload, write_deliverables, write_share_package
from .schema import protocol_payload, protocol_scope_hash, sha256_json
from .stats import summarize
from .workload import convert_sharegpt, load_workload, validation_report


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_manifest(path: str):
    return load_workload(path)


def cmd_validate(args) -> int:
    workload = _load_manifest(args.manifest)
    report = validation_report(workload)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _print(report)
    return 0 if workload.ok else 2


def cmd_dry_run(args) -> int:
    workload = _load_manifest(args.manifest)
    validation = validation_report(workload)
    if not workload.ok:
        if args.out_dir:
            write_deliverables(args.out_dir, validation=validation)
        _print(validation)
        return 2
    ready = readiness(workload.manifest)
    plan = build_plan(workload.manifest, validation=validation, readiness=ready, budget_gpu_hours=args.budget_gpu_hours)
    if args.out_dir:
        write_deliverables(args.out_dir, validation=validation, plan=plan)
    _print({"validation": validation, "plan": plan})
    return 0


def _summary_and_decision(records, manifest, workload, *, threshold, seed, min_groups):
    bucket_rules = (manifest or {}).get("bucket_rules") or {}
    min_groups = int(bucket_rules.get("per_bucket_min_groups", min_groups))
    summary = summarize(
        records,
        bucket_rules=bucket_rules,
        threshold=threshold,
        bootstrap_seed=seed,
        min_groups=min_groups,
        include_synthetic=bool(args_include_synthetic(records)),
    )
    decision = decide(summary, threshold=threshold, bootstrap_seed=seed, min_groups=min_groups)
    payload = summary_payload(summary, decision)
    return payload, decision


def args_include_synthetic(records) -> bool:
    """Fixtures opt in explicitly; real records with synthetic rows exclude them."""
    return bool(records) and all(record.get("synthetic") for record in records)


def cmd_report(args, *, recommend_only: bool = False) -> int:
    records, issues = load_records(args.records)
    if not records:
        _print({"status": "failed", "issues": issues, "message": "no valid records"})
        return 2
    workload = None
    manifest = None
    validation = None
    if args.manifest:
        workload = _load_manifest(args.manifest)
        validation = validation_report(workload)
        if not workload.ok:
            _print(validation)
            return 2
        manifest = workload.manifest
        expected = protocol_scope_hash(protocol_payload(manifest, workload.rows_digest, workload.labels_digest))
        observed = {record.get("protocol_scope_hash") for record in records if record.get("protocol_scope_hash")}
        if observed and observed != {expected}:
            issues.append(
                {
                    "code": "protocol_mismatch",
                    "message": f"records protocol_scope_hash {sorted(observed)} != manifest {expected}",
                }
            )
    synthetic = all(record.get("synthetic") for record in records)
    summary, decision = _summary_and_decision(
        records,
        manifest,
        workload,
        threshold=args.threshold,
        seed=args.bootstrap_seed,
        min_groups=args.min_groups,
    )
    if synthetic:
        summary["synthetic"] = True
        summary["overall_status"] = "synthetic_fixture"
        decision = decide(summary, threshold=args.threshold, bootstrap_seed=args.bootstrap_seed, min_groups=args.min_groups)
        summary = summary_payload(summary, decision)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        write_records(Path(args.out_dir) / "records.synthetic.jsonl", records)
    summary["issues"] = issues
    report = render_markdown(summary, decision, manifest=manifest, validation=validation)
    written = write_deliverables(
        args.out_dir,
        validation=validation,
        records=records if not synthetic else None,
        summary=summary,
        decision=decision,
        report_md=report,
    )
    if args.share:
        write_share_package(
            Path(args.out_dir) / "share",
            records=records,
            summary=summary,
            decision=decision,
            report_md=render_markdown(summary, decision, manifest=manifest, validation=validation, share=True),
        )
    _print({"summary": summary, "decision": decision, "written": written})
    return 0


def cmd_run(args) -> int:
    workload = _load_manifest(args.manifest)
    validation = validation_report(workload)
    if not workload.ok:
        if args.out_dir:
            write_deliverables(args.out_dir, validation=validation)
        _print(validation)
        return 2
    ready = readiness(workload.manifest)
    plan = build_plan(workload.manifest, validation=validation, readiness=ready, budget_gpu_hours=args.budget_gpu_hours)
    if ready["status"] == "blocked":
        blocker = {
            "schema_version": "veladraft-platform-v1",
            "project_scope": "platform_extension",
            "include_in_paper": False,
            "status": "blocked",
            "reason": "backend_not_ready",
            "readiness": ready,
            "plan": plan,
            "note": "No GPU work was started and no timing record was fabricated.",
        }
        if args.out_dir:
            write_deliverables(args.out_dir, validation=validation, plan=plan)
            with open(Path(args.out_dir) / "blocker.json", "w", encoding="utf-8") as handle:
                json.dump(blocker, handle, ensure_ascii=False, indent=2, sort_keys=True)
        _print(blocker)
        return 3
    from .backend_hf import HFWorkloadBackend  # lazy: imports torch only when GPU is ready

    backend = HFWorkloadBackend(
        workload,
        out_dir=args.out_dir,
        budget_gpu_hours=args.budget_gpu_hours,
        sessions=args.sessions,
        warmups=args.warmups,
        device=args.device,
    )
    result = backend.run(max_requests=args.max_requests)
    if result["status"] == "blocked":
        if args.out_dir:
            write_deliverables(args.out_dir, validation=validation, plan=plan)
            with open(Path(args.out_dir) / "blocker.json", "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        _print(result)
        return 3
    records = result.pop("records")
    summary, decision = _summary_and_decision(
        records,
        workload.manifest,
        workload,
        threshold=args.threshold,
        seed=args.bootstrap_seed,
        min_groups=args.min_groups,
    )
    summary["run"] = result
    summary = summary_payload(summary, decision)
    report = render_markdown(summary, decision, manifest=workload.manifest, validation=validation)
    written = write_deliverables(
        args.out_dir,
        validation=validation,
        plan=plan,
        records=records,
        summary=summary,
        decision=decision,
        report_md=report,
    )
    if args.share:
        write_share_package(
            Path(args.out_dir) / "share",
            records=records,
            summary=summary,
            decision=decision,
            report_md=render_markdown(summary, decision, manifest=workload.manifest, validation=validation, share=True),
        )
    _print({"status": result["status"], "summary": summary, "decision": decision, "written": written})
    return 0


def cmd_convert(args) -> int:
    report = convert_sharegpt(
        args.input,
        args.out_dir,
        media_root=args.media_root,
        task=args.task,
        split=args.split,
        last_turn_only=args.last_turn_only,
    )
    _print(report)
    return 0 if not report["errors"] else 2


def cmd_import_legacy(args) -> int:
    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("format") != "vlm-dflash-benchmark-v1":
        _print({"status": "failed", "message": "not a vlm-dflash-benchmark-v1 report"})
        return 2
    workloads = report.get("workloads") or {}
    selected = args.workload or [name for name, entry in workloads.items() if entry.get("status") == "measured"]
    if not selected:
        _print({"status": "failed", "message": "no measured workloads in legacy report"})
        return 2
    protocol = report.get("protocol") or {}
    target_identity = {
        "source": "legacy_report",
        "model_id": protocol.get("target") or protocol.get("model_path") or "legacy_unknown",
        "revision": protocol.get("revision"),
        "dtype": protocol.get("dtype"),
        "processor": protocol.get("processor"),
        "structure": "legacy_unknown",
    }
    backend_identity = {
        "name": "legacy_report",
        "version": report.get("format"),
        "code_commit": protocol.get("code_commit"),
        "device_topology": {"devices": [], "concurrency": protocol.get("concurrency")},
    }
    records = []
    for workload_name in selected:
        record = aggregate_record_from_legacy(
            report,
            workload_name,
            run_id=args.run_id,
            protocol_scope_hash=sha256_json({"legacy_report": args.report, "workload": workload_name}),
            target_identity=target_identity,
            backend_identity=backend_identity,
            candidate_id=args.candidate_id,
        )
        records.append(record)
    write_records(args.out, records)
    _print({"status": "imported", "records": len(records), "out": args.out, "aggregate_only": True})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate manifest, rows, labels and media (CPU only)")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--json-out", default=None)
    validate.set_defaults(func=cmd_validate)

    dry = sub.add_parser("dry-run", help="show candidate capabilities, counts and calibration needs")
    dry.add_argument("--manifest", required=True)
    dry.add_argument("--out-dir", default=None)
    dry.add_argument("--budget-gpu-hours", type=float, default=4.0)
    dry.set_defaults(func=cmd_dry_run)

    run = sub.add_parser("run", help="run readiness/correctness-gated candidates on a ready backend")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--budget-gpu-hours", type=float, default=4.0)
    run.add_argument("--sessions", type=int, default=3)
    run.add_argument("--warmups", type=int, default=4)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--max-requests", type=int, default=None, help="smoke limit; records remain partial evidence")
    run.add_argument("--threshold", type=float, default=1.05)
    run.add_argument("--bootstrap-seed", type=int, default=7)
    run.add_argument("--min-groups", type=int, default=20)
    run.add_argument("--share", action="store_true")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="offline report from existing records (no torch/CUDA)")
    report.add_argument("--records", required=True)
    report.add_argument("--manifest", default=None)
    report.add_argument("--out-dir", required=True)
    report.add_argument("--threshold", type=float, default=1.05)
    report.add_argument("--bootstrap-seed", type=int, default=7)
    report.add_argument("--min-groups", type=int, default=20)
    report.add_argument("--share", action="store_true")
    report.set_defaults(func=lambda args: cmd_report(args, recommend_only=False))

    recommend = sub.add_parser("recommend", help="offline decision from existing records")
    recommend.add_argument("--records", required=True)
    recommend.add_argument("--manifest", default=None)
    recommend.add_argument("--out-dir", required=True)
    recommend.add_argument("--threshold", type=float, default=1.05)
    recommend.add_argument("--bootstrap-seed", type=int, default=7)
    recommend.add_argument("--min-groups", type=int, default=20)
    recommend.add_argument("--share", action="store_true")
    recommend.set_defaults(func=lambda args: cmd_report(args, recommend_only=True))

    convert = sub.add_parser("convert-sharegpt", help="deterministically convert legacy ShareGPT JSONL")
    convert.add_argument("--input", required=True)
    convert.add_argument("--out-dir", required=True)
    convert.add_argument("--media-root", default="images")
    convert.add_argument("--task", default="general_vqa")
    convert.add_argument("--split", default="holdout", choices=["tuning", "holdout"])
    convert.add_argument("--last-turn-only", action="store_true")
    convert.set_defaults(func=cmd_convert)

    legacy = sub.add_parser("import-legacy", help="import old benchmark reports as aggregate_only records")
    legacy.add_argument("--report", required=True)
    legacy.add_argument("--workload", action="append", default=[])
    legacy.add_argument("--candidate-id", default="legacy")
    legacy.add_argument("--run-id", default="legacy-import")
    legacy.add_argument("--out", required=True)
    legacy.set_defaults(func=cmd_import_legacy)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    main()
