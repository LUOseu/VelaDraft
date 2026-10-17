#!/usr/bin/env python
"""VelaDraft serving gate: capability audit, AR/N-Gram correctness, HTTP lifecycle.

Examples (run with the pinned vLLM environment):

  python scripts/run_serving_gate.py fixture-check --manifest tests/serving/fixtures/manifest.json
  python scripts/run_serving_gate.py audit --model /home/lyh/models/Qwen2.5-VL-3B-Instruct \
      --out configs/platform_extensions/serving/capabilities.json --source-commit <sha>
  python scripts/run_serving_gate.py serve --config configs/platform_extensions/serving/ngram.json
  python scripts/run_serving_gate.py offline --mode ngram --model ... --manifest ... --out ...
  python scripts/run_serving_gate.py compare --ar ... --ngram ... --out ...
  python scripts/run_serving_gate.py measure --mode ngram --base-url http://127.0.0.1:8101 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash.serving import audit as audit_mod
from vlm_dflash.serving import compare as compare_mod
from vlm_dflash.serving import fixtures as fixtures_mod
from vlm_dflash.serving import launch as launch_mod
from vlm_dflash.serving import measure as measure_mod
from vlm_dflash.serving import offline as offline_mod
from vlm_dflash.serving.contracts import read_json, write_json_atomic


def _devices_json(index: int) -> str:
    import torch

    props = torch.cuda.get_device_properties(index)
    try:
        uuid = launch_mod.gpu_uuid(index)
    except Exception:
        uuid = None
    return json.dumps(
        [
            {
                "index": index,
                "name": props.name,
                "uuid": uuid,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        ]
    )


def cmd_serve(args) -> int:
    launch_mod.serve(args.config, port=args.port, gpu_index=args.gpu_index)
    return 0


def cmd_audit(args) -> int:
    evidence = {
        "gpu_smoke": {
            "ar": {"path": args.gpu_smoke_ar, "note": "AR GPU smoke"} if args.gpu_smoke_ar else None,
            "ngram": {"path": args.gpu_smoke_ngram, "note": "N-Gram GPU smoke"} if args.gpu_smoke_ngram else None,
        },
        "lifecycle": {"stream": {"path": args.lifecycle}, "cancel": {"path": args.lifecycle}} if args.lifecycle else {},
        "correctness": {"path": args.correctness} if args.correctness else {},
        "dflash": {"wp5": {"path": args.dflash_wp5}} if args.dflash_wp5 else {},
    }
    _, doc = audit_mod.collect_audit(
        model_path=args.model,
        source_commit=args.source_commit,
        evidence=evidence,
    )
    write_json_atomic(args.out, doc)
    print(json.dumps({"out": args.out, "capabilities": {k: v["status"] for k, v in doc["capabilities"].items()}}, indent=2))
    return 0


def cmd_fixture_check(args) -> int:
    manifest = fixtures_mod.load_manifest(args.manifest, verify_hashes=not args.skip_hash_check)
    print(
        json.dumps(
            {
                "manifest": str(Path(args.manifest).resolve()),
                "natural": len(manifest["requests"]),
                "boundary": len(manifest["boundary"]),
                "diagnostic": len(manifest["diagnostic"]),
                "counts": manifest.get("counts"),
                "ok": True,
            },
            indent=2,
        )
    )
    return 0


def cmd_offline(args) -> int:
    doc = offline_mod.run_offline(args)
    print(json.dumps(doc["summary"], indent=2))
    return 0 if doc["summary"]["num_errors"] == 0 else 2


def cmd_compare(args) -> int:
    summary = compare_mod.run_compare(args.ar, args.ngram, out=args.out)
    print(json.dumps({k: summary[k] for k in ("ar_ar", "sd_ar", "diagnostic", "natural", "completion_levels")}, indent=2))
    return 0 if summary["completion_levels"]["SD_EXECUTED"] else 2


def cmd_measure(args) -> int:
    if not args.devices_json:
        args.devices_json = _devices_json(args.gpu_index)
    doc = measure_mod.run_measure(args)
    print(json.dumps(doc["summary"], indent=2))
    return 0


def _load_windows(path: str) -> list[dict]:
    doc = read_json(path)
    return doc.get("windows", [])


def _measure_lifecycle_checks(doc: dict) -> dict:
    """Validate the actual C=1/2/4 stream, cancel and pairing evidence."""

    if not doc:
        return {"ok": False, "reason": "measurement document missing"}
    meta = doc.get("meta", {})
    try:
        concurrency_values = [int(value) for value in str(meta.get("concurrency", "1,2,4")).split(",")]
        sessions = int(meta.get("sessions", 3))
    except (TypeError, ValueError):
        return {"ok": False, "reason": "invalid concurrency/session metadata"}
    expected = {(c, s) for c in concurrency_values for s in range(1, sessions + 1)}
    windows = doc.get("windows", [])
    observed = {
        (window.get("measurement_scope", {}).get("concurrency"), window.get("measurement_scope", {}).get("session_index"))
        for window in windows
    }
    records = doc.get("records", [])
    summary = doc.get("summary", {})
    boundaries = summary.get("boundary_results", [])
    cancels = summary.get("cancel_probes", [])
    nonstream = summary.get("nonstream_probes", [])
    pairs = summary.get("pair_checks", [])
    cancel_keys = {
        (item.get("concurrency"), item.get("session"))
        for item in cancels
        if isinstance(item, dict)
    }
    checks = {
        "expected_windows": sorted(expected),
        "observed_windows": sorted(observed),
        "all_windows_present": expected.issubset(observed),
        "window_errors": sum(int(window.get("num_errors") or 0) for window in windows),
        "record_errors": sum(1 for record in records if record.get("error")),
        "records": len(records),
        "stream_records": sum(1 for record in records if record.get("token_ids")),
        "boundary_ok": len(boundaries) == 8 and all(item.get("ok") for item in boundaries),
        "cancel_probes": len(cancels),
        "cancel_keys": sorted(cancel_keys),
        "cancel_all_reclaimed": len(cancels) == len(expected) and cancel_keys == expected and all(
            item.get("cancelled") is True
            and item.get("observed_in_flight") is True
            and item.get("cancel_reclaimed_s") is not None
            and not item.get("error")
            for item in cancels
        ),
        "nonstream_probes": len(nonstream),
        "nonstream_ok": len(nonstream) >= 1 and all(item.get("ok") for item in nonstream),
        "pair_checks": len(pairs),
        "pairs_ok": len(pairs) == len(expected) and all(item.get("ok") for item in pairs),
    }
    checks["ok"] = bool(
        checks["all_windows_present"]
        and checks["window_errors"] == 0
        and checks["record_errors"] == 0
        and checks["records"] > 0
        and checks["stream_records"] > 0
        and checks["boundary_ok"]
        and checks["cancel_all_reclaimed"]
        and checks["nonstream_ok"]
        and checks["pairs_ok"]
    )
    return checks


def cmd_summary(args) -> int:
    correctness = read_json(args.correctness) if args.correctness else {}
    ar_doc = read_json(args.ar_measure) if args.ar_measure else {}
    sd_doc = read_json(args.ngram_measure) if args.ngram_measure else {}

    def by_key(doc, key):
        out = {}
        for window in doc.get("windows", []):
            scope = window.get("measurement_scope", {})
            out[(scope.get("concurrency"), scope.get("session_index"))] = window
        return out

    ar_windows = by_key(ar_doc, "ar")
    sd_windows = by_key(sd_doc, "sd")
    speedups = []
    for (concurrency, session), ar_window in sorted(ar_windows.items()):
        sd_window = sd_windows.get((concurrency, session))
        if sd_window is None:
            continue
        ar_wall = ar_window.get("latency", {}).get("total_s")
        sd_wall = sd_window.get("latency", {}).get("total_s")
        if ar_wall and sd_wall:
            speedups.append(
                {
                    "concurrency": concurrency,
                    "session": session,
                    "ar_wall_s": ar_wall,
                    "sd_wall_s": sd_wall,
                    "speedup": ar_wall / sd_wall,
                }
            )
    speedup_by_c = {}
    for concurrency in sorted({item["concurrency"] for item in speedups}):
        items = [item for item in speedups if item["concurrency"] == concurrency]
        ar_total = sum(item["ar_wall_s"] for item in items)
        sd_total = sum(item["sd_wall_s"] for item in items)
        speedup_by_c[str(concurrency)] = {
            "pairs": len(items),
            "ar_total_s": ar_total,
            "sd_total_s": sd_total,
            "speedup_paired_total": (ar_total / sd_total) if sd_total else None,
            "raw": items,
        }

    def latency_summary(doc):
        records = doc.get("records", [])
        # Exclude the cancellation and same-text/different-image probes from
        # the natural-request latency summary; they remain in raw records and
        # in the lifecycle checks below.
        records = [
            record
            for record in records
            if not record.get("cancelled")
            and record.get("category") in {"ocr", "chart_doc", "vqa", "caption"}
            and "-nat-" in str(record.get("request_id", ""))
        ]
        ttft = [r["latency"].get("ttft_s") for r in records if r["latency"].get("ttft_s") is not None]
        e2e = [r["latency"].get("e2e_s") for r in records if r["latency"].get("e2e_s") is not None]
        tpot = [r["latency"].get("tpot_s") for r in records if r["latency"].get("tpot_s") is not None]
        from vlm_dflash.serving.metrics import summarize_latencies

        return {"ttft": summarize_latencies(ttft), "e2e": summarize_latencies(e2e), "tpot": summarize_latencies(tpot)}

    ar_checks = _measure_lifecycle_checks(ar_doc)
    sd_checks = _measure_lifecycle_checks(sd_doc)
    cancel_ok = bool(
        ar_checks.get("cancel_all_reclaimed") is True
        and sd_checks.get("cancel_all_reclaimed") is True
    )
    lifecycle_ok = bool(ar_checks.get("ok") and sd_checks.get("ok"))
    levels = dict(correctness.get("completion_levels", {}))
    levels["AR_API_PASS"] = bool(ar_checks.get("ok"))
    levels["SERVING_LIFECYCLE_PASS"] = bool(lifecycle_ok)
    summary = {
        "schema_version": "veladraft-serving-summary-v1",
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "completion_levels": levels,
        "correctness": {
            "ar_ar": correctness.get("ar_ar"),
            "sd_ar": correctness.get("sd_ar"),
            "categories": correctness.get("categories"),
            "diagnostic": correctness.get("diagnostic"),
            "natural": correctness.get("natural"),
        },
        "latency": {
            "ar": latency_summary(ar_doc) if ar_doc else None,
            "ngram": latency_summary(sd_doc) if sd_doc else None,
        },
        "speedup": speedup_by_c,
        "cancel": {
            "ar": ar_doc.get("summary", {}).get("cancel_probes") if ar_doc else None,
            "ngram": sd_doc.get("summary", {}).get("cancel_probes") if sd_doc else None,
            "all_reclaimed_within_window": cancel_ok,
        },
        "budget": {
            "ar_gpu_hours": ar_doc.get("summary", {}).get("gpu_hours") if ar_doc else None,
            "ngram_gpu_hours": sd_doc.get("summary", {}).get("gpu_hours") if sd_doc else None,
            "ar_allocation_wall_s": ar_doc.get("summary", {}).get("allocation_wall_s") if ar_doc else None,
            "ngram_allocation_wall_s": sd_doc.get("summary", {}).get("allocation_wall_s") if sd_doc else None,
        },
        "lifecycle_checks": {"ar": ar_checks, "ngram": sd_checks},
    }
    if args.out:
        write_json_atomic(args.out, summary)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="launch vLLM from a pinned JSON config, holding the GPU UUID flock")
    p.add_argument("--config", required=True)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--gpu-index", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("audit", help="inspect the installed vLLM source and write capabilities.json")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-commit", default=None)
    p.add_argument("--gpu-smoke-ar", default=None)
    p.add_argument("--gpu-smoke-ngram", default=None)
    p.add_argument("--lifecycle", default=None)
    p.add_argument("--correctness", default=None)
    p.add_argument("--dflash-wp5", default=None)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("fixture-check", help="verify the frozen 32+8+8 fixture manifest and hashes")
    p.add_argument("--manifest", required=True)
    p.add_argument("--skip-hash-check", action="store_true")
    p.set_defaults(func=cmd_fixture_check)

    p = sub.add_parser("offline", help="run AR or N-Gram offline greedy correctness (token IDs from LLM.chat)")
    p.add_argument("--mode", choices=["ar", "ngram"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", default="serving-offline")
    p.add_argument("--vllm-version", default="0.22.1")
    p.add_argument("--code-commit", default=os.environ.get("VLLM_SOURCE_COMMIT", "unknown-wheel-build"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    p.add_argument("--prompt-lookup-min", type=int, default=2)
    p.add_argument("--prompt-lookup-max", type=int, default=2)
    p.add_argument("--min-pixels", type=int, default=200704)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="smoke only: limit fixtures")
    p.add_argument("--skip-hash-check", action="store_true")
    p.add_argument("--attention-backend", default="FLASH_ATTN")
    p.add_argument("--cudagraph-mode", default="FULL_AND_PIECEWISE")
    p.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    p.set_defaults(func=cmd_offline)

    p = sub.add_parser("compare", help="compare AR repeats and N-Gram transcripts")
    p.add_argument("--ar", nargs="+", required=True)
    p.add_argument("--ngram", nargs="+", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("measure", help="measure SSE stream, cancellation, C=1/2/4 and latency")
    p.add_argument("--mode", choices=["ar", "ngram"], required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--model-path", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", default="serving-measure")
    p.add_argument("--vllm-version", default="0.22.1")
    p.add_argument("--code-commit", default=os.environ.get("VLLM_SOURCE_COMMIT", "unknown-wheel-build"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    p.add_argument("--prompt-lookup-min", type=int, default=2)
    p.add_argument("--prompt-lookup-max", type=int, default=2)
    p.add_argument("--min-pixels", type=int, default=200704)
    p.add_argument("--max-pixels", type=int, default=1003520)
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--devices-json", default=None)
    p.add_argument("--server-log", default=None)
    p.add_argument(
        "--allocation-start-epoch",
        type=float,
        default=None,
        help="Unix timestamp when the GPU allocation/server launch began; GPU-hours include load and warmup",
    )
    p.add_argument("--cancel-window-s", type=float, default=10.0)
    p.add_argument("--cancel-max-tokens", type=int, default=512)
    p.add_argument("--request-timeout-s", type=float, default=300.0)
    p.add_argument("--health-timeout-s", type=float, default=300.0)
    p.add_argument("--attention-backend", default="FLASH_ATTN")
    p.add_argument("--cudagraph-mode", default="FULL_AND_PIECEWISE")
    p.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--skip-hash-check", action="store_true")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("summary", help="merge correctness and HTTP measurement documents")
    p.add_argument("--correctness", default=None)
    p.add_argument("--ar-measure", default=None)
    p.add_argument("--ngram-measure", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_summary)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
