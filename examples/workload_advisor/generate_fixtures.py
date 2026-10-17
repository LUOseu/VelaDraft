#!/usr/bin/env python3
"""Regenerate the synthetic workload-advisor fixtures.

Everything here is deterministic: rerunning the script must reproduce the
same bytes (tests check this). The records are CPU fixtures, never real GPU
measurements, and every record carries ``synthetic=true``.
"""

from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vlm_dflash.workload_advisor.records import build_latency, build_raw_counts, build_resource_usage, make_record
from vlm_dflash.workload_advisor.schema import (
    protocol_payload,
    protocol_scope_hash,
    request_protocol_hash,
    sha256_file,
    sha256_json,
)

BASE = Path(__file__).resolve().parent
MINI = BASE / "mini"
FIXTURES = BASE / "fixtures"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

TASKS = ("invoice_fields", "chart_qa", "general_vqa")
REFERENCES = {
    "invoice_fields": {"total": "12.00", "currency": "USD"},
    "chart_qa": "42",
    "general_vqa": "blue",
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_workload() -> dict:
    media = MINI / "media"
    media.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = []
    for index in range(30):
        task = TASKS[index % len(TASKS)]
        split = "tuning" if index < 6 else "holdout"
        group_id = f"grp-{index:02d}"
        request_id = f"req-{index:02d}"
        image_name = f"{group_id}.png"
        (media / image_name).write_bytes(PNG)
        if task == "invoice_fields":
            prompt = "Extract the invoice total and currency as JSON."
        elif task == "chart_qa":
            prompt = "What value does the chart show?"
        else:
            prompt = "What color is the object?"
        rows.append(
            {
                "request_id": request_id,
                "group_id": group_id,
                "task": task,
                "messages": [{"role": "user", "content": prompt}],
                "media": [{"type": "image", "path": image_name}],
                "split": split,
            }
        )
        labels.append({"request_id": request_id, "reference": REFERENCES[task], "task": task})
    rows_path = MINI / "rows.jsonl"
    labels_path = MINI / "labels.jsonl"
    _write_jsonl(rows_path, rows)
    _write_jsonl(labels_path, labels)
    manifest = {
        "schema_version": "veladraft-platform-v1",
        "kind": "workload_manifest",
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "workload_id": "mini-enterprise-v1",
        "description": "Synthetic fixture: invoice fields, chart QA and general VQA.",
        "media_root": "media",
        "rows_path": "rows.jsonl",
        "labels_path": "labels.jsonl",
        "hashes": {
            "rows_sha256": sha256_file(rows_path),
            "labels_sha256": sha256_file(labels_path),
        },
        "target": {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_path": "synthetic/not-a-real-checkpoint",
            "revision": "synthetic-rev-1",
            "weights_sha256": "0" * 64,
            "tokenizer": "Qwen/Qwen2.5-VL-3B-Instruct",
            "processor": "Qwen/Qwen2.5-VL-3B-Instruct",
            "chat_template": "qwen2_5_vl",
            "dtype": "bfloat16",
            "structure": "qwen2_5_vl",
            "vision": {"min_pixels": 200704, "max_pixels": 12845056},
            "local_files_only": True,
        },
        "draft_candidates": [
            {"candidate_id": "ar", "method": "ar", "draft_identity": None},
            {
                "candidate_id": "dflash-exact",
                "method": "sd",
                "algorithm": "dflash",
                "checkpoint": "synthetic/dflash.pt",
                "block_size": 8,
                "draft_identity": {"source": "synthetic", "recipe": "dflash", "revision": "fixture"},
            },
            {
                "candidate_id": "dfly-fast-wrong",
                "method": "sd",
                "algorithm": "dfly",
                "checkpoint": "synthetic/dfly.pt",
                "block_size": 8,
                "draft_identity": {"source": "synthetic", "recipe": "dfly", "revision": "fixture"},
            },
        ],
        "processor": {"revision": "synthetic-rev-1", "chat_template": "qwen2_5_vl"},
        "decode": {
            "temperature": 0,
            "max_new_tokens": 128,
            "stop_ids": [151645, 151643],
            "eos_policy": "natural_eos_first_stop",
            "verification": "sequential",
        },
        "backend": {"name": "hf-transformers", "version": "synthetic", "device_topology": ["synthetic-GPU-0"]},
        "seed": 7,
        "resolution": {"rule": "processor_default", "min_pixels": 200704, "max_pixels": 12845056},
        "scorers": {
            "invoice_fields": {"name": "field_f1", "version": "1", "threshold": 0.8},
            "chart_qa": {"name": "normalized_exact_match", "version": "1", "threshold": 1.0},
            "general_vqa": {"name": "normalized_exact_match", "version": "1", "threshold": 1.0},
        },
        "quality_thresholds": {"invoice_fields": 0.8, "chart_qa": 1.0, "general_vqa": 1.0},
        "measurement_scope": "local_client",
        "timing": {"boundary": "request_ready_to_committed_output"},
        "concurrency": 1,
        "bucket_rules": {"target_V": [256, 512, 1024, 2048], "ar_R": [16, 32, 64, 96, 128], "per_bucket_min_groups": 20},
        "splits": {"tuning": 6, "holdout": 24},
        "synthetic": True,
    }
    _write_json(MINI / "manifest.json", manifest)
    return manifest


def _identities(manifest: dict) -> tuple[dict, dict]:
    target = manifest["target"]
    target_identity = {
        "source": "synthetic",
        "model_id": target["model_id"],
        "model_path": target["model_path"],
        "revision": target["revision"],
        "weights_sha256": target["weights_sha256"],
        "tokenizer": target["tokenizer"],
        "processor": target["processor"],
        "chat_template": target["chat_template"],
        "dtype": target["dtype"],
        "structure": target["structure"],
        "vision": target["vision"],
    }
    backend_identity = {
        "name": "hf-transformers",
        "version": "synthetic",
        "code_commit": "synthetic-fixture",
        "device_topology": {"devices": [{"uuid": "synthetic-GPU-0", "name": "Synthetic A6000"}], "concurrency": 1},
    }
    return target_identity, backend_identity


def _records(manifest: dict, *, speed: dict[str, float], wrong: bool, prefix: str) -> list[dict]:
    rows = [json.loads(line) for line in (MINI / "rows.jsonl").read_text().splitlines()]
    holdout = [row for row in rows if row["split"] == "holdout"]
    base_protocol = protocol_payload(manifest, sha256_file(MINI / "rows.jsonl"), sha256_file(MINI / "labels.jsonl"))
    scope = protocol_scope_hash(base_protocol)
    target_identity, backend_identity = _identities(manifest)
    records: list[dict] = []
    candidates = [
        ("ar", "ar", None),
        ("dflash-exact", "sd", {"source": "synthetic", "recipe": "dflash", "revision": "fixture", "algorithm": "dflash", "checkpoint": "synthetic/dflash.pt"}),
        ("dfly-fast-wrong", "sd", {"source": "synthetic", "recipe": "dfly", "revision": "fixture", "algorithm": "dfly", "checkpoint": "synthetic/dfly.pt"}),
    ]
    for session in range(3):
        for index, row in enumerate(holdout):
            rng = random.Random(1000 + index * 17 + session)
            task_factor = {"invoice_fields": 1.0, "chart_qa": 0.9, "general_vqa": 1.1}[row["task"]]
            ar_seconds = 1.0 * task_factor * (0.95 + rng.random() * 0.1)
            R = 24 + (index * 7 + session * 3) % 80
            V = 300 + (index * 37) % 1000
            for candidate_id, method, draft_identity in candidates:
                factor = speed[candidate_id] if candidate_id in speed else 1.0
                wall = ar_seconds * factor * (0.97 + rng.random() * 0.06)
                counts = build_raw_counts(
                    emitted_tokens=R,
                    proposed_draft_tokens=None if method == "ar" else R * 2,
                    accepted_draft_tokens=None if method == "ar" else int(R * 1.8),
                    verify_rounds=None if method == "ar" else max(1, R // 2),
                    target_forward_calls=None if method == "ar" else max(1, R // 2),
                    draft_forward_calls=None if method == "ar" else max(1, R // 2),
                    method=method,
                )
                latency, reasons = build_latency(e2e_s=wall)
                reasons["raw_counts.target_forward_calls"] = "synthetic_fixture"
                reasons["visual_tokens.draft_V"] = "synthetic_fixture"
                failed = method == "sd" and wrong and candidate_id == "dfly-fast-wrong" and index % 3 == 0
                status = "verified_greedy_exact" if not failed else "failed"
                correctness = {
                    "reference_hash": sha256_json({"request": row["request_id"], "tokens": R}),
                    "candidate_hash": sha256_json({"request": row["request_id"], "candidate": candidate_id, "tokens": R}),
                    "input_sha256": sha256_json({"row": row["request_id"], "media": row["media"]}),
                    "n_requests": 1,
                    "failed_request_ids": [row["request_id"]] if failed else [],
                    "compared_tokens": R,
                }
                quality = {
                    "evaluated": True,
                    "scorer": {"name": "fixture", "version": "1"},
                    "score": 0.0 if failed else 1.0,
                    "threshold": 1.0,
                    "status": "fail" if failed else "pass",
                }
                record = make_record(
                    run_id=f"{prefix}-run",
                    request_id=row["request_id"],
                    group_id=row["group_id"],
                    task=row["task"],
                    split=row["split"],
                    method=method,
                    candidate_id=candidate_id,
                    session_id=f"{prefix}-s{session}",
                    repeat_index=session,
                    execution_order=index * 3 + session,
                    target_identity=target_identity,
                    draft_identity=draft_identity,
                    backend_identity=backend_identity,
                    protocol_hash=request_protocol_hash(base_protocol, {"request_id": row["request_id"], "group_id": row["group_id"], "task": row["task"], "messages_sha256": sha256_json(row["messages"]), "media_sha256": [sha256_file(MINI / "media" / row["media"][0]["path"])]}),
                    protocol_scope_hash=scope,
                    correctness_status=status,
                    acceptance_policy="strict_greedy_token_exact" if method == "sd" else "not_applicable",
                    correctness=correctness,
                    quality=quality,
                    output_tokens_R=R,
                    visual_tokens={"target_V": V, "draft_V": None},
                    measurement_scope="local_client",
                    raw_counts=counts,
                    latency=latency,
                    resource_usage=build_resource_usage(
                        all_allocated_gpu_hours=0.002,
                        measured_run_gpu_hours=0.0015,
                        devices=[{"uuid": "synthetic-GPU-0", "name": "Synthetic A6000", "peak_allocated_bytes": 20_000_000_000, "peak_reserved_bytes": 21_000_000_000, "peak_scope": "request"}],
                        gpu_memory_fit=True,
                        note="synthetic fixture; not a real device measurement",
                    ),
                    error={"type": "token_mismatch", "stage": "correctness", "message": "synthetic wrong candidate"} if failed else None,
                    synthetic=True,
                    missing_reasons=reasons,
                    extra={"concurrency": 1},
                )
                records.append(record)
    return records


def main() -> None:
    manifest = make_workload()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        FIXTURES / "records.jsonl",
        _records(manifest, speed={"ar": 1.0, "dflash-exact": 0.70, "dfly-fast-wrong": 0.48}, wrong=True, prefix="fixture"),
    )
    _write_jsonl(
        FIXTURES / "records_ar_better.jsonl",
        _records(manifest, speed={"ar": 1.0, "dflash-exact": 1.20, "dfly-fast-wrong": 1.15}, wrong=False, prefix="ar-better"),
    )
    legacy = {
        "format": "vlm-dflash-benchmark-v1",
        "algorithm": "dflash",
        "run_dir": "legacy/run",
        "created": "2026-01-01T00:00:00Z",
        "protocol": {"temperature": 0.0, "verification": "sequential", "checkpoint": "ckpt_final.pt"},
        "workloads": {
            "textvqa": {
                "status": "measured",
                "num_prompts": 128,
                "max_new_tokens": 128,
                "al": 1.62,
                "acceptance_rate": 0.09,
                "token_exact": True,
                "wall_seconds_per_sample": 1.20,
                "target_only_wall_seconds_per_sample": 1.55,
                "speedup_vs_target_only": 1.29,
            }
        },
    }
    _write_json(FIXTURES / "legacy_report.json", legacy)
    print(f"wrote fixtures under {BASE}")


if __name__ == "__main__":
    main()
