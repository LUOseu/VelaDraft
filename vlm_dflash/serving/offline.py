"""Offline vLLM greedy correctness runs (AR vs native N-Gram)."""

from __future__ import annotations

import copy
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from .contracts import sha256_file, write_json_atomic
from .fixtures import build_messages, diagnostic_fixtures, load_manifest, natural_fixtures
from .metrics import diff_metrics, spec_counts
from .records import (
    backend_identity,
    exposure_usage,
    iso_now,
    make_run_record,
    model_identity,
    ngram_draft_identity,
    protocol_payload,
    resource_usage,
)


def _device_info(index: int = 0) -> dict[str, Any]:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    logical_index = index
    if visible and not visible.startswith("GPU-"):
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if str(index) in entries:
            logical_index = entries.index(str(index))
        elif len(entries) == 1 and entries[0].isdigit() and int(entries[0]) == index:
            logical_index = 0
    props = torch.cuda.get_device_properties(logical_index)
    uuid = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        uuid = out or None
    except Exception:
        pass
    return {
        "index": index,
        "name": props.name,
        "uuid": uuid,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def metrics_dict(llm: Any) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Convert vLLM's Metric objects into the same shape as a /metrics scrape."""

    parsed: dict[str, list[tuple[dict[str, str], float]]] = {}
    for metric in llm.get_metrics() or []:
        name = getattr(metric, "name", None)
        if not name:
            continue
        name = name[:-6] if name.endswith("_total") else name
        values = getattr(metric, "values", None)
        if values is not None:
            for position, value in enumerate(values):
                parsed.setdefault(name, []).append(({"position": str(position)}, float(value)))
            continue
        value = getattr(metric, "value", None)
        if value is None:
            continue
        labels = getattr(metric, "labels", None) or {}
        if not isinstance(labels, dict):
            labels = {}
        parsed.setdefault(name, []).append((labels, float(value)))
    return parsed


def _spec_config(mode: str, args) -> Optional[dict[str, Any]]:
    if mode == "ar":
        return None
    if mode == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": int(args.num_speculative_tokens),
            "prompt_lookup_min": int(args.prompt_lookup_min),
            "prompt_lookup_max": int(args.prompt_lookup_max),
        }
    raise ValueError(f"unknown mode {mode!r}")


def _engine_applied_evidence(llm: Any, backend: dict[str, Any]) -> None:
    """Mark backend labels as applied only when the engine exposes its config."""

    engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    if vllm_config is None:
        return
    observed: dict[str, Any] = {}
    model_config = getattr(vllm_config, "model_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if model_config is not None:
        for field in ("dtype", "max_model_len"):
            value = getattr(model_config, field, None)
            if value is not None:
                observed[field] = str(value)
    if speculative_config is not None:
        observed["speculative_method"] = getattr(speculative_config, "method", None)
        observed["num_speculative_tokens"] = getattr(speculative_config, "num_spec_tokens", None)
    backend["labels_source"] = "engine_object"
    backend["applied_evidence"] = True
    backend["applied_config"] = observed


def run_offline(args) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from .measure import GpuSampler

    # ``LLM`` selects a physical device from the process environment.  Keep
    # the requested physical index for identity/evidence and use the logical
    # index only for torch's device properties.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_index))
    manifest = load_manifest(args.manifest, verify_hashes=not args.skip_hash_check)
    fixtures = natural_fixtures(manifest) + diagnostic_fixtures(manifest)
    if args.limit:
        fixtures = fixtures[: int(args.limit)]
    model = str(Path(args.model).resolve())
    target = model_identity(model, precision=args.dtype)
    spec = _spec_config(args.mode, args)
    draft = ngram_draft_identity(args.vllm_version, target, spec) if args.mode == "ngram" else None
    devices = [_device_info(args.gpu_index)]
    backend = backend_identity(
        vllm_version=args.vllm_version,
        code_commit=args.code_commit,
        devices=devices,
        attention_backend=args.attention_backend,
        cudagraph_mode=args.cudagraph_mode,
    )
    protocol = protocol_payload(
        model_path=model,
        mode=args.mode,
        spec_config=copy.deepcopy(spec),
        sampling={"temperature": 0.0, "max_tokens": int(args.max_new_tokens), "ignore_eos": False},
        pixel_bounds={"min_pixels": int(args.min_pixels), "max_pixels": int(args.max_pixels)},
        max_model_len=int(args.max_model_len),
        prefix_caching=False,
        backend_version=args.vllm_version,
        timing_boundary="offline_llm_chat_single_request_wall",
        stream=False,
        async_scheduling=args.async_scheduling,
    )
    llm_kwargs: dict[str, Any] = dict(
        model=model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": int(args.min_pixels), "max_pixels": int(args.max_pixels)},
        seed=0,
        disable_log_stats=False,
    )
    if spec is not None:
        llm_kwargs["speculative_config"] = copy.deepcopy(spec)
    if args.async_scheduling is not None:
        llm_kwargs["async_scheduling"] = bool(args.async_scheduling)
    sampler = GpuSampler(args.gpu_index, interval_s=0.5).start()
    started_load = time.perf_counter()
    llm = LLM(**llm_kwargs)
    _engine_applied_evidence(llm, backend)
    load_s = time.perf_counter() - started_load
    sampling = SamplingParams(temperature=0.0, max_tokens=int(args.max_new_tokens))
    records = []
    raw_requests = []
    for repeat in range(int(args.repeats)):
        for fixture in fixtures:
            body = build_messages(fixture)
            before = metrics_dict(llm)
            started = time.perf_counter()
            try:
                outputs = llm.chat([body], sampling_params=sampling, use_tqdm=False)
                wall_s = time.perf_counter() - started
                after = metrics_dict(llm)
                output = outputs[0]
                completion = output.outputs[0]
                token_ids = [int(x) for x in completion.token_ids]
                text = completion.text
                finish_reason = completion.finish_reason
                prompt_len = len(output.prompt_token_ids or [])
                error = None
            except Exception as exc:  # keep the other fixtures running
                wall_s = time.perf_counter() - started
                after = metrics_dict(llm)
                token_ids, text, finish_reason, prompt_len = [], "", None, None
                error = f"{type(exc).__name__}: {exc}"
            diff = diff_metrics(before, after)
            spec_delta = diff["spec"]
            if args.mode == "ngram":
                raw_counts = {
                    "emitted": len(token_ids),
                    "proposed": spec_delta.get("draft_tokens"),
                    "accepted": spec_delta.get("accepted_tokens"),
                    "verify_rounds": spec_delta.get("drafts"),
                    "target_forward_calls": None,
                    "draft_forward_calls": 0,
                    "acceptance_length": spec_delta.get("acceptance_length"),
                    "acceptance_rate": spec_delta.get("acceptance_rate"),
                    "accepted_per_position": spec_delta.get("accepted_per_position"),
                    "unavailable": {
                        "target_forward_calls": "vLLM 0.22.1 exposes aggregate spec counters but no target-forward-call counter",
                    },
                }
            else:
                raw_counts = {
                    "emitted": len(token_ids),
                    "proposed": 0,
                    "accepted": 0,
                    "verify_rounds": 0,
                    "target_forward_calls": None,
                    "draft_forward_calls": 0,
                    "unavailable": {
                        "target_forward_calls": "AR baseline does not expose per-request target forward calls",
                    },
                    "structural_zero_note": "AR has no speculative path; propose/accept/verify zeros are structural, not engine counters",
                }
            missing = {}
            if raw_counts["target_forward_calls"] is None:
                missing["raw_counts.target_forward_calls"] = raw_counts["unavailable"]["target_forward_calls"]
            if draft is None:
                missing["draft_identity"] = "AR baseline has no draft model or proposal path"
            usage = exposure_usage(
                wall_s=wall_s,
                devices=devices,
                peak_memory_mib={"process_sampler": sampler.peak().get("memory_mib", 0)},
                reason=(
                    "offline request wall-time is retained as exposure; the additive "
                    "GPU-hours source is the document allocation ledger"
                ),
            )
            record = make_run_record(
                run_id=args.run_id,
                request_id=f"{fixture.request_id}-r{repeat}",
                group_id=f"offline:{args.mode}:rep{repeat}",
                target_identity=target,
                draft_identity=draft,
                backend_identity=backend,
                protocol_payload=protocol,
                correctness_status="failed" if error else "not_checked",
                measurement_scope={
                    "surface": "offline_llm_chat",
                    "mode": args.mode,
                    "repeat_index": repeat,
                    "concurrency": 1,
                    "stream": False,
                    "prefix_caching": False,
                    "max_new_tokens": int(args.max_new_tokens),
                    "fixture_category": fixture.category,
                    "request_budget_verified": True,
                },
                raw_counts=raw_counts,
                latency={"total_s": wall_s, "ttft_s": None},
                resource_usage_value=usage,
                missing_reasons=missing,
                category=fixture.category,
                prompt=fixture.prompt,
                prompt_sha256=fixture.prompt_sha256(),
                prompt_token_count=prompt_len,
                image_sha256=fixture.image_sha256,
                token_ids=token_ids,
                text=text,
                finish_reason=finish_reason,
                repeat_index=repeat,
                error=error,
                spec_counts=spec_delta,
                request_control={
                    "source": "offline_sampling_params",
                    "max_tokens": int(args.max_new_tokens),
                    "temperature": 0.0,
                    "ignore_eos": False,
                    "stream": False,
                },
            )
            records.append(record)
            raw_requests.append({"request_id": record["request_id"], "category": fixture.category})
    request_wall_s = sum(float(record["latency"]["total_s"]) for record in records)
    allocation_wall_s = float(load_s) + request_wall_s
    sampler.stop()
    peak_memory_mib: dict[str, int] = {"device": sampler.peak().get("memory_mib", 0)}
    allocation_usage = resource_usage(
        wall_s=allocation_wall_s,
        gpu_count=1,
        devices=devices,
        peak_memory_mib=peak_memory_mib,
    )
    summary = {
        "run_id": args.run_id,
        "mode": args.mode,
        "vllm_version": args.vllm_version,
        "model": model,
        "speculative_config": spec,
        "load_s": load_s,
        "request_wall_s": request_wall_s,
        "allocation_wall_s": allocation_wall_s,
        "peak_memory_mib": peak_memory_mib,
        "num_requests": len(records),
        "num_errors": sum(1 for record in records if record.get("error")),
        "total_output_tokens": sum(int(record["raw_counts"]["emitted"]) for record in records),
        "total_spec_drafts": sum(int(record["raw_counts"]["verify_rounds"] or 0) for record in records),
        "total_proposed_tokens": sum(int(record["raw_counts"]["proposed"] or 0) for record in records),
        "total_accepted_tokens": sum(int(record["raw_counts"]["accepted"] or 0) for record in records),
        "gpu_hours": allocation_usage["gpu_hours"],
    }
    document = {
        "schema_version": "veladraft-serving-offline-v1",
        "created_at": iso_now(),
        "meta": {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "model": model,
            "mode": args.mode,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "num_speculative_tokens": args.num_speculative_tokens,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
            "repeats": args.repeats,
            "load_s": load_s,
            "request_wall_s": request_wall_s,
            "allocation_wall_s": allocation_wall_s,
            "device": devices,
            "speculative_config": spec,
            "stream": False,
            "run_plan": {
                "requests": [
                    {
                        "request_id": fixture.request_id,
                        "category": fixture.category,
                        "prompt_sha256": fixture.prompt_sha256(),
                        "image_sha256": fixture.image_sha256,
                    }
                    for fixture in fixtures
                ],
                "repeats": int(args.repeats),
            },
        },
        "resource_usage": allocation_usage,
        "resource_semantics": {
            "allocation_gpu_hours_additive": True,
            "request_gpu_hours_additive": False,
        },
        "summary": summary,
        "records": records,
    }
    if args.out:
        write_json_atomic(args.out, document)
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return document
