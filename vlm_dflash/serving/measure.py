"""HTTP SSE / cancellation / concurrency measurement for a live vLLM server."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from .contracts import sha256_file, write_json_atomic
from .fixtures import Fixture, boundary_fixtures, build_messages, diagnostic_fixtures, load_manifest, natural_fixtures, request_body
from .metrics import active_requests, diff_metrics, parse_prometheus, summarize_latencies
from .openai_api import ChatResult, ServingClient, result_timing_fields
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


class GpuSampler:
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[dict[str, float]] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                ).strip()
                parts = [part.strip() for part in out.split(",")]
                if len(parts) >= 2:
                    self.samples.append({"memory_mib": float(parts[0]), "utilization_pct": float(parts[1])})
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def peak(self) -> dict[str, int]:
        return {
            "memory_mib": int(max((sample["memory_mib"] for sample in self.samples), default=0)),
            "utilization_pct": int(max((sample["utilization_pct"] for sample in self.samples), default=0)),
        }


def _metrics(client: ServingClient) -> dict[str, Any]:
    return parse_prometheus(client.metrics_text())


def _active(client: ServingClient) -> dict[str, float]:
    return active_requests(_metrics(client))


def _wait_reclaimed(
    client: ServingClient,
    baseline_total: float,
    timeout_s: float = 10.0,
) -> tuple[Optional[float], dict[str, float], list[dict[str, float]]]:
    started = time.monotonic()
    last = _active(client)
    samples = [{"t_s": 0.0, "total": float(last.get("total", 0.0))}]
    while time.monotonic() - started < timeout_s:
        if last.get("total", 0) <= baseline_total:
            return time.monotonic() - started, last, samples
        time.sleep(0.25)
        last = _active(client)
        samples.append({"t_s": time.monotonic() - started, "total": float(last.get("total", 0.0))})
    return None, last, samples


def _server_log_matches(path: Optional[str], request_id: str) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {"available": False}
    matches = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if request_id in line:
                    matches.append(line.strip()[-500:])
    except OSError:
        return {"available": False}
    return {
        "available": True,
        "path": path,
        "matched_lines": matches[-10:],
        "abort_or_finish": [line for line in matches if "abort" in line.lower() or "finish" in line.lower()],
    }


def _server_log_config_evidence(
    path: Optional[str],
    *,
    vllm_version: str,
    attention_backend: Optional[str],
    cudagraph_mode: Optional[str],
    mode: str,
    spec_config: Optional[dict[str, Any]],
    async_scheduling: Optional[bool],
) -> dict[str, Any]:
    """Derive applied backend labels from the engine's own startup log."""

    if not path or not Path(path).is_file():
        return {
            "available": False,
            "applied_evidence": False,
            "labels_source": "cli_declared_unverified",
            "checks": {},
            "missing_reasons": ["server log unavailable; CLI backend labels are not applied evidence"],
        }
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "available": False,
            "applied_evidence": False,
            "labels_source": "cli_declared_unverified",
            "checks": {},
            "missing_reasons": [f"server log unreadable: {type(exc).__name__}: {exc}"],
        }

    checks: dict[str, bool] = {"vllm_version": f"v{vllm_version}" in text}
    if attention_backend:
        checks["attention_backend"] = bool(
            f"Using {attention_backend} attention backend" in text
            or f"AttentionBackendEnum.{attention_backend}" in text
        )
    if cudagraph_mode:
        checks["cudagraph_mode"] = f"CUDAGraphMode.{cudagraph_mode}" in text
    if mode == "ngram" and spec_config:
        checks["speculative_method"] = "SpeculativeConfig(method='ngram'" in text
        checks["num_speculative_tokens"] = f"num_spec_tokens={int(spec_config['num_speculative_tokens'])}" in text
    elif mode == "ar":
        checks["speculative_config"] = "speculative_config=None" in text
    if async_scheduling is False:
        checks["async_scheduling"] = "Asynchronous scheduling is disabled." in text
    elif async_scheduling is True:
        checks["async_scheduling"] = "Asynchronous scheduling is enabled." in text

    missing = [f"server log does not evidence {name}" for name, found in checks.items() if not found]
    applied = bool(checks) and not missing
    return {
        "available": True,
        "applied_evidence": applied,
        "labels_source": "server_log_config_echo" if applied else "cli_declared_unverified",
        "checks": checks,
        "missing_reasons": missing,
    }


def _sampled_resource(wall_s: float, sampler: GpuSampler, devices: list[dict[str, Any]]) -> dict[str, Any]:
    peak = sampler.peak()
    return resource_usage(
        wall_s=wall_s,
        gpu_count=1,
        devices=devices,
        peak_memory_mib={"gpu0": peak["memory_mib"]},
    )


def _make_fixture_request(
    fixture: Fixture,
    model: str,
    *,
    stream: bool = True,
    request_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    body = request_body(
        fixture,
        model=model,
        max_tokens=max_tokens,
        stream=stream,
        return_token_ids=True,
        request_id=request_id or fixture.request_id,
    )
    if ignore_eos:
        body["ignore_eos"] = True
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _run_one(
    client: ServingClient,
    fixture: Fixture,
    *,
    model: str,
    mode: str,
    request_id: str,
    cancel_after_first: bool = False,
    max_tokens: Optional[int] = None,
    ignore_eos: bool = False,
) -> ChatResult:
    body = _make_fixture_request(
        fixture,
        model,
        stream=True,
        request_id=request_id,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    return client.collect_stream(body, mode=mode, request_id=request_id, cancel_after_first_content=cancel_after_first)


def _request_control(
    fixture: Fixture,
    *,
    model: str,
    max_tokens: Optional[int],
    ignore_eos: bool = False,
) -> dict[str, Any]:
    return {
        "source": "actual_request_body",
        "model": model,
        "max_tokens": int(max_tokens if max_tokens is not None else fixture.max_tokens),
        "fixture_max_tokens": int(fixture.max_tokens),
        "ignore_eos": bool(ignore_eos),
        "temperature": 0.0,
        "stream": True,
        "return_token_ids": True,
    }


def _run_cancel_probe(
    client: ServingClient,
    fixture: Fixture,
    *,
    model: str,
    mode: str,
    request_id: str,
    concurrency: int,
    session: int,
    timeout_s: float,
    max_tokens: int,
    server_log: Optional[str],
) -> dict[str, Any]:
    """Cancel one long-enough request from an idle baseline.

    The main C>1 windows are concurrent, so polling their active gauge after
    every future completed cannot attribute a decrease to the disconnect.  This
    probe runs after the window drains, records the gauge immediately after the
    client closes the stream, and then measures reclamation from that baseline.
    """

    before = _active(client)
    baseline_total = float(before.get("total", 0.0))
    during_samples: list[dict[str, float]] = []
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.is_set():
            try:
                during_samples.append(_active(client))
            except Exception:
                pass
            stop_monitor.wait(0.05)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        result = _run_one(
            client,
            fixture,
            model=model,
            mode=mode,
            request_id=request_id,
            cancel_after_first=True,
            max_tokens=max_tokens,
            ignore_eos=True,
        )
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=1.0)
    immediate = _active(client)
    reclaimed_s, after, samples = _wait_reclaimed(client, baseline_total, timeout_s=timeout_s)
    observed_before_close = any(float(sample.get("total", 0.0)) > baseline_total for sample in during_samples)
    observed_after_close = float(immediate.get("total", 0.0)) > baseline_total
    observed_in_flight = bool(result.cancelled and not result.error and (observed_before_close or observed_after_close))
    log_match = _server_log_matches(server_log, request_id)
    internal_abort_evidence = bool(log_match.get("abort_or_finish"))
    return {
        "mode": mode,
        "concurrency": concurrency,
        "session": session,
        "request_id": request_id,
        "max_tokens": max_tokens,
        "cancelled": result.cancelled,
        "cancel_requested_s": result.cancel_requested_s,
        "cancel_reclaimed_s": reclaimed_s,
        "ignore_eos": True,
        "request_control": {
            "source": "actual_request_body",
            "max_tokens": int(max_tokens),
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
            "probe_kind": "cancel",
        },
        "probe_context": "idle_after_window_drain",
        "concurrent_with_healthy_request": False,
        "observed_in_flight_before_close": observed_before_close,
        "observed_in_flight_after_close": observed_after_close,
        "observed_in_flight": observed_in_flight,
        "baseline_active": before,
        "active_immediately_after_close": immediate,
        "active_after": after,
        "active_samples_during_request": during_samples,
        "active_samples": samples,
        "server_log": log_match,
        "internal_abort_evidence": internal_abort_evidence,
        "cancel_claim_scope": "client_socket_close_and_global_gauge_only",
        "error": result.error,
    }


def _record_from_result(
    *,
    args,
    result: ChatResult,
    fixture: Fixture,
    target: dict[str, Any],
    draft: Optional[dict[str, Any]],
    backend: dict[str, Any],
    protocol: dict[str, Any],
    devices: list[dict[str, Any]],
    sampler: GpuSampler,
    concurrency: int,
    session_index: int,
    raw_spec: Optional[dict[str, Any]],
    spec_attribution: str,
    request_control: dict[str, Any],
    extra_scope: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    timings = result_timing_fields(result)
    if raw_spec is None and args.mode == "ngram":
        raw_counts = {
            "emitted": timings["num_output_tokens"],
            "proposed": None,
            "accepted": None,
            "verify_rounds": None,
            "target_forward_calls": None,
            "draft_forward_calls": 0,
            "unavailable": {
                "proposed": "vLLM spec counters are global; no request tag under concurrency",
                "accepted": "vLLM spec counters are global; no request tag under concurrency",
                "verify_rounds": "vLLM spec counters are global; no request tag under concurrency",
                "target_forward_calls": "vLLM 0.22.1 does not expose target-forward calls",
            },
        }
    elif raw_spec is None:
        raw_counts = {
            "emitted": timings["num_output_tokens"],
            "proposed": 0,
            "accepted": 0,
            "verify_rounds": 0,
            "target_forward_calls": None,
            "draft_forward_calls": 0,
            "unavailable": {"target_forward_calls": "AR baseline does not expose target-forward calls"},
            "structural_zero_note": "AR has no speculative path",
        }
    else:
        raw_counts = {
            "emitted": timings["num_output_tokens"],
            "proposed": raw_spec.get("draft_tokens"),
            "accepted": raw_spec.get("accepted_tokens"),
            "verify_rounds": raw_spec.get("drafts"),
            "target_forward_calls": None,
            "draft_forward_calls": 0,
            "acceptance_length": raw_spec.get("acceptance_length"),
            "acceptance_rate": raw_spec.get("acceptance_rate"),
            "unavailable": {"target_forward_calls": "vLLM 0.22.1 does not expose target-forward calls"},
        }
    missing = {}
    for key in ("proposed", "accepted", "verify_rounds", "target_forward_calls"):
        if raw_counts.get(key) is None:
            missing[f"raw_counts.{key}"] = raw_counts.get("unavailable", {}).get(key, "not visible")
    if draft is None:
        missing["draft_identity"] = "AR baseline has no draft model or proposal path"
    scope = {
        "surface": "openai_chat_completions",
        "mode": args.mode,
        "concurrency": concurrency,
        "session_index": session_index,
        "stream": True,
        "prefix_caching": False,
        "max_new_tokens": int(request_control["max_tokens"]),
        "fixture_max_tokens": int(fixture.max_tokens),
        "fixture_category": fixture.category,
        "timing_boundary": "client_request_start_to_last_nonempty_sse_content_chunk",
        "spec_counter_attribution": spec_attribution,
    }
    if extra_scope:
        scope.update(extra_scope)
    latency = {
        "ttft_s": timings["ttft_s"],
        "e2e_s": timings["e2e_s"],
        "tpot_s": timings["tpot_s"],
        "tpot_unavailable_reason": timings["tpot_unavailable_reason"],
        "client_visible_token_tpot_proxy_s": timings["client_visible_token_tpot_proxy_s"],
        "client_chunk_gap_mean_s": timings["client_chunk_gap_mean_s"],
        "cancel_requested_s": timings["cancel_requested_s"],
    }
    if timings.get("tpot_unavailable_reason"):
        missing["latency.tpot_s"] = timings["tpot_unavailable_reason"]
    success = result.is_successful()
    return make_run_record(
        run_id=args.run_id,
        request_id=result.request_id,
        group_id=f"http:{args.mode}:c{concurrency}:s{session_index}",
        target_identity=target,
        draft_identity=draft,
        backend_identity=backend,
        protocol_payload=protocol,
        correctness_status="not_checked" if success else "failed",
        measurement_scope=scope,
        raw_counts=raw_counts,
        latency=latency,
        resource_usage_value=exposure_usage(
            wall_s=result.e2e_s or 0.0,
            devices=devices,
            peak_memory_mib={"process_sampler": sampler.peak().get("memory_mib", 0)},
            reason=(
                "request wall-time exposure is not an additive GPU allocation; "
                "concurrent requests share the same device interval; use the document allocation ledger"
            ),
        ),
        missing_reasons=missing,
        category=fixture.category,
        prompt=fixture.prompt,
        prompt_sha256=fixture.prompt_sha256(),
        image_sha256=fixture.image_sha256,
        token_ids=result.token_ids,
        text=result.text,
        finish_reason=result.finish_reason,
        response_id=result.response_id,
        num_content_chunks=result.num_content_chunks,
        chunk_token_counts=result.chunk_token_counts,
        chunk_gap_count=len(result.chunk_gaps_s),
        chunk_gaps_s=result.chunk_gaps_s,
        cancelled=result.cancelled,
        error=result.error,
        http_status=result.http_status,
        terminal_status=result.terminal_status,
        done_seen=result.done_seen,
        finish_seen=result.finish_seen,
        truncated=result.truncated,
        protocol_error=result.protocol_error,
        token_ids_complete=result.token_ids_complete,
        request_control=request_control,
    )


def _window_record(
    *,
    args,
    wall_s: float,
    target: dict[str, Any],
    draft: Optional[dict[str, Any]],
    backend: dict[str, Any],
    protocol: dict[str, Any],
    devices: list[dict[str, Any]],
    sampler: GpuSampler,
    concurrency: int,
    session_index: int,
    diff: dict[str, Any],
    results: list[ChatResult],
    extra_scope: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    spec = diff["spec"]
    successful = [result for result in results if result.is_successful()]
    emitted = sum(len(result.token_ids) for result in successful)
    emitted_including_failed = sum(len(result.token_ids) for result in results)
    num_failed = sum(1 for result in results if not result.is_successful() and not result.cancelled)
    num_cancelled = sum(1 for result in results if result.cancelled)
    if args.mode == "ngram":
        raw_counts = {
            "emitted": emitted,
            "proposed": spec.get("draft_tokens"),
            "accepted": spec.get("accepted_tokens"),
            "verify_rounds": spec.get("drafts"),
            "target_forward_calls": None,
            "draft_forward_calls": 0,
            "acceptance_length": spec.get("acceptance_length"),
            "acceptance_rate": spec.get("acceptance_rate"),
            "accepted_per_position": spec.get("accepted_per_position"),
            "unavailable": {"target_forward_calls": "vLLM 0.22.1 does not expose target-forward calls"},
        }
    else:
        raw_counts = {
            "emitted": emitted,
            "proposed": 0,
            "accepted": 0,
            "verify_rounds": 0,
            "target_forward_calls": None,
            "draft_forward_calls": 0,
            "unavailable": {"target_forward_calls": "AR baseline does not expose target-forward calls"},
            "structural_zero_note": "AR has no speculative path",
        }
    e2e = [result.e2e_s for result in results if result.e2e_s is not None]
    ttft = [result.first_content_s for result in results if result.first_content_s is not None]
    scope = {
        "surface": "openai_chat_completions",
        "mode": args.mode,
        "concurrency": concurrency,
        "session_index": session_index,
        "stream": True,
        "prefix_caching": False,
        "requests": len(results),
        "successful_requests": len(successful),
        "window": True,
        "spec_counter_attribution": "global_window",
        "timing_boundary": "session_start_to_last_request_completion",
    }
    if extra_scope:
        scope.update(extra_scope)
    missing = {}
    if raw_counts["target_forward_calls"] is None:
        missing["raw_counts.target_forward_calls"] = "vLLM 0.22.1 does not expose target-forward calls"
    if draft is None:
        missing["draft_identity"] = "AR baseline has no draft model or proposal path"
    return make_run_record(
        run_id=args.run_id,
        request_id=f"window-{args.mode}-c{concurrency}-s{session_index}-{int(time.time() * 1000)}",
        group_id=f"http:{args.mode}:c{concurrency}:s{session_index}",
        target_identity=target,
        draft_identity=draft,
        backend_identity=backend,
        protocol_payload=protocol,
        correctness_status="failed" if num_failed else "not_checked",
        measurement_scope=scope,
        raw_counts=raw_counts,
        latency={
            "total_s": wall_s,
            "ttft_p50_s": summarize_latencies(ttft)["p50_s"],
            "ttft_p95_s": summarize_latencies(ttft)["p95_s"],
            "e2e_p50_s": summarize_latencies(e2e)["p50_s"],
            "e2e_p95_s": summarize_latencies(e2e)["p95_s"],
        },
        resource_usage_value=exposure_usage(
            wall_s=wall_s,
            devices=devices,
            peak_memory_mib={"process_sampler": sampler.peak().get("memory_mib", 0)},
            reason=(
                "window wall-time exposure is not an additive allocation interval; "
                "the document allocation ledger is the only additive GPU-hours source"
            ),
            scope="window_exposure",
        ),
        missing_reasons=missing,
        num_requests=len(results),
        num_success=len(successful),
        num_successful=len(successful),
        num_errors=num_failed,
        num_failed=num_failed,
        num_cancelled=num_cancelled,
        completion_rate=(len(successful) / len(results)) if results else None,
        output_tokens=emitted,
        output_tokens_including_failed=emitted_including_failed,
        throughput_tokens_s=(emitted / wall_s) if wall_s > 0 else None,
        throughput_scope="successful_output_tokens_per_window_wall_s",
        active_after=diff.get("active_after"),
        gauges_after=diff.get("gauges_after"),
    )


def run_measure(args) -> dict[str, Any]:
    manifest = load_manifest(args.manifest, verify_hashes=not args.skip_hash_check)
    natural = natural_fixtures(manifest)
    diagnostic = diagnostic_fixtures(manifest)
    boundary = boundary_fixtures(manifest)
    model = args.model
    target = model_identity(args.model_path, precision=args.dtype)
    spec = None if args.mode == "ar" else {
        "method": "ngram",
        "num_speculative_tokens": int(args.num_speculative_tokens),
        "prompt_lookup_min": int(args.prompt_lookup_min),
        "prompt_lookup_max": int(args.prompt_lookup_max),
    }
    draft = ngram_draft_identity(args.vllm_version, target, spec) if spec else None
    devices = json.loads(args.devices_json)
    backend_evidence = _server_log_config_evidence(
        args.server_log,
        vllm_version=args.vllm_version,
        attention_backend=args.attention_backend,
        cudagraph_mode=args.cudagraph_mode,
        mode=args.mode,
        spec_config=spec,
        async_scheduling=args.async_scheduling,
    )
    backend = backend_identity(
        vllm_version=args.vllm_version,
        code_commit=args.code_commit,
        devices=devices,
        attention_backend=args.attention_backend,
        cudagraph_mode=args.cudagraph_mode,
        labels_source=backend_evidence["labels_source"],
        applied_evidence=backend_evidence["applied_evidence"],
    )
    backend["applied_config"] = backend_evidence["checks"]
    backend["applied_evidence_source"] = args.server_log
    protocol = protocol_payload(
        model_path=args.model_path,
        mode=args.mode,
        spec_config=spec,
        sampling={"temperature": 0.0, "max_tokens": int(args.max_new_tokens), "ignore_eos": False},
        pixel_bounds={"min_pixels": int(args.min_pixels), "max_pixels": int(args.max_pixels)},
        max_model_len=int(args.max_model_len),
        prefix_caching=False,
        backend_version=args.vllm_version,
        timing_boundary="client_request_start_to_last_nonempty_sse_content_chunk",
        stream=True,
        async_scheduling=args.async_scheduling,
    )
    client = ServingClient(args.base_url, timeout_s=args.request_timeout_s)
    if not client.wait_for_health(timeout_s=args.health_timeout_s):
        raise RuntimeError(f"server at {args.base_url} did not become healthy")
    sampler = GpuSampler(args.gpu_index).start()
    records: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    cancel_probes: list[dict[str, Any]] = []
    nonstream_probes: list[dict[str, Any]] = []
    boundary_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    measure_started_epoch = time.time()
    warmup_s = 0.0
    try:
        # Warmup (4 requests, as registered in the measurement protocol).
        warm_started = time.perf_counter()
        for fixture in (natural + diagnostic)[:4]:
            _run_one(
                client,
                fixture,
                model=model,
                mode=args.mode,
                request_id=f"warmup-{fixture.request_id}",
                max_tokens=int(args.max_new_tokens),
            )
        warmup_s = time.perf_counter() - warm_started

        # The same request is exercised through both API modes.  The streamed
        # transcript and the non-streamed message must agree on text and stop
        # reason; token IDs are checked only when the non-stream response
        # exposes them.
        probe_fixture = natural[0]
        stream_probe = _run_one(
            client,
            probe_fixture,
            model=model,
            mode=args.mode,
            request_id=f"api-probe-stream-{args.mode}",
            max_tokens=int(args.max_new_tokens),
        )
        nonstream_probe = client.chat(
            _make_fixture_request(
                probe_fixture,
                model,
                stream=False,
                request_id=f"api-probe-nonstream-{args.mode}",
                max_tokens=int(args.max_new_tokens),
            ),
            mode=args.mode,
            request_id=f"api-probe-nonstream-{args.mode}",
        )
        nonstream_probes.append(
            {
                "request_id": probe_fixture.request_id,
                "stream_error": stream_probe.error,
                "nonstream_error": nonstream_probe.error,
                "stream_text": stream_probe.text,
                "nonstream_text": nonstream_probe.text,
                "stream_finish_reason": stream_probe.finish_reason,
                "nonstream_finish_reason": nonstream_probe.finish_reason,
                "text_match": stream_probe.text == nonstream_probe.text,
                "finish_reason_match": stream_probe.finish_reason == nonstream_probe.finish_reason,
                "stream_token_ids": len(stream_probe.token_ids),
                "nonstream_token_ids": len(nonstream_probe.token_ids),
                "ok": bool(
                    not stream_probe.error
                    and not nonstream_probe.error
                    and stream_probe.text == nonstream_probe.text
                    and stream_probe.finish_reason == nonstream_probe.finish_reason
                ),
            }
        )

        # Boundary probes are API validation, not part of the 32-natural stats.
        for fixture in boundary:
            result = _run_one(
                client,
                fixture,
                model=model,
                mode=args.mode,
                request_id=f"boundary-{fixture.request_id}",
                max_tokens=int(args.max_new_tokens),
            )
            boundary_results.append(
                {
                    "request_id": fixture.request_id,
                    "variant": fixture.boundary_variant,
                    "expect": fixture.expect,
                    "http_status": result.http_status,
                    "error": result.error,
                    "text": result.text[:500],
                    "finish_reason": result.finish_reason,
                    "ok": (result.error is None) if fixture.expect == "success" else (result.error is not None),
                }
            )

        for concurrency in [int(x) for x in args.concurrency.split(",")]:
            for session in range(1, int(args.sessions) + 1):
                # 32 natural requests + one normal-end + a same-text/different-image pair.
                # A dedicated cancel probe follows after this timed window drains.
                base = list(natural)
                cancel_fixture = diagnostic[(session - 1) % len(diagnostic)]
                normal_fixture = diagnostic[(session + 1) % len(diagnostic)]
                pair_a = replace(base[0], request_id=f"pair-a-{session}", prompt="Describe the main object in this image in one short sentence.")
                pair_b = replace(base[1], request_id=f"pair-b-{session}", prompt="Describe the main object in this image in one short sentence.")
                requests = base + [normal_fixture, pair_a, pair_b]
                metric_before = _metrics(client)
                metric_session_start = metric_before
                window_started = time.perf_counter()
                results: list[ChatResult] = []

                def one(fixture: Fixture) -> ChatResult:
                    return _run_one(
                        client,
                        fixture,
                        model=model,
                        mode=args.mode,
                        request_id=f"{args.mode}-c{concurrency}-s{session}-{fixture.request_id}",
                        max_tokens=int(args.max_new_tokens),
                    )

                if concurrency == 1:
                    for fixture in requests:
                        result = one(fixture)
                        # Per-request attribution is valid only at C=1.
                        metric_after = _metrics(client)
                        spec_delta = diff_metrics(metric_before, metric_after)["spec"] if args.mode == "ngram" else None
                        record = _record_from_result(
                            args=args,
                            result=result,
                            fixture=fixture,
                            target=target,
                            draft=draft,
                            backend=backend,
                            protocol=protocol,
                            devices=devices,
                            sampler=sampler,
                            concurrency=concurrency,
                            session_index=session,
                            raw_spec=spec_delta,
                            spec_attribution="per_request_c1",
                            request_control=_request_control(
                                fixture,
                                model=model,
                                max_tokens=int(args.max_new_tokens),
                            ),
                            extra_scope={"cancelled_probe": fixture is cancel_fixture},
                        )
                        records.append(record)
                        metric_before = metric_after
                        results.append(result)
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = {pool.submit(one, fixture): fixture for fixture in requests}
                        for future in as_completed(futures):
                            fixture = futures[future]
                            result = future.result()
                            record = _record_from_result(
                                args=args,
                                result=result,
                                fixture=fixture,
                                target=target,
                                draft=draft,
                                backend=backend,
                                protocol=protocol,
                                devices=devices,
                                sampler=sampler,
                                concurrency=concurrency,
                                session_index=session,
                                raw_spec=None,
                                spec_attribution="global_window_only",
                                request_control=_request_control(
                                    fixture,
                                    model=model,
                                    max_tokens=int(args.max_new_tokens),
                                ),
                            )
                            records.append(record)
                            results.append(result)
                wall_s = time.perf_counter() - window_started
                metric_after = _metrics(client)
                window_diff = diff_metrics(metric_session_start, metric_after)
                window = _window_record(
                    args=args,
                    wall_s=wall_s,
                    target=target,
                    draft=draft,
                    backend=backend,
                    protocol=protocol,
                    devices=devices,
                    sampler=sampler,
                    concurrency=concurrency,
                    session_index=session,
                    diff=window_diff,
                    results=results,
                )
                windows.append(window)
                cancel_probes.append(
                    _run_cancel_probe(
                        client,
                        cancel_fixture,
                        model=model,
                        mode=args.mode,
                        request_id=f"{args.mode}-c{concurrency}-s{session}-cancel",
                        concurrency=concurrency,
                        session=session,
                        timeout_s=args.cancel_window_s,
                        max_tokens=args.cancel_max_tokens,
                        server_log=args.server_log,
                    )
                )

        # Same-text/different-image sanity for every concurrency/session window.
        pair_checks = []
        for concurrency in [int(value) for value in args.concurrency.split(",")]:
            for session in range(1, int(args.sessions) + 1):
                suffix_a = f"-pair-a-{session}"
                suffix_b = f"-pair-b-{session}"
                matching = [
                    record
                    for record in records
                    if record["measurement_scope"].get("concurrency") == concurrency
                    and record["measurement_scope"].get("session_index") == session
                    and (record["request_id"].endswith(suffix_a) or record["request_id"].endswith(suffix_b))
                ]
                pair_checks.append(
                    {
                        "concurrency": concurrency,
                        "session": session,
                        "requests": [record["request_id"] for record in matching],
                        "prompts": [record.get("prompt") for record in matching],
                        "image_sha256": [record.get("image_sha256") for record in matching],
                        "distinct_request_ids": len({record["request_id"] for record in matching}) == len(matching),
                        "same_text": len({record.get("prompt") for record in matching}) == 1 if matching else False,
                        "different_images": len({record.get("image_sha256") for record in matching}) == 2 if matching else False,
                        "responses_distinct": len(
                            {(tuple(record.get("token_ids") or []), record.get("text")) for record in matching}
                        )
                        == 2
                        if matching
                        else False,
                        "errors": [record.get("error") for record in matching if record.get("error")],
                        "input_submission_ok": bool(
                            len(matching) == 2
                            and len({record["request_id"] for record in matching}) == 2
                            and len({record.get("prompt") for record in matching}) == 1
                            and len({record.get("image_sha256") for record in matching}) == 2
                            and not any(record.get("error") for record in matching)
                        ),
                        "output_attribution_verified": False,
                        "cross_session_checked": False,
                        "claim_scope": "submitted prompt/image identity only",
                    }
                )
    finally:
        sampler.stop()
        total_s = time.perf_counter() - started

    window_metrics = [window for window in windows]
    ended_epoch = time.time()
    allocation_start_epoch = float(args.allocation_start_epoch) if args.allocation_start_epoch else measure_started_epoch
    allocation_wall_s = max(0.0, ended_epoch - allocation_start_epoch)
    allocation_usage = resource_usage(
        wall_s=allocation_wall_s,
        gpu_count=1,
        devices=devices,
        peak_memory_mib={"gpu": sampler.peak().get("memory_mib", 0)},
    )
    summary = {
        "run_id": args.run_id,
        "mode": args.mode,
        "base_url": args.base_url,
        "warmup_s": warmup_s,
        "measurement_wall_s": total_s,
        "measurement_gpu_hours": round((total_s / 3600.0), 6),
        "allocation_start_epoch": allocation_start_epoch,
        "allocation_end_epoch": ended_epoch,
        "allocation_wall_s": allocation_wall_s,
        "gpu_hours": allocation_usage["gpu_hours"],
        "gpu_hours_source": "allocation_ledger",
        "resource_semantics": {
            "allocation_gpu_hours_additive": True,
            "request_gpu_hours_additive": False,
            "request_gpu_hours_scope": "request_exposure",
            "window_gpu_hours_scope": "window_exposure",
            "peak_memory_scope": "process_sampler_shared_across_requests",
        },
        "timing_semantics": {
            "tpot_definition": "visible token_ids only: (last_content-first_content)/(visible_tokens-1)",
            "client_chunk_gap_mean_definition": "mean gap between non-empty content chunks; not ITL/TPOT",
            "tpot_null_when_token_ids_incomplete": True,
        },
        "records": len(records),
        "windows": len(windows),
        "boundary_results": boundary_results,
        "cancel_probes": cancel_probes,
        "nonstream_probes": nonstream_probes,
        "pair_checks": pair_checks,
        "window_throughput_tokens_s": {
            f"c{window['measurement_scope']['concurrency']}-s{window['measurement_scope']['session_index']}": window.get("throughput_tokens_s")
            for window in window_metrics
        },
        "window_wall_s": {
            f"c{window['measurement_scope']['concurrency']}-s{window['measurement_scope']['session_index']}": window["latency"]["total_s"]
            for window in window_metrics
        },
        "spec_window_totals": {
            "drafts": sum(int(window["raw_counts"].get("verify_rounds") or 0) for window in window_metrics),
            "proposed_tokens": sum(int(window["raw_counts"].get("proposed") or 0) for window in window_metrics),
            "accepted_tokens": sum(int(window["raw_counts"].get("accepted") or 0) for window in window_metrics),
        },
        "window_success": {
            f"c{window['measurement_scope']['concurrency']}-s{window['measurement_scope']['session_index']}": {
                "num_success": window.get("num_success"),
                "num_failed": window.get("num_failed"),
                "num_cancelled": window.get("num_cancelled"),
                "completion_rate": window.get("completion_rate"),
            }
            for window in window_metrics
        },
        "peak_gpu": sampler.peak(),
    }
    document = {
        "schema_version": "veladraft-serving-measure-v1",
        "created_at": iso_now(),
        "meta": {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "model": args.model,
            "mode": args.mode,
            "dtype": args.dtype,
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "sessions": args.sessions,
            "warmup": 4,
            "stream": True,
            "max_new_tokens": args.max_new_tokens,
            "speculative_config": spec,
            "server_log": args.server_log,
            "devices": devices,
            "allocation_start_epoch": allocation_start_epoch,
            "allocation_end_epoch": ended_epoch,
            "allocation_wall_s": allocation_wall_s,
            "request_budget": {
                "max_new_tokens": int(args.max_new_tokens),
                "source": "cli_override_applied_to_actual_request_body",
                "fixture_max_tokens_retained_in_record": True,
                "cancel_probe_exception": {
                    "max_tokens": int(args.cancel_max_tokens),
                    "ignore_eos": True,
                },
            },
        },
        "resource_usage": allocation_usage,
        "summary": summary,
        "windows": windows,
        "records": records,
    }
    if args.out:
        write_json_atomic(args.out, document)
    return document
