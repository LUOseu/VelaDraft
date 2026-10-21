from __future__ import annotations

import importlib.util
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from vlm_dflash.serving import measure
from vlm_dflash.serving.fixtures import Fixture
from vlm_dflash.serving.openai_api import ChatResult
from vlm_dflash.serving.openai_api import result_timing_fields
from vlm_dflash.serving.records import backend_identity, protocol_payload


def _fixture() -> Fixture:
    return Fixture(request_id="probe", category="diagnostic", prompt="repeat", image_path=Path("unused.jpg"))


def test_cancel_probe_observes_in_flight_during_request(monkeypatch):
    state = {"running": False}

    def active(client):
        total = 1.0 if state["running"] else 0.0
        return {"running": total, "waiting": 0.0, "total": total}

    def run_one(*args, **kwargs):
        state["running"] = True
        time.sleep(0.2)
        state["running"] = False
        return ChatResult(
            request_id="probe",
            mode="ar",
            cancelled=True,
            cancel_requested_s=0.1,
        )

    monkeypatch.setattr(measure, "_active", active)
    monkeypatch.setattr(
        measure,
        "_run_one",
        run_one,
    )

    probe = measure._run_cancel_probe(
        object(),
        _fixture(),
        model="m",
        mode="ar",
        request_id="probe",
        concurrency=4,
        session=1,
        timeout_s=1.0,
        max_tokens=512,
        server_log=None,
    )

    assert probe["cancelled"] is True
    assert probe["observed_in_flight_before_close"] is True
    assert probe["observed_in_flight_after_close"] is False
    assert probe["cancel_reclaimed_s"] is not None
    assert probe["observed_in_flight"] is True
    assert any(sample["total"] == 1.0 for sample in probe["active_samples_during_request"])


def test_cancel_probe_rejects_finished_request(monkeypatch):
    def run_one(*args, **kwargs):
        time.sleep(0.2)
        return ChatResult(
            request_id="probe",
            mode="ar",
            cancelled=True,
            cancel_requested_s=0.1,
        )

    monkeypatch.setattr(
        measure,
        "_active",
        lambda client: {"running": 0.0, "waiting": 0.0, "total": 0.0},
    )
    monkeypatch.setattr(measure, "_run_one", run_one)

    probe = measure._run_cancel_probe(
        object(),
        _fixture(),
        model="m",
        mode="ar",
        request_id="probe",
        concurrency=1,
        session=1,
        timeout_s=1.0,
        max_tokens=512,
        server_log=None,
    )

    assert probe["observed_in_flight_before_close"] is False
    assert probe["observed_in_flight_after_close"] is False
    assert probe["observed_in_flight"] is False


def test_ar_window_record_explains_null_draft_identity():
    class Sampler:
        def peak(self):
            return {"memory_mib": 1, "utilization_pct": 1}

    target = {
        "model": "/models/qwen",
        "revision": "abc",
        "tokenizer": {"sha256": "t"},
        "processor": {"sha256": "p"},
        "template": {"sha256": "c"},
        "precision": "bfloat16",
        "missing_reasons": [],
    }
    backend = backend_identity(
        vllm_version="0.22.1",
        code_commit="0" * 40,
        devices=[{"index": 0, "uuid": "GPU-x"}],
    )
    protocol = protocol_payload(
        model_path="/models/qwen",
        mode="ar",
        spec_config=None,
        sampling={"temperature": 0.0},
        pixel_bounds={"min_pixels": 1, "max_pixels": 2},
        max_model_len=8192,
        prefix_caching=False,
        backend_version="0.22.1",
        timing_boundary="test",
    )
    result = ChatResult(
        request_id="r1",
        mode="ar",
        token_ids=[1],
        first_content_s=0.1,
        last_content_s=0.1,
        e2e_s=0.2,
        finish_reason="stop",
    )

    record = measure._window_record(
        args=SimpleNamespace(run_id="run", mode="ar"),
        wall_s=0.2,
        target=target,
        draft=None,
        backend=backend,
        protocol=protocol,
        devices=[{"index": 0, "uuid": "GPU-x"}],
        sampler=Sampler(),
        concurrency=1,
        session_index=1,
        diff={"spec": {}, "active_after": {}, "gauges_after": {}},
        results=[result],
    )

    assert record["draft_identity"] is None
    assert "draft model" in record["missing_reasons"]["draft_identity"]


def test_measure_parser_exposes_ngram_flags():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_serving_gate.py"
    spec = importlib.util.spec_from_file_location("run_serving_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "measure",
            "--mode",
            "ngram",
            "--base-url",
            "http://127.0.0.1:8101",
            "--model",
            "m",
            "--model-path",
            "/models/m",
            "--manifest",
            "manifest.json",
            "--out",
            "out.json",
        ]
    )

    assert args.num_speculative_tokens == 4
    assert args.prompt_lookup_min == 2
    assert args.prompt_lookup_max == 2


def test_window_errors_do_not_enter_throughput_numerator():
    class Sampler:
        def peak(self):
            return {"memory_mib": 1, "utilization_pct": 1}

    target = {
        "model": "/models/qwen",
        "revision": "abc",
        "weight_content_sha256": "w",
        "identity_complete": True,
        "tokenizer": {"sha256": "t"},
        "processor": {"sha256": "p"},
        "template": {"sha256": "c", "effective_source": "chat_template.json"},
        "precision": "bfloat16",
        "config_sha256": "cfg",
        "architecture": "qwen",
        "missing_reasons": [],
    }
    backend = backend_identity(
        vllm_version="0.22.1",
        code_commit="0" * 40,
        devices=[{"index": 0, "uuid": "GPU-x"}],
        applied_evidence=True,
    )
    protocol = protocol_payload(
        model_path="/models/qwen",
        mode="ar",
        spec_config=None,
        sampling={"temperature": 0.0},
        pixel_bounds={"min_pixels": 1, "max_pixels": 2},
        max_model_len=8192,
        prefix_caching=False,
        backend_version="0.22.1",
        timing_boundary="test",
    )
    good = ChatResult(
        request_id="good",
        mode="ar",
        token_ids=[1, 2],
        first_content_s=0.1,
        last_content_s=0.2,
        e2e_s=0.3,
        finish_reason="stop",
        terminal_status="completed",
    )
    bad = ChatResult(
        request_id="bad",
        mode="ar",
        token_ids=[3, 4, 5],
        first_content_s=0.1,
        last_content_s=0.2,
        e2e_s=0.3,
        finish_reason=None,
        terminal_status="truncated",
        error="protocol truncated",
    )

    record = measure._window_record(
        args=SimpleNamespace(run_id="run", mode="ar"),
        wall_s=1.0,
        target=target,
        draft=None,
        backend=backend,
        protocol=protocol,
        devices=[{"index": 0, "uuid": "GPU-x"}],
        sampler=Sampler(),
        concurrency=2,
        session_index=1,
        diff={"spec": {}, "active_after": {}, "gauges_after": {}},
        results=[good, bad],
    )

    assert record["num_errors"] == 1
    assert record["num_success"] == 1
    assert record["output_tokens"] == 2
    assert record["output_tokens_including_failed"] == 5
    assert record["throughput_tokens_s"] == 2.0
    assert record["resource_usage"]["gpu_hours"] is None
    assert record["resource_usage"]["additive"] is False


def test_sse_tpot_is_null_and_proxies_are_named_separately():
    result = ChatResult(
        request_id="r",
        mode="ar",
        stream_state="stream",
        token_ids=[1, 2, 3, 4],
        token_ids_complete=True,
        chunk_token_counts=[3, 1],
        chunk_gaps_s=[0.02],
        num_content_chunks=2,
        first_content_s=0.1,
        last_content_s=0.12,
        e2e_s=0.2,
        finish_reason="stop",
        terminal_status="completed",
    )

    timings = result_timing_fields(result)

    assert timings["tpot_s"] is None
    assert timings["tpot_unavailable_reason"]
    assert timings["client_visible_token_tpot_proxy_s"] is not None
    assert timings["client_chunk_gap_mean_s"] == 0.02
    assert timings["chunk_token_counts"] == [3, 1]


def test_request_control_uses_actual_cli_budget():
    fixture = Fixture(request_id="probe", category="diagnostic", prompt="repeat")
    control = measure._request_control(fixture, model="m", max_tokens=32)
    body = measure._make_fixture_request(fixture, "m", max_tokens=control["max_tokens"])

    assert control["max_tokens"] == 32
    assert body["max_tokens"] == 32


def test_backend_evidence_comes_from_server_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "Initializing a V1 LLM engine (v0.22.1)",
                "Using FLASH_ATTN attention backend out of potential backends",
                "cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>",
                "speculative_config=SpeculativeConfig(method='ngram', model=None, num_spec_tokens=4)",
                "Asynchronous scheduling is disabled.",
            ]
        )
    )

    evidence = measure._server_log_config_evidence(
        str(log),
        vllm_version="0.22.1",
        attention_backend="FLASH_ATTN",
        cudagraph_mode="FULL_AND_PIECEWISE",
        mode="ngram",
        spec_config={"method": "ngram", "num_speculative_tokens": 4},
        async_scheduling=False,
    )

    assert evidence["applied_evidence"] is True
    assert evidence["labels_source"] == "server_log_config_echo"
    assert all(evidence["checks"].values())


def test_missing_server_log_keeps_backend_labels_unverified():
    evidence = measure._server_log_config_evidence(
        None,
        vllm_version="0.22.1",
        attention_backend="FLASH_ATTN",
        cudagraph_mode="FULL_AND_PIECEWISE",
        mode="ar",
        spec_config=None,
        async_scheduling=None,
    )

    assert evidence["applied_evidence"] is False
    assert evidence["labels_source"] == "cli_declared_unverified"
