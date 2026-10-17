from __future__ import annotations

import importlib.util
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from vlm_dflash.serving import measure
from vlm_dflash.serving.fixtures import Fixture
from vlm_dflash.serving.openai_api import ChatResult
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
