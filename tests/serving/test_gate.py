from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _gate_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_serving_gate.py"
    spec = importlib.util.spec_from_file_location("run_serving_gate_lifecycle_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _document(observed_in_flight: bool) -> dict:
    return {
        "meta": {"concurrency": "1", "sessions": 1},
        "windows": [
            {
                "measurement_scope": {"concurrency": 1, "session_index": 1},
                "num_requests": 1,
                "num_errors": 0,
            }
        ],
        "records": [{"token_ids": [1], "error": None, "finish_reason": "stop", "terminal_status": "completed"}],
        "summary": {
            "boundary_results": [{"ok": True}] * 8,
            "cancel_probes": [
                {
                    "concurrency": 1,
                    "session": 1,
                    "cancelled": True,
                    "observed_in_flight": observed_in_flight,
                    "cancel_reclaimed_s": 0.1,
                    "error": None,
                    "concurrent_with_healthy_request": False,
                    "internal_abort_evidence": False,
                }
            ],
            "nonstream_probes": [{"ok": True}],
            "pair_checks": [
                {
                    "concurrency": 1,
                    "session": 1,
                    "input_submission_ok": True,
                    "output_attribution_verified": False,
                }
            ],
        },
    }


def test_lifecycle_gate_rejects_unobserved_cancel():
    checks = _gate_module()._measure_lifecycle_checks(_document(False))

    assert checks["socket_close_reclaimed"] is False
    assert checks["full_lifecycle_ok"] is False


def test_lifecycle_gate_accepts_observed_cancel():
    checks = _gate_module()._measure_lifecycle_checks(_document(True))

    assert checks["socket_close_reclaimed"] is True
    assert checks["cancel_claim_scope"] == "client_socket_close_and_global_gauge_only"
    assert checks["full_lifecycle_ok"] is False


def test_lifecycle_gate_requires_concurrent_abort_and_output_attribution():
    document = _document(True)
    document["summary"]["cancel_probes"][0]["concurrent_with_healthy_request"] = True
    document["summary"]["cancel_probes"][0]["internal_abort_evidence"] = True
    document["summary"]["pair_checks"][0]["output_attribution_verified"] = True

    checks = _gate_module()._measure_lifecycle_checks(document)

    assert checks["api_ok"] is True
    assert checks["full_lifecycle_ok"] is True


def test_lifecycle_gate_rejects_window_errors_and_unexpected_cancellation():
    document = _document(True)
    document["windows"][0]["num_errors"] = 1
    document["records"][0]["cancelled"] = True

    checks = _gate_module()._measure_lifecycle_checks(document)

    assert checks["api_ok"] is False
    assert checks["window_errors_total"] == 1
    assert checks["unexpected_cancelled_records"]
    assert checks["full_lifecycle_ok"] is False


def test_lifecycle_gate_rejects_record_count_mismatch():
    document = _document(True)
    document["windows"][0]["num_requests"] = 2

    checks = _gate_module()._measure_lifecycle_checks(document)

    assert checks["record_count_ok"] is False
    assert checks["api_ok"] is False


def test_lifecycle_gate_requires_pair_input_evidence():
    document = _document(True)
    document["summary"]["cancel_probes"][0]["concurrent_with_healthy_request"] = True
    document["summary"]["cancel_probes"][0]["internal_abort_evidence"] = True
    document["summary"]["pair_checks"][0]["input_submission_ok"] = False
    document["summary"]["pair_checks"][0]["output_attribution_verified"] = True

    checks = _gate_module()._measure_lifecycle_checks(document)

    assert checks["pair_input_ok"] is False
    assert checks["full_lifecycle_ok"] is False
