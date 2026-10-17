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
                "num_errors": 0,
            }
        ],
        "records": [{"token_ids": [1], "error": None}],
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
                }
            ],
            "nonstream_probes": [{"ok": True}],
            "pair_checks": [{"ok": True}],
        },
    }


def test_lifecycle_gate_rejects_unobserved_cancel():
    checks = _gate_module()._measure_lifecycle_checks(_document(False))

    assert checks["cancel_all_reclaimed"] is False
    assert checks["ok"] is False


def test_lifecycle_gate_accepts_observed_cancel():
    checks = _gate_module()._measure_lifecycle_checks(_document(True))

    assert checks["cancel_all_reclaimed"] is True
    assert checks["ok"] is True
