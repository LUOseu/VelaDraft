"""Control-flow tests only: these mocks do not establish hardware evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import torch


def _runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/npu/run_npu_smoke.py"
    spec = importlib.util.spec_from_file_location("npu_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.runtime, "ensure_runtime_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(runner.runtime, "environment_report", lambda: {})
    monkeypatch.setattr(runner, "_npu_smi", lambda: {})
    return runner


def test_explicit_step_flags_override_defaults_and_are_recorded(monkeypatch, tmp_path):
    runner = _runner(monkeypatch)
    seen = {}

    def phase(name, spec, args, directory):
        command = runner._phase_command(name, spec, args, directory / "result.json", directory)
        seen[name] = int(command[command.index("--steps") + 1])
        return {"phase": name, "status": "NPU_SMOKE_PASSED", "npu_verified": True}

    monkeypatch.setattr(runner, "_run_phase", phase)
    monkeypatch.setattr(sys, "argv", [
        "run_npu_smoke", "--phases", "H2,H3", "--steps", "12", "--h3-steps", "210",
        "--work-dir", str(tmp_path),
    ])
    assert runner.main() == 0
    assert seen == {"H2": 12, "H3": 210}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["phases"]["H2"]["steps"] == 12
    assert manifest["phases"]["H3"]["steps"] == 210


def test_requested_unexecuted_phase_prevents_success(monkeypatch, tmp_path):
    runner = _runner(monkeypatch)
    monkeypatch.setattr(runner, "_run_phase", lambda name, *a: {
        "phase": name, "status": "NPU_SMOKE_PASSED", "npu_verified": True,
    })
    monkeypatch.setattr(sys, "argv", [
        "run_npu_smoke", "--phases", "H0,H4", "--work-dir", str(tmp_path),
    ])
    assert runner.main() != 0
    payload = json.loads((tmp_path / "run_npu_smoke.json").read_text())
    assert payload["status"] != "NPU_SMOKE_PASSED"
    assert any("H4" in problem for problem in payload["problems"])
