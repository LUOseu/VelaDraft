"""The NPU support matrix must cover every method and never overclaim hardware."""

from __future__ import annotations

import json
from pathlib import Path

from vlm_dflash.algorithms import available_algorithms

MATRIX = Path(__file__).resolve().parents[1] / "docs" / "npu" / "support_matrix.json"
NPU_STATES = {"NPU_UNVERIFIED", "NOT_RUN", "NPU_SMOKE_PASSED", "NPU_TRAINING_VALIDATED"}
STAGES = {"forward", "backward", "update", "resume", "export"}


def _payload() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_matrix_covers_every_registered_method_and_stage():
    payload = _payload()
    assert {entry["name"] for entry in payload["methods"]} == set(available_algorithms())
    for entry in payload["methods"]:
        assert entry["implementation"] == "COMPLETE"
        assert entry["npu_evidence"] in NPU_STATES
        _check_evidence(entry, "npu_evidence")
        assert set(entry["offline"]) == STAGES
        assert set(entry["online"]) == STAGES
        assert all(value == "implemented" for value in entry["offline"].values())
        assert all(value == "implemented" for value in entry["online"].values())


def test_matrix_covers_all_adapters_and_required_data_paths():
    payload = _payload()
    assert {entry["name"] for entry in payload["adapters"]} == {
        "qwen25_vl",
        "qwen3_vl",
        "qwen3_5_moe",
        "smolvlm",
    }
    stages = {entry["name"]: entry for entry in payload["stages"]}
    assert {"prepare", "regen", "capture", "offline_training", "online_training", "resume", "export"} <= set(stages)
    assert stages["target_device_map"]["implementation"] in {"PARTIAL", "COMPLETE"}
    for entry in payload["adapters"] + payload["stages"]:
        _check_evidence(entry, "npu_evidence")
    assert stages["ddp"]["implementation"] == "BLOCKED"
    assert stages["ddp"]["npu_evidence"] == "NOT_RUN"


def test_matrix_phase_states_are_fail_closed():
    payload = _payload()
    phases = {entry["name"]: entry["status"] for entry in payload["phases"]}
    assert set(phases) == {"H0", "H1", "H2", "H3", "H4"}
    for entry in payload["phases"]:
        _check_evidence(entry, "status")
    assert set(phases.values()) <= NPU_STATES
    assert payload["overall"]["npu_evidence"] in NPU_STATES
    if payload["overall"]["npu_evidence"] == "NPU_SMOKE_PASSED":
        assert all(phases[name] == "NPU_SMOKE_PASSED" for name in ("H0", "H1", "H2"))


def _check_evidence(entry, state_key):
    if entry[state_key] not in {"NPU_SMOKE_PASSED", "NPU_TRAINING_VALIDATED"}:
        return
    assert entry.get("evidence"), f"{entry['name']}: hardware claim needs evidence"
    for reference in entry["evidence"]:
        path = MATRIX.parents[2] / reference
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "NPU_SMOKE_PASSED", reference
        if payload.get("format") == "vlm-dflash-npu-seven-method-smoke-v1" and "offline" in entry:
            record = next(item for item in payload["records"] if item["method"] == entry["name"])
            assert record["device_type"] == "npu"
            assert all(record[key] for key in ("forward", "backward", "update", "resume", "export"))
