"""Synthetic smoke artifacts must retain actual measured training results."""

import importlib.util
import json
from pathlib import Path

import torch


def test_smoke_export_records_measured_steps_and_loss(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/npu/seven_method_smoke.py"
    spec = importlib.util.spec_from_file_location("seven_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = module._make_target(tmp_path / "target")
    result = module._run_method(
        "dflash", module._contract(target), torch.device("cpu"),
        steps=10, seed=2026, artifact_dir=tmp_path / "artifacts",
    )
    run = tmp_path / "artifacts/dflash/train"
    summary = json.loads((run / "train_summary.json").read_text())
    assert summary["steps"] == 10
    assert summary["final_loss"] == result["final_loss"]
    checkpoint = torch.load(run / "ckpt_final.pt", weights_only=False)
    assert checkpoint["step"] == 10
    history = [json.loads(row) for row in (run / "train_log.jsonl").read_text().splitlines()]
    assert len(history) == 10
    assert history[-1]["loss"] == result["final_loss"]
    assert result["memory_samples"][0]["step"] == 0
    assert result["memory_samples"][-1]["step"] == 10
    assert all(sample["host_peak_rss_bytes"] > 0 for sample in result["memory_samples"])
