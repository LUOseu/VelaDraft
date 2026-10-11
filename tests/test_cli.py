"""CLI plumbing: presets, config building, doctor report structure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.cli import (
    build_parser,
    build_train_config,
    collect_doctor_report,
    deep_merge,
    load_preset,
)


def test_all_presets_load():
    for name in ("smoke", "single-gpu-24g", "single-gpu-48g"):
        cfg = load_preset(name)
        assert "train" in cfg and "drafter" in cfg
        assert cfg["drafter"]["block_size"] == 8
        # family-specific values must stay runtime-resolved
        assert "spatial_rope_sections" not in cfg["drafter"]


def test_deep_merge_overrides_nested():
    base = {"train": {"lr": 1e-3, "betas": [0.9, 0.95]}, "drafter": {"block_size": 8}}
    override = {"train": {"lr": 2e-3}}
    merged = deep_merge(base, override)
    assert merged["train"]["lr"] == 2e-3
    assert merged["train"]["betas"] == [0.9, 0.95]
    assert base["train"]["lr"] == 1e-3


def test_build_train_config_from_jsonl(tmp_path):
    parser = build_parser()
    dataset_path = tmp_path / "data.jsonl"
    args = parser.parse_args(
        [
            "train",
            "--target", "/models/Qwen2.5-VL-3B-Instruct",
            "--dataset", str(dataset_path),
            "--preset", "single-gpu-24g",
            "--limit", "32",
            "--steps", "10",
        ]
    )
    cfg = build_train_config(args)
    assert cfg["dataset"]["source"] == "jsonl"
    assert cfg["dataset"]["path"] == str(dataset_path)
    assert cfg["dataset"]["slice"] == 32
    assert cfg["target"]["model"].endswith("Qwen2.5-VL-3B-Instruct")
    assert cfg["run"]["dir"].startswith("runs/")
    assert cfg["train"]["micro_batch_size"] == 2


def test_build_train_config_requires_a_dataset():
    parser = build_parser()
    args = parser.parse_args(["train", "--target", "/models/x"])
    with pytest.raises(SystemExit):
        build_train_config(args)


def test_doctor_report_structure_without_target():
    report = collect_doctor_report(None, 151669, 1000, 2048)
    assert isinstance(report["ok"], bool)
    assert any(entry["name"].startswith("python") for entry in report["checks"])
    assert "cuda" in report


def test_doctor_json_cli_runs():
    from vlm_dflash.cli import cmd_doctor

    parser = build_parser()
    args = parser.parse_args(["doctor", "--json"])
    code = cmd_doctor(args)
    assert code in (0, 2)

def test_cmd_eval_passes_run_dir_positionally(monkeypatch, tmp_path):
    """Regression: cmd_eval used to swap evaluate()'s run_dir/contract argument order."""
    import vlm_dflash.cli as cli
    import vlm_dflash.contract as contract_module
    import vlm_dflash.eval.run as eval_module

    captured = {}

    def fake_evaluate(cfg, run_dir, contract, num_prompts=None, max_new_tokens=None, tag=None):
        captured["run_dir"] = run_dir
        captured["contract"] = contract
        captured["num_prompts"] = num_prompts
        captured["verification"] = cfg["eval"]["verification"]
        captured["strict_margin"] = cfg["eval"]["strict_parallel_margin"]
        captured["tag"] = tag
        return {"runs": {}}

    monkeypatch.setattr(eval_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(contract_module, "ensure_contract", lambda cfg, run_dir: {"frozen": True})
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            str(tmp_path),
            "--num-prompts",
            "4",
            "--max-new-tokens",
            "8",
            "--verification",
            "strict_parallel",
            "--strict-margin",
            "1.5",
            "--tag",
            "bench",
        ]
    )
    code = cli.cmd_eval(args)
    assert code == 0
    assert captured["run_dir"] == tmp_path.resolve()
    assert captured["contract"] == {"frozen": True}
    assert captured["num_prompts"] == 4
    assert captured["verification"] == "strict_parallel"
    assert captured["strict_margin"] == 1.5
    assert captured["tag"] == "bench"
