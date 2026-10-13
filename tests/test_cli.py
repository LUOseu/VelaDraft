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


def test_presets_load_from_the_package_anywhere(tmp_path, monkeypatch):
    """Packaged presets must resolve from CWD-independent package resources.

    Regression: the presets used to be loaded from a repo-relative directory,
    so an installed wheel could not find them (`train --preset smoke` raised
    FileNotFoundError from site-packages).
    """
    monkeypatch.chdir(tmp_path)
    for name in ("smoke", "single-gpu-24g", "single-gpu-48g"):
        cfg = load_preset(name)
        assert cfg["drafter"]["block_size"] == 8
    # a filesystem path still wins for custom presets
    custom = tmp_path / "custom.yaml"
    custom.write_text("target: {model: /models/x}\ndrafter: {block_size: 4}\n", encoding="utf-8")
    assert load_preset(str(custom))["drafter"]["block_size"] == 4
    with pytest.raises(FileNotFoundError, match="known presets"):
        load_preset("no-such-preset")


def test_tested_constraints_are_packaged():
    from vlm_dflash.cli import _parse_constraints

    pins = _parse_constraints()
    assert pins, "constraints-tested.txt must ship inside the package"
    assert "torch" in pins and "transformers" in pins


def test_version_flag_and_single_version_source(capsys):
    import importlib.metadata as md

    from vlm_dflash import __version__

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"vlm-dflash {__version__}"
    try:
        dist_version = md.version("vlm-dflash")
    except md.PackageNotFoundError:
        pytest.skip("vlm-dflash is not installed as a distribution here")
    assert dist_version == __version__


def _train_args(tmp_path, *extra):
    parser = build_parser()
    return parser.parse_args(
        [
            "train",
            "--target", "/models/Qwen2.5-VL-3B-Instruct",
            "--dataset", str(tmp_path / "data.jsonl"),
            *extra,
        ]
    )


def _fake_contract() -> dict:
    return {
        "target": {"model_path": "/models/x", "model_fingerprint": "mf", "processor_fingerprint": "pf"},
        "mask": {"token_id": 151669},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"draft_layer_ids": [1, 12, 23]},
        "image": {"min_pixels": 200704, "max_pixels": 1003520},
    }


def test_cmd_train_refuses_invalid_features_dir(monkeypatch, tmp_path):
    import vlm_dflash.cli as cli
    import vlm_dflash.contract as contract_module

    cfg = {
        "target": {"model": "/models/x"},
        "dataset": {"source": "jsonl", "path": "d.jsonl"},
        "run": {"name": "x", "dir": str(tmp_path / "run")},
    }
    monkeypatch.setattr(cli, "build_train_config", lambda args: cfg)
    monkeypatch.setattr(contract_module, "ensure_contract", lambda cfg, run_dir: _fake_contract())

    args = _train_args(tmp_path, "--features", str(tmp_path / "missing_features"))
    with pytest.raises(SystemExit, match="not a captured feature store"):
        cli.cmd_train(args)


def test_cmd_train_reuses_external_features_and_skips_regen_capture(monkeypatch, tmp_path):
    import vlm_dflash.cli as cli
    import vlm_dflash.contract as contract_module
    import vlm_dflash.data.prepare as prepare_module
    import vlm_dflash.dflash.train as train_module
    import vlm_dflash.target.capture as capture_module
    import vlm_dflash.target.regen as regen_module

    features = tmp_path / "shared_features"
    features.mkdir()
    (features / "index.jsonl").write_text("", encoding="utf-8")
    (features / "manifest.json").write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "run"
    cfg = {
        "target": {"model": "/models/x"},
        "dataset": {"source": "jsonl", "path": "d.jsonl"},
        "run": {"name": "x", "dir": str(run_dir)},
    }
    monkeypatch.setattr(cli, "build_train_config", lambda args: cfg)
    monkeypatch.setattr(contract_module, "ensure_contract", lambda cfg, run_dir: _fake_contract())

    calls: dict = {}
    monkeypatch.setattr(prepare_module, "build_dataset", lambda cfg, out, limit=None: calls.setdefault("data", True))
    monkeypatch.setattr(
        regen_module, "run_regen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("regen must not run"))
    )
    monkeypatch.setattr(
        capture_module, "run_capture", lambda *a, **k: (_ for _ in ()).throw(AssertionError("capture must not run"))
    )

    def fake_train(cfg, run_dir, contract, steps=None, limit=None, resume=False, features_dir=None, feature_mode=None):
        calls["features_dir"] = str(features_dir)
        return {"steps": 1, "final_loss": 1.0}

    monkeypatch.setattr(train_module, "train_dflash", fake_train)

    args = _train_args(tmp_path, "--features", str(features))
    assert cli.cmd_train(args) == 0
    assert calls["data"] is True
    assert calls["features_dir"] == str(features.resolve())


def test_preset_pixel_bounds_can_be_overridden_with_null():
    # Qwen3-VL configs set image_*_pixels: null so a CLI run (which merges a
    # preset first) resolves the bounds from the processor instead of leaking
    # the preset's Qwen2.5-VL bounds into contract validation.
    preset = load_preset("single-gpu-48g")
    assert preset["target"]["image_min_pixels"] == 200704
    merged = deep_merge(preset, {"target": {"image_min_pixels": None, "image_max_pixels": None}})
    assert merged["target"]["image_min_pixels"] is None
    assert merged["target"]["image_max_pixels"] is None


def test_deep_merge_overrides_nested():
    base = {"train": {"lr": 1e-3, "betas": [0.9, 0.95]}, "drafter": {"block_size": 8}}
    override = {"train": {"lr": 2e-3}}
    merged = deep_merge(base, override)
    assert merged["train"]["lr"] == 2e-3
    assert merged["train"]["betas"] == [0.9, 0.95]
    assert base["train"]["lr"] == 1e-3


def test_cli_set_overrides_reach_algorithm_validation(tmp_path):
    """Dotted --set values must be merged before validate_config runs.

    Regression: validation used to read the pre-override config, so a required
    knob supplied via --set (JetSpec's ``drafter.loss_decay_gamma=null`` +
    ``capture.store_final_hidden=true``) failed even though the run config was
    satisfied.
    """
    parser = build_parser()
    base = [
        "train",
        "--target", "/models/Qwen2.5-VL-3B-Instruct",
        "--dataset", str(tmp_path / "data.jsonl"),
        "--preset", "single-gpu-24g",
        "--algorithm", "jetspec",
    ]
    cfg = build_train_config(
        parser.parse_args(
            base
            + ["--set", "capture.store_final_hidden=true", "--set", "drafter.loss_decay_gamma=null"]
        )
    )
    assert cfg["capture"]["store_final_hidden"] is True
    assert cfg["drafter"]["loss_decay_gamma"] is None

    # Without the overrides the merged config must fail validation (the preset
    # ships DFlash's decayed weighting, which JetSpec rejects).
    with pytest.raises(ValueError, match="loss_decay_gamma"):
        build_train_config(parser.parse_args(base))


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
