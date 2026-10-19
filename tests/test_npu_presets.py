"""Packaged NPU preset schema and method coverage tests."""

from __future__ import annotations

from importlib import resources

import pytest

from vlm_dflash.algorithms import algorithm_name, available_algorithms, get_algorithm
from vlm_dflash.cli import _PRESET_NAMES, load_preset

METHOD_PRESETS = (
    "npu-dflash-smoke",
    "npu-dspark-smoke",
    "npu-domino-smoke",
    "npu-jetspec-smoke",
    "npu-dflash2-smoke",
    "npu-dflare-smoke",
    "npu-dfly-smoke",
)


def test_every_npu_preset_is_packaged_and_loadable():
    npu_names = [name for name in _PRESET_NAMES if name.startswith("npu-")]
    assert set(METHOD_PRESETS) <= set(npu_names)
    assert {"npu-online-smoke", "npu-split-device", "npu-target-device-map"} <= set(npu_names)
    for name in npu_names:
        cfg = load_preset(name)
        assert "train" in cfg, name
        assert "training" not in cfg, name
        assert cfg["train"]["device"].startswith("npu:"), name
        assert cfg["drafter"]["attention_backend"] == "sdpa_chunked", name
        assert int(cfg["drafter"]["attention_chunk_size"]) > 0, name


def test_seven_method_presets_cover_the_registry_and_validate():
    covered = set()
    for name in METHOD_PRESETS:
        cfg = load_preset(name)
        method = algorithm_name(cfg.get("algorithm"))
        get_algorithm(method).validate_config(cfg)
        covered.add(method)
    assert covered == set(available_algorithms())


def test_online_and_split_presets_use_real_npu_routing_fields():
    online = load_preset("npu-online-smoke")
    assert online["train"]["feature_mode"] == "online"
    assert online["train"]["num_workers"] == 0
    assert online["target"]["device"] == "npu:0"
    split = load_preset("npu-split-device")
    assert split["target"]["device"] == "npu:0"
    assert split["train"]["device"] == "npu:1"
    candidate = load_preset("npu-target-device-map")
    assert "device_map" in candidate["target"]
    assert "device" not in candidate["target"]
    assert set(candidate["target"]["device_map"].values()) <= {"npu:0", "npu:1"}
    assert "model.language_model" in candidate["target"]["device_map"]


def test_npu_presets_are_visible_as_package_resources():
    directory = resources.files("vlm_dflash") / "presets"
    names = {path.name for path in directory.iterdir()}
    for name in METHOD_PRESETS:
        assert f"{name}.yaml" in names


def test_npu_presets_do_not_claim_a_chip_or_a_benchmark():
    for name in [item for item in _PRESET_NAMES if item.startswith("npu-")]:
        text = (resources.files("vlm_dflash") / "presets" / f"{name}.yaml").read_text(encoding="utf-8")
        lowered = text.lower()
        assert "910b" not in lowered and "910c" not in lowered, name
        assert "tokens/s" not in lowered and "throughput" not in lowered, name
