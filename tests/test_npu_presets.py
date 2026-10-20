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


def test_qwen_smoke_context_leaves_room_for_supervised_response():
    for name in METHOD_PRESETS:
        cfg = load_preset(name)
        # Qwen2.5-VL uses 14px patches merged 2x2. Even a single image at
        # the configured maximum must leave space for prompt and response.
        visual_tokens = cfg["target"]["image_max_pixels"] // (28 * 28)
        assert cfg["capture"]["seq_length"] >= visual_tokens + cfg["regen"]["max_new_tokens"] + 128, name


def test_explicit_images_directory_overrides_packaged_default():
    from vlm_dflash.cli import build_parser, build_train_config
    args = build_parser().parse_args([
        "train", "--preset", "npu-dflash-smoke", "--dataset", "/data/rows.jsonl",
        "--images-dir", "/data/actual-images", "--set", "train.device=cpu",
        "--set", "target.device=cpu",
    ])
    cfg = build_train_config(args)
    assert cfg["dataset"]["jsonl_images_dir"] == "/data/actual-images"


@pytest.mark.parametrize("target_layers", [28, 36])
def test_presets_capture_the_layers_consumed_by_the_target(target_layers):
    from vlm_dflash.contract import build_target_layer_ids, contract_defaults_from_config
    for name in METHOD_PRESETS:
        defaults = contract_defaults_from_config(load_preset(name))
        routing = build_target_layer_ids(target_layers, defaults["draft_num_hidden_layers"])
        capture = defaults["capture_layer_ids"] or routing
        consumed = defaults["algorithm_context_layer_ids"] or routing
        assert set(consumed) <= set(capture), name
