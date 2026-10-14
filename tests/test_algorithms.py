"""Algorithm registry: default registration, metadata, fail-loud gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import (
    DrafterAlgorithm,
    UnknownAlgorithmError,
    available_algorithms,
    get_algorithm,
    register,
)
from vlm_dflash.cli import build_parser, build_train_config
from vlm_dflash.eval.run import check_checkpoint_algorithm


def test_default_registry_exposes_dflash():
    assert "dflash" in available_algorithms()
    algorithm = get_algorithm("dflash")
    assert algorithm.name == "dflash"
    metadata = algorithm.metadata
    assert metadata.maturity["training"] in {"stable", "validated", "experimental", "architecture-only", "unsupported"}
    assert metadata.capabilities["image"] is True
    assert metadata.capabilities["multi_image"] is True
    assert metadata.capabilities["video"] is False
    assert {"qwen25_vl", "qwen3_vl"} <= set(metadata.targets)


def test_none_resolves_default_algorithm():
    assert get_algorithm(None).name == "dflash"
    assert get_algorithm("").name == "dflash"


def test_unknown_algorithm_error_lists_available():
    with pytest.raises(UnknownAlgorithmError) as excinfo:
        get_algorithm("nosuchalgorithm")
    message = str(excinfo.value)
    assert "nosuchalgorithm" in message and "dflash" in message and "dspark" in message


def test_dflash_validate_config_gates():
    algorithm = get_algorithm("dflash")
    algorithm.validate_config({"drafter": {}})  # defaults are valid
    with pytest.raises(ValueError, match="block_size"):
        algorithm.validate_config({"drafter": {"block_size": 1}})
    with pytest.raises(ValueError, match="num_anchors"):
        algorithm.validate_config({"drafter": {"num_anchors": 0}})
    with pytest.raises(ValueError, match="loss_decay_gamma"):
        algorithm.validate_config({"drafter": {"loss_decay_gamma": 0}})
    with pytest.raises(ValueError, match="drafter"):
        algorithm.validate_config({})


def test_register_custom_algorithm_round_trip():
    class DummyAlgorithm(DrafterAlgorithm):
        name = "dummy-test"
        metadata = get_algorithm("dflash").metadata

        def validate_config(self, cfg):
            return None

        def build_draft_config(self, contract, cfg):
            return None

        def build_trainer_module(self, draft_config, contract, cfg, embed_weight, lm_head_weight=None):
            return None

        def load_decoder(self, checkpoint_path, map_location="cpu"):
            return None, {}

        def checkpoint_metadata(self, contract, cfg, draft_config):
            return {}

    register(DummyAlgorithm())
    try:
        assert "dummy-test" in available_algorithms()
        assert get_algorithm("dummy-test").name == "dummy-test"
    finally:
        # keep the global registry clean for other tests
        from vlm_dflash.algorithms import registry as registry_module

        registry_module._REGISTRY.pop("dummy-test", None)


def test_cli_train_algorithm_flag_and_run_name(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--target", "/models/Qwen2.5-VL-3B-Instruct",
            "--dataset", str(tmp_path / "data.jsonl"),
            "--preset", "single-gpu-24g",
            "--algorithm", "dflash",
        ]
    )
    cfg = build_train_config(args)
    assert cfg["algorithm"] == "dflash"
    assert cfg["run"]["dir"].endswith("-dflash")


def test_cli_train_unknown_algorithm_fails_loud(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--target", "/models/Qwen2.5-VL-3B-Instruct",
            "--dataset", str(tmp_path / "data.jsonl"),
            "--preset", "single-gpu-24g",
            "--algorithm", "nosuchalgorithm",
        ]
    )
    with pytest.raises(UnknownAlgorithmError):
        build_train_config(args)


def test_checkpoint_algorithm_mismatch_is_rejected():
    algorithm = get_algorithm("dflash")
    check_checkpoint_algorithm("ckpt_final.pt", {}, algorithm)  # legacy checkpoints default to dflash
    check_checkpoint_algorithm("ckpt_final.pt", {"algorithm": "dflash"}, algorithm)
    with pytest.raises(RuntimeError, match="algorithm"):
        check_checkpoint_algorithm("ckpt_final.pt", {"algorithm": "dspark"}, algorithm)


TARGET_STATES = {
    "stable",
    "validated",
    "reduced_validation",
    "experimental",
    "architecture-only",
    "unsupported",
}


@pytest.mark.parametrize("name", available_algorithms())
def test_metadata_target_states_use_the_documented_vocabulary(name):
    metadata = get_algorithm(name).metadata
    for family, state in metadata.targets.items():
        assert state in TARGET_STATES, f"{name}.targets[{family}] = {state!r}"
    for axis, state in metadata.maturity.items():
        assert state in {"stable", "validated", "experimental", "architecture-only", "unsupported"}, (
            f"{name}.maturity[{axis}] = {state!r}"
        )


def test_support_matrix_renders_from_algorithm_metadata():
    """SUPPORT_MATRIX.md must not drift from the machine-readable metadata."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "render_support_matrix", root / "scripts" / "render_support_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render()
    assert (root / "SUPPORT_MATRIX.md").read_text(encoding="utf-8") == rendered


@pytest.mark.parametrize("name", available_algorithms())
def test_verify_backend_switch_reaches_shared_inference_engine(name):
    """The evaluator's verify_sdpa_backend setting must bind every algorithm.

    Regression for the DSpark token-exact gate failure (2026-10-12): the base
    class left ``configure_verify_backend`` as a no-op, so DSpark verified under
    the reference SDPA backend narrowing while ``generate()`` used the default
    backends; the bf16 kernel mismatch flipped a near-tie on 1/8 COCO smoke
    prompts.
    """
    from vlm_dflash.dflash.draft_model import verify_sdpa_backend_switch_enabled

    algorithm = get_algorithm(name)
    try:
        algorithm.configure_verify_backend(False)
        assert verify_sdpa_backend_switch_enabled() is False
        algorithm.configure_verify_backend(True)
        assert verify_sdpa_backend_switch_enabled() is True
    finally:
        algorithm.configure_verify_backend(True)
