"""Export -> load_export round trip on a tiny draft checkpoint (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.export.hf import export_run, load_export


def _tiny_draft():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

    torch.manual_seed(3)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    return config, Qwen3DFlashDraftModel(config)


def test_export_roundtrip(tmp_path):
    config, draft = _tiny_draft()
    run_dir = tmp_path / "run"
    (run_dir / "train").mkdir(parents=True)
    (run_dir / "data").mkdir()
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 100, "max_pixels": 200},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"num_draft_layers": 2},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
    }
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"drafter": {"spatial_rope_mode": "replace", "spatial_rope_enabled": False}}), encoding="utf-8"
    )
    (run_dir / "data" / "manifest.json").write_text(
        json.dumps({"source": "unit", "num_train": 4, "num_val": 2, "rows_sha256": "d" * 64}), encoding="utf-8"
    )
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "draft_config": config.to_dict(),
        "attention_backend": "sdpa",
        "mask_token_id": 255,
        "block_size": 8,
        "num_anchors": 16,
        "loss_decay_gamma": 7.0,
        "target_layer_ids": [0, 1],
        "identity": {"model_fingerprint": "f" * 64},
        "step": 3,
        "model": {key: value.clone() for key, value in draft.state_dict().items()},
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")

    export_dir = export_run(run_dir)
    for name in ("model.safetensors", "config.json", "model_card.md", "checksums.json"):
        assert (export_dir / name).exists(), name

    model, exported = load_export(export_dir)
    original_state = draft.state_dict()
    loaded_state = model.state_dict()
    assert set(original_state) == set(loaded_state)
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key]), key

    assert exported["drafter"]["block_size"] == 8
    assert exported["drafter"]["spatial_rope"]["sections"] == [16, 24, 24]
    # Full-head rotary when the contract records no partial factor (pre-field contracts too).
    assert exported["drafter"]["spatial_rope"]["partial_rotary_factor"] == 1.0
    assert exported["drafter"]["spatial_rope"]["interleaved"] is False
    assert exported["target"]["model_fingerprint"] == "f" * 64
    assert exported["data"]["num_train"] == 4
    # Checkpoints written before the feature-mode tag export as offline.
    assert exported["feature_mode"] == "offline"
    # No recipe recorded on this checkpoint -> the export states that plainly.
    assert exported["train_recipe"] is None
    card = (export_dir / "model_card.md").read_text(encoding="utf-8")
    assert "tiny-vl" in card
    assert "offline (pre-captured feature store)" in card


def test_export_records_online_feature_mode(tmp_path):
    config, draft = _tiny_draft()
    run_dir = tmp_path / "run"
    (run_dir / "train").mkdir(parents=True)
    (run_dir / "data").mkdir()
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 100, "max_pixels": 200},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"num_draft_layers": 2},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
    }
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"drafter": {"spatial_rope_mode": "replace", "spatial_rope_enabled": False}}), encoding="utf-8"
    )
    (run_dir / "train" / "train_summary.json").write_text(
        json.dumps({"steps": 3, "final_loss": 1.0, "feature_mode": "online"}), encoding="utf-8"
    )
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "draft_config": config.to_dict(),
        "attention_backend": "sdpa",
        "mask_token_id": 255,
        "block_size": 8,
        "num_anchors": 16,
        "loss_decay_gamma": 7.0,
        "target_layer_ids": [0, 1],
        "identity": {"model_fingerprint": "f" * 64},
        "feature_mode": "online",
        "feature_backend": "huggingface",
        "step": 3,
        "model": {key: value.clone() for key, value in draft.state_dict().items()},
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")

    export_dir = export_run(run_dir)
    exported = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    assert exported["feature_mode"] == "online"
    assert exported["training"]["feature_mode"] == "online"
    card = (export_dir / "model_card.md").read_text(encoding="utf-8")
    assert "online (live target extraction)" in card


def _tiny_fusion_draft(kind: str):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    dflash_config = {
        "mask_token_id": 255,
        "target_layer_ids": [0, 1],
        "context_layer_ids": [0, 1],
        "spatial_rope_enabled": False,
        "fusion_trainable": True,
        "fusion_init": "distance",
    }
    if kind == "dfly":
        dflash_config.update(
            {
                "separate_context_kv": False,
                "sample_from_anchor": False,
                "enable_hidden_correction": True,
                "hidden_correction_intermediate_size": None,
                "objective_profile": "controlled_ce",
            }
        )
        from vlm_dflash.dfly.model import Qwen3DFlyDraftModel

        model_class = Qwen3DFlyDraftModel
    else:
        dflash_config.update({"separate_context_kv": True})
        from vlm_dflash.dflare.model import Qwen3DFlareDraftModel

        model_class = Qwen3DFlareDraftModel
    torch.manual_seed(3)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config=dflash_config,
    )
    config._attn_implementation = "sdpa"
    return config, model_class(config), dflash_config


def _write_fusion_run(tmp_path, kind: str):
    config, draft, dflash_config = _tiny_fusion_draft(kind)
    run_dir = tmp_path / f"run_{kind}"
    (run_dir / "train").mkdir(parents=True)
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 100, "max_pixels": 200},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"num_draft_layers": 2, "draft_layer_ids": [0, 1]},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
        "head": {"tie_word_embeddings": True},
    }
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"drafter": {"spatial_rope_mode": "replace", "spatial_rope_enabled": False}}),
        encoding="utf-8",
    )
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": kind,
        "draft_config": config.to_dict(),
        "attention_backend": "sdpa",
        "mask_token_id": 255,
        "block_size": 8,
        "num_anchors": 16,
        "loss_decay_gamma": 7.0,
        "target_layer_ids": [0, 1],
        "capture_layer_ids": [0, 1],
        "identity": {"model_fingerprint": "f" * 64},
        "step": 3,
        "model": {key: value.clone() for key, value in draft.state_dict().items()},
    }
    if kind == "dfly":
        checkpoint["objective_profile"] = "controlled_ce"
        checkpoint["dfly"] = {"enable_hidden_correction": True, "fusion_init": "distance"}
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")
    return run_dir, draft


def test_fusion_export_reload_roundtrip_and_objective_profile(tmp_path):
    """DFlare/DFly checkpoints export and reload through the registry; the DFly
    CE adaptation is recorded machine-readably in the export and model card."""
    import pytest

    for kind in ("dflare", "dfly"):
        run_dir, draft = _write_fusion_run(tmp_path, kind)
        export_dir = export_run(run_dir)
        model, exported = load_export(export_dir)
        assert exported["method"] == kind
        original_state = draft.state_dict()
        loaded_state = model.state_dict()
        assert set(original_state) == set(loaded_state)
        for key in original_state:
            assert torch.equal(original_state[key], loaded_state[key]), (kind, key)
        card = (export_dir / "model_card.md").read_text(encoding="utf-8")
        if kind == "dfly":
            assert exported["objective_profile"] == "controlled_ce"
            assert exported["drafter"]["draft_config"]["dflash_config"]["objective_profile"] == "controlled_ce"
            assert "controlled_ce" in card
            assert "two-stage" in card  # upstream recipe split stays explicit
        else:
            assert exported["objective_profile"] is None
            # DFlare keeps the dedicated context K/V in the serving config.
            assert exported["drafter"]["draft_config"]["dflash_config"]["separate_context_kv"] is True


def _write_dspark_run(tmp_path):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dspark.model import Qwen3DSparkDraftModel

    torch.manual_seed(3)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config={
            "mask_token_id": 255,
            "target_layer_ids": [0, 1],
            "spatial_rope_enabled": False,
            "markov_rank": 32,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "sample_from_anchor": True,
        },
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DSparkDraftModel(config)

    run_dir = tmp_path / "run_dspark"
    (run_dir / "train").mkdir(parents=True)
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 100, "max_pixels": 200},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"num_draft_layers": 2, "draft_layer_ids": [0, 1]},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
        "head": {"tie_word_embeddings": True},
    }
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"drafter": {"spatial_rope_mode": "replace", "spatial_rope_enabled": False}}),
        encoding="utf-8",
    )
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": "dspark",
        "draft_config": config.to_dict(),
        "attention_backend": "sdpa",
        "mask_token_id": 255,
        "block_size": 8,
        "num_anchors": 16,
        "loss_decay_gamma": 7.0,
        "target_layer_ids": [0, 1],
        "capture_layer_ids": [0, 1],
        "identity": {"model_fingerprint": "f" * 64},
        "step": 3,
        "train_recipe": {
            "name": "upstream_dspark",
            "scheme": "vlm-dflash-recipe-v1",
            "cold_start_fraction": 1.0,
            "needs_teacher": True,
            "stages": [
                {
                    "name": "e2e_tv",
                    "weighting": "decay",
                    "ce_coef": 0.0,
                    "lk_coef": 1.0,
                    "e2e_tv_coef": 1.0,
                    "lk_loss_type": "hybrid",
                    "lk_eta": 3.0,
                    "dpace_alpha": 0.5,
                    "confidence_coef": 0.25,
                }
            ],
        },
        "dspark": {
            "markov_rank": 32,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "confidence_head_alpha": 1.0,
            "sample_from_anchor": True,
        },
        "model": {key: value.clone() for key, value in draft.state_dict().items()},
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")
    return run_dir, draft


def test_export_records_train_recipe(tmp_path):
    """The resolved ``train_recipe`` must survive export: config.json,
    training_manifest.json and the model card all record it (regression for
    DSpark exports that previously dropped it, leaving differently-trained
    weights indistinguishable)."""
    run_dir, draft = _write_dspark_run(tmp_path)
    export_dir = export_run(run_dir)
    exported = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((export_dir / "training_manifest.json").read_text(encoding="utf-8"))
    card = (export_dir / "model_card.md").read_text(encoding="utf-8")
    recipe = exported["train_recipe"]
    assert exported["method"] == "dspark"
    assert recipe["name"] == "upstream_dspark"
    assert recipe["stages"][0]["confidence_coef"] == 0.25
    assert manifest["train_recipe"] == recipe
    assert "| train recipe | upstream_dspark (vlm-dflash-recipe-v1) |" in card
    model, reloaded = load_export(export_dir)
    original_state = draft.state_dict()
    loaded_state = model.state_dict()
    assert set(original_state) == set(loaded_state)
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key]), key
    assert reloaded["train_recipe"] == recipe
