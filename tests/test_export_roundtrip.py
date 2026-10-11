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
    assert exported["target"]["model_fingerprint"] == "f" * 64
    assert exported["data"]["num_train"] == 4
    card = (export_dir / "model_card.md").read_text(encoding="utf-8")
    assert "tiny-vl" in card
