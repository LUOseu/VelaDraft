"""vLLM bridge: export -> speculators draft conversion (CPU, no vLLM needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.export.vllm_bridge import (  # noqa: E402
    build_speculators_config,
    expected_weight_keys,
    export_vllm_draft,
)

HIDDEN = 16
NUM_LAYERS = 2
TARGET_LAYER_IDS = [1, 3]
MASK_TOKEN_ID = 151669


def _export_config(**overrides) -> dict:
    draft_config = {
        "vocab_size": 256,
        "hidden_size": HIDDEN,
        "intermediate_size": 32,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "attention_bias": False,
        "architectures": ["Qwen3DFlashDraftModel"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "dflash_config": {"mask_token_id": MASK_TOKEN_ID, "target_layer_ids": TARGET_LAYER_IDS},
    }
    config = {
        "format": "vlm-dflash-export-v1",
        "created": "2026-10-14T00:00:00+0800",
        "method": "dflash",
        "target": {"model_path": "/models/Qwen2.5-VL-3B-Instruct", "model_type": "qwen2_5_vl"},
        "drafter": {
            "num_hidden_layers": NUM_LAYERS,
            "block_size": 8,
            "mask_token_id": MASK_TOKEN_ID,
            "target_layer_ids": TARGET_LAYER_IDS,
            "spatial_rope": {"sections": [2, 1, 1], "mode": "replace", "enabled": True},
            "draft_config": draft_config,
        },
    }
    config.update(overrides)
    return config


def _write_export(tmp_path: Path, config: dict) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {}
    for key in expected_weight_keys(NUM_LAYERS):
        tensors[key] = torch.zeros(4) if key != "fc.weight" else torch.zeros(HIDDEN, HIDDEN * len(TARGET_LAYER_IDS))
    save_file(tensors, export_dir / "model.safetensors")
    return export_dir


def test_speculators_mapping_matches_vllm_derivation():
    config = build_speculators_config(_export_config(), target_path="/models/tiny-target")
    assert config["speculators_model_type"] == "dflash"
    assert config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] == 7
    assert config["speculators_config"]["verifier"]["name_or_path"] == "/models/tiny-target"
    assert config["mask_token_id"] == MASK_TOKEN_ID
    assert config["target_hidden_size"] == HIDDEN
    assert config["draft_vocab_size"] == 256
    # vLLM aux index == HF output_hidden_states index == vlm-dflash layer_id + 1.
    aux = config["aux_hidden_state_layer_ids"]
    assert aux == [layer + 1 for layer in TARGET_LAYER_IDS]
    # Emulate vLLM's speculators updater (algos.update_dflash): it derives its
    # internal target_layer_ids as aux - 1 and must reproduce the export ids.
    assert [index - 1 for index in aux] == TARGET_LAYER_IDS
    # The updater owns architectures / dflash_config; the layer config must not
    # shadow them.
    assert "architectures" not in config["transformer_layer_config"]
    assert "dflash_config" not in config["transformer_layer_config"]
    assert config["transformer_layer_config"]["hidden_size"] == HIDDEN


def test_export_vllm_draft_writes_weights_and_notes(tmp_path):
    export_dir = _write_export(tmp_path, _export_config())
    out_dir = tmp_path / "draft"
    summary = export_vllm_draft(export_dir, out_dir)
    assert summary["aux_hidden_state_layer_ids"] == [2, 4]
    assert summary["weights"]["num_tensors"] == len(expected_weight_keys(NUM_LAYERS))
    assert summary["weights"]["own_embed_tokens"] is False
    assert summary["weights"]["own_lm_head"] is False
    assert (out_dir / "model.safetensors").read_bytes() == (export_dir / "model.safetensors").read_bytes()
    written = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert written["aux_hidden_state_layer_ids"] == [2, 4]
    note = (out_dir / "VLLM_DRAFT.md").read_text(encoding="utf-8")
    assert "spatial_rope" in note  # the MRoPE caveat is recorded for operators
    assert "1-D RoPE" in note


def test_speculative_tokens_override():
    config = build_speculators_config(_export_config(), speculative_tokens=3)
    assert config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] == 3
    with pytest.raises(ValueError):
        build_speculators_config(_export_config(), speculative_tokens=0)


def test_rejects_other_methods():
    with pytest.raises(ValueError, match="DFlash block drafters only"):
        build_speculators_config(_export_config(method="dfly"))


def test_weight_layout_is_checked(tmp_path):
    export_dir = _write_export(tmp_path, _export_config())
    from safetensors.torch import load_file

    tensors = dict(load_file(export_dir / "model.safetensors"))
    del tensors["layers.1.mlp.down_proj.weight"]
    save_file(tensors, export_dir / "model.safetensors")
    with pytest.raises(ValueError, match="missing expected tensors"):
        export_vllm_draft(export_dir, tmp_path / "draft")
