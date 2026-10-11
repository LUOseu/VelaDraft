"""Build the Qwen3-style DFlash draft config/model from a target contract.

Mirrors the reference ``_build_draft_config``: the draft inherits the target's
text config shape, keeps three full-attention layers, and converts the target's
MRoPE rope scaling into the draft's plain 1-D rope (the spatial conditioning is
carried by explicit ``spatial_rope_*`` DFlash settings instead).
"""

from __future__ import annotations

import torch
from torch import nn


def build_draft_config(contract: dict, cfg: dict):
    from transformers import AutoConfig
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    target_cfg = AutoConfig.from_pretrained(contract["target"]["model_path"], local_files_only=True)
    text_config = getattr(target_cfg, "text_config", target_cfg)
    drafter = cfg["drafter"]
    draft = text_config.to_dict()
    num_layers = int(drafter["num_hidden_layers"])
    draft["architectures"] = ["Qwen3DFlashDraftModel"]
    draft["num_hidden_layers"] = num_layers
    draft["layer_types"] = ["full_attention"] * num_layers
    draft["max_window_layers"] = num_layers
    draft["num_target_layers"] = int(contract["layers"]["num_target_layers"])
    draft["block_size"] = int(drafter["block_size"])
    draft["dflash_config"] = {
        "mask_token_id": int(drafter["mask_token_id"]),
        "target_layer_ids": [int(layer) for layer in contract["layers"]["draft_layer_ids"]],
        "spatial_rope_enabled": bool(drafter.get("spatial_rope_enabled", True)),
        "spatial_rope_mode": str(drafter.get("spatial_rope_mode", "replace")),
        "spatial_rope_sections": [int(section) for section in contract["rope"]["sections"]],
        "spatial_rope_interleaved": bool(contract["rope"].get("interleaved", False)),
        "visual_num_query_tokens": 0,
    }
    rope = draft.get("rope_scaling") or draft.get("rope_parameters")
    if isinstance(rope, dict) and "mrope_section" in rope:
        rope_theta = float(rope.get("rope_theta", 1000000.0))
        draft.pop("rope_scaling", None)
        draft.pop("rope_parameters", None)
        draft["rope_parameters"] = {"rope_type": "default", "rope_theta": rope_theta}
    config = Qwen3Config.from_dict(draft)
    config._attn_implementation = str(drafter.get("attention_backend", "flex_attention"))
    return config


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight: torch.Tensor):
    """Trainer module with frozen target embed/lm-head (tied if the target ties)."""
    from .draft_model import Qwen3DFlashDraftModel
    from .trainer_core import DFlashTrainerModule

    drafter = cfg["drafter"]
    draft = Qwen3DFlashDraftModel(draft_config)
    vocab_size, hidden_size = int(embed_weight.shape[0]), int(embed_weight.shape[1])
    embed = nn.Embedding(vocab_size, hidden_size, dtype=embed_weight.dtype)
    with torch.no_grad():
        embed.weight.copy_(embed_weight)
    embed.weight.requires_grad_(False)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=embed_weight.dtype)
    lm_head.weight = embed.weight
    module = DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=int(drafter["mask_token_id"]),
        block_size=int(drafter["block_size"]),
        attention_backend=str(drafter.get("attention_backend", "flex_attention")),
        num_anchors=int(drafter.get("num_anchors", 256)),
        loss_decay_gamma=float(drafter.get("loss_decay_gamma", 7.0)),
        loss_type="dflash",
    )
    return module


def load_checkpoint(path, map_location="cpu"):
    """Load a vlm-dflash draft checkpoint (own format; trusted local file)."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from .draft_model import Qwen3DFlashDraftModel

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "vlm-dflash-ckpt-v1":
        raise ValueError(f"Unrecognized checkpoint format in {path}: {checkpoint.get('format')!r}")
    config = Qwen3Config.from_dict(checkpoint["draft_config"])
    config._attn_implementation = checkpoint.get("attention_backend", "flex_attention")
    draft = Qwen3DFlashDraftModel(config)
    draft.load_state_dict(checkpoint["model"])
    draft.eval()
    return draft, checkpoint
