"""DFlash: the stable reference algorithm (frozen; do not repurpose).

Semantics are ported unchanged from the read-only reference
(``jacky566/Automodel_lao`` @ ``d21fb5aa``, Apache-2.0; see docs/REFERENCE.md):
block size 8, mask-token denoising over sampled anchors, decayed block CE,
block attention mask, target-layer context features. This module is the thin
adapter that exposes those semantics through the algorithm interface; the
implementation itself stays in ``vlm_dflash/dflash/``.
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata, DrafterAlgorithm
from .registry import register


class DFlashAlgorithm(DrafterAlgorithm):
    name = "dflash"
    metadata = AlgorithmMetadata(
        name="dflash",
        description=(
            "DFlash block-parallel drafter: mask-token denoising over sampled "
            "anchors with decayed block cross-entropy."
        ),
        maturity={
            "architecture": "stable",
            "training": "validated",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "upstream_reference": "jacky566/Automodel_lao@d21fb5aa (Apache-2.0)",
            "port_notes": "docs/REFERENCE.md",
            "algorithm_doc": "docs/algorithms/dflash.md",
        },
        capabilities={"image": True, "multi_image": False, "video": False},
        targets={"qwen25_vl": "validated", "qwen3_vl": "validated"},
    )

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #

    def validate_config(self, cfg: dict) -> None:
        drafter = cfg.get("drafter")
        if not isinstance(drafter, dict):
            raise ValueError(
                f"algorithm=dflash requires a 'drafter' config block, got {type(drafter).__name__}."
            )
        block_size = int(drafter.get("block_size", 8))
        if block_size < 2:
            raise ValueError(f"drafter.block_size must be >= 2, got {block_size}.")
        num_layers = int(drafter.get("num_hidden_layers", 3))
        if num_layers < 1:
            raise ValueError(f"drafter.num_hidden_layers must be >= 1, got {num_layers}.")
        num_anchors = int(drafter.get("num_anchors", 256))
        if num_anchors < 1:
            raise ValueError(f"drafter.num_anchors must be >= 1, got {num_anchors}.")
        gamma = float(drafter.get("loss_decay_gamma", 7.0))
        if gamma <= 0:
            raise ValueError(f"drafter.loss_decay_gamma must be > 0, got {gamma}.")

    # ------------------------------------------------------------------ #
    # training side
    # ------------------------------------------------------------------ #

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..dflash.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..dflash.factory import build_trainer_module

        return build_trainer_module(
            draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight
        )

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        drafter = cfg["drafter"]
        return {
            "draft_config": draft_config.to_dict(),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": float(drafter.get("loss_decay_gamma", 7.0)),
            "target_layer_ids": [int(x) for x in contract["layers"]["draft_layer_ids"]],
        }

    # ------------------------------------------------------------------ #
    # inference side
    # ------------------------------------------------------------------ #

    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        from ..dflash.factory import load_checkpoint

        return load_checkpoint(checkpoint_path, map_location=map_location)

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from ..dflash.draft_model import Qwen3DFlashDraftModel

        config = Qwen3Config.from_dict(draft_config_dict)
        config._attn_implementation = attention_backend or "flex_attention"
        return Qwen3DFlashDraftModel(config)

register(DFlashAlgorithm())
