"""DFlare: DFlash with learnable per-draft-layer target fusion.

Upstream: Tencent AngelSpec ``angelspec/models/draft/dflare.py`` @
``d3412bed231025ea85d35dbf5e0af44f1ac62a5b`` (MIT); paper arXiv:2606.02091.
Audit and adaptation notes: ``docs/upstream/dflare.md``.
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata, DrafterAlgorithm
from .registry import register


class DFlareAlgorithm(DrafterAlgorithm):
    name = "dflare"
    metadata = AlgorithmMetadata(
        name="dflare",
        description=(
            "DFlash block-parallel drafter with a learnable softmax fusion over the captured "
            "target layers per draft layer, and dedicated context K/V projections."
        ),
        maturity={
            "architecture": "validated",
            "training": "validated",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "paper": "arXiv:2606.02091 (DFlare)",
            "upstream_reference": (
                "Tencent/AngelSpec@d3412bed231025ea85d35dbf5e0af44f1ac62a5b "
                "angelspec/models/draft/dflare.py (MIT)"
            ),
            "port_notes": "docs/upstream/dflare.md",
            "algorithm_doc": "docs/algorithms/dflare.md",
        },
        capabilities={"image": True, "multi_image": False, "video": False, "eval": True, "export": True},
        targets={"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"},
    )

    def validate_config(self, cfg: dict) -> None:
        drafter = cfg.get("drafter")
        if not isinstance(drafter, dict):
            raise ValueError(
                f"algorithm=dflare requires a 'drafter' config block, got {type(drafter).__name__}."
            )
        block_size = int(drafter.get("block_size", 8))
        if block_size < 2:
            raise ValueError(f"drafter.block_size must be >= 2, got {block_size}.")
        num_layers = int(drafter.get("num_hidden_layers", 3))
        if num_layers < 1:
            raise ValueError(f"drafter.num_hidden_layers must be >= 1, got {num_layers}.")
        gamma = float(drafter.get("loss_decay_gamma", 7.0))
        if gamma <= 0:
            raise ValueError(f"drafter.loss_decay_gamma must be > 0, got {gamma}.")
        from ..dflare.factory import resolve_dflare_config

        dflare = resolve_dflare_config(cfg)
        if str(dflare["fusion_init"]) not in {"distance", "uniform"}:
            raise ValueError(
                f"algorithm.dflare.fusion_init must be 'distance' (upstream) or 'uniform', "
                f"got {dflare['fusion_init']!r}."
            )
        if dflare.get("context_layer_ids") is not None and not isinstance(dflare["context_layer_ids"], (list, tuple)):
            raise ValueError("algorithm.dflare.context_layer_ids must be a list of target layer indices.")

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..dflare.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..dflare.factory import build_trainer_module

        return build_trainer_module(
            draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight
        )

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        from ..contract import contract_capture_layer_ids
        from ..dflare.factory import resolve_dflare_config

        drafter = cfg["drafter"]
        resolved = resolve_dflare_config(cfg)
        dflash_config = dict(getattr(draft_config, "dflash_config", {}) or {})
        context = dflash_config.get("context_layer_ids") or contract_capture_layer_ids(contract)
        return {
            "draft_config": draft_config.to_dict(),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": float(drafter.get("loss_decay_gamma", 7.0)),
            "target_layer_ids": [int(x) for x in context],
            "capture_layer_ids": contract_capture_layer_ids(contract),
            "dflare": {
                "context_layer_ids": [int(x) for x in context],
                "separate_context_kv": bool(
                    dflash_config.get("separate_context_kv", resolved["separate_context_kv"])
                ),
                "fusion_trainable": bool(
                    dflash_config.get("fusion_trainable", resolved["fusion_trainable"])
                ),
                "fusion_init": str(dflash_config.get("fusion_init", resolved["fusion_init"])),
            },
        }

    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        from ..dflare.factory import load_checkpoint

        return load_checkpoint(checkpoint_path, map_location=map_location)

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from ..dflare.model import Qwen3DFlareDraftModel

        config = Qwen3Config.from_dict(draft_config_dict)
        config._attn_implementation = attention_backend or "flex_attention"
        return Qwen3DFlareDraftModel(config)


register(DFlareAlgorithm())

