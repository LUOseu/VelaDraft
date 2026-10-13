"""Domino: DFlash backbone plus a causal GRU low-rank logit correction.

Reference implementation (see ``docs/upstream/domino.md``): the maintainer
implementation of SpecForge#571 as carried by NVIDIA-NeMo/Automodel
(Apache-2.0): ``nemo_automodel/components/speculative/dflash/domino_core.py``
and ``examples/speculative/dflash/qwen2_5_vl_domino.yaml``. The draft backbone,
block CE objective and verification protocol are the frozen DFlash ones; Domino
adds the causal correction head and the base-anchor curriculum.
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata
from .dflash import DFlashAlgorithm
from .registry import register


class DominoAlgorithm(DFlashAlgorithm):
    """DFlash plus a GRU-conditioned causal correction and base-anchor curriculum."""

    name = "domino"
    metadata = AlgorithmMetadata(
        name="domino",
        description=(
            "Domino: DFlash block-parallel drafter plus a causal GRU low-rank "
            "logit correction, trained with a base-anchor curriculum "
            "(final and base logits jointly supervised)."
        ),
        maturity={
            "architecture": "validated",
            "training": "validated",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "upstream_reference": "SpecForge#571 via NVIDIA-NeMo/Automodel (Apache-2.0)",
            "port_notes": "docs/upstream/domino.md",
            "algorithm_doc": "docs/algorithms/domino.md",
        },
        capabilities={"image": True, "multi_image": False, "video": False, "eval": True, "export": True},
        targets={"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"},
    )

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #

    def validate_config(self, cfg: dict) -> None:
        super().validate_config(cfg)
        from ..domino.factory import resolve_domino_config

        domino = resolve_domino_config(cfg)
        emb_dim = int(domino["emb_dim"])
        if emb_dim < 1:
            raise ValueError(f"algorithm.domino.emb_dim must be >= 1, got {emb_dim}.")
        gru_hidden_dim = int(domino["gru_hidden_dim"])
        if gru_hidden_dim < 1:
            raise ValueError(
                f"algorithm.domino.gru_hidden_dim must be >= 1, got {gru_hidden_dim}."
            )
        prefix_len = int(domino["pure_draft_prefix_len"])
        if prefix_len < 0:
            raise ValueError(
                f"algorithm.domino.pure_draft_prefix_len must be >= 0, got {prefix_len}."
            )
        block_size = int(cfg["drafter"].get("block_size", 8))
        suffix_start = prefix_len if bool(domino["shift_label"]) else 1 + prefix_len
        if not 0 <= suffix_start < block_size:
            raise ValueError(
                "algorithm.domino.pure_draft_prefix_len leaves no corrected block position: "
                f"suffix_start={suffix_start} for block_size={block_size}."
            )
        lambda_start = float(domino["lambda_base_start"])
        if not 0.0 <= lambda_start <= 1.0:
            raise ValueError(
                f"algorithm.domino.lambda_base_start must be in [0, 1], got {lambda_start}."
            )
        decay_ratio = float(domino["lambda_base_decay_ratio"])
        if not 0.0 < decay_ratio <= 1.0:
            raise ValueError(
                "algorithm.domino.lambda_base_decay_ratio must be in (0, 1], got "
                f"{decay_ratio}."
            )

    # ------------------------------------------------------------------ #
    # training side
    # ------------------------------------------------------------------ #

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..domino.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..domino.factory import build_trainer_module

        return build_trainer_module(
            draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight
        )

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        drafter = cfg["drafter"]
        # The built draft config is the source of truth for the head knobs
        # (checkpoint metadata must record what was actually constructed).
        dflash_config = dict(getattr(draft_config, "dflash_config", {}) or {})
        from ..domino.factory import resolve_domino_config

        resolved = resolve_domino_config(cfg)
        return {
            "draft_config": draft_config.to_dict(),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": float(drafter.get("loss_decay_gamma", 7.0)),
            "target_layer_ids": [int(x) for x in contract["layers"]["draft_layer_ids"]],
            "domino": {
                "emb_dim": int(dflash_config.get("emb_dim", resolved["emb_dim"])),
                "gru_hidden_dim": int(dflash_config.get("gru_hidden_dim", resolved["gru_hidden_dim"])),
                "pure_draft_prefix_len": int(
                    dflash_config.get("pure_draft_prefix_len", resolved["pure_draft_prefix_len"])
                ),
                "shift_label": bool(dflash_config.get("shift_label", resolved["shift_label"])),
                "lambda_base_start": float(resolved["lambda_base_start"]),
                "lambda_base_decay_ratio": float(resolved["lambda_base_decay_ratio"]),
                "enable_correction_head": bool(resolved["enable_correction_head"]),
            },
        }


register(DominoAlgorithm())
