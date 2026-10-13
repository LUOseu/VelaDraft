"""DFlash2 (experimental): local dynamic convolutions + candidate selector.

Status: **Experimental**. The inference architecture and checkpoint contract come
from the maintainer implementation (vllm-project/speculators @ 55ebbd5b, itself
adapted from z-lab/dflash @ 07ebd93d); the *training objective* is undisclosed
upstream and this port implements speculators' experimental split objective
(unary forward KL + K-way selector CE with target injection). See
``docs/upstream/dflash2.md``; do not read this as an official reproduction.
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata, DrafterAlgorithm
from .registry import register


class DFlash2Algorithm(DrafterAlgorithm):
    name = "dflash2"
    metadata = AlgorithmMetadata(
        name="dflash2",
        description=(
            "DFlash2 block drafter: DFlash backbone + local dynamic convolutions "
            "and a predecessor-conditioned candidate selector (experimental split "
            "objective reconstructed from the maintainer implementation)."
        ),
        maturity={
            "architecture": "validated",
            "training": "experimental",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "paper": "no paper; Z Lab DFlash2 blog (https://inco.ai/blog/dflash2/)",
            "upstream_reference": (
                "z-lab/dflash@07ebd93d via vllm-project/speculators@55ebbd5b (MIT model code / Apache-2.0)"
            ),
            "port_notes": "docs/upstream/dflash2.md",
            "algorithm_doc": "docs/algorithms/dflash2.md",
        },
        capabilities={"image": True, "multi_image": False, "video": False, "parallel_tree_runtime": False, "eval": True, "export": True},
        targets={"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"},
    )

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #

    def validate_config(self, cfg: dict) -> None:
        drafter = cfg.get("drafter")
        if not isinstance(drafter, dict):
            raise ValueError(
                f"algorithm=dflash2 requires a 'drafter' config block, got {type(drafter).__name__}."
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
        if drafter.get("sample_from_anchor", False):
            raise ValueError(
                "algorithm=dflash2 follows the DFlash slot convention (the anchor is the verifier's "
                "bonus token); drafter.sample_from_anchor must be false."
            )
        if drafter.get("projector_type", None) is not None:
            raise ValueError("algorithm=dflash2 does not combine with the Domino correction head.")
        capture = cfg.get("capture") or {}
        if not bool(capture.get("store_final_hidden", False)):
            raise ValueError(
                "algorithm=dflash2 requires capture.store_final_hidden=true: the unary objective distills "
                "the target distribution projected from its captured final hidden states, and the selector's "
                "hard label is that projection's argmax."
            )
        from ..dflash2.factory import resolve_dflash2_config

        dflash2 = resolve_dflash2_config(cfg)
        kernel = int(dflash2["conv_kernel_size"])
        if kernel < 1:
            raise ValueError(f"algorithm.dflash2.conv_kernel_size must be >= 1, got {kernel}.")
        if kernel > block_size:
            raise ValueError(
                f"algorithm.dflash2.conv_kernel_size ({kernel}) cannot exceed drafter.block_size ({block_size})."
            )
        group_size = int(dflash2["conv_group_size"])
        if group_size < 1:
            raise ValueError(f"algorithm.dflash2.conv_group_size must be >= 1, got {group_size}.")
        rank = int(dflash2["selector_rank"])
        if rank < 1:
            raise ValueError(f"algorithm.dflash2.selector_rank must be >= 1, got {rank}.")
        top_k = int(dflash2["selector_top_k"])
        if top_k < 1:
            raise ValueError(f"algorithm.dflash2.selector_top_k must be >= 1, got {top_k}.")
        if top_k > 1 << 16:
            raise ValueError(f"algorithm.dflash2.selector_top_k ({top_k}) is implausibly large.")
        alpha = float(dflash2["selector_loss_alpha"])
        if alpha < 0:
            raise ValueError(f"algorithm.dflash2.selector_loss_alpha must be >= 0, got {alpha}.")
        chunk = int(dflash2["kd_chunk_size"])
        if chunk < 0:
            raise ValueError(f"algorithm.dflash2.kd_chunk_size must be >= 0, got {chunk}.")

    # ------------------------------------------------------------------ #
    # training side
    # ------------------------------------------------------------------ #

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..dflash2.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..dflash2.factory import build_trainer_module

        return build_trainer_module(
            draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight
        )

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        drafter = cfg["drafter"]
        from ..dflash2.factory import resolve_dflash2_config

        dflash2 = resolve_dflash2_config(cfg)
        gamma = drafter.get("loss_decay_gamma", 4.0)
        return {
            "draft_config": draft_config.to_dict(),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": None if gamma is None else float(gamma),
            "target_layer_ids": [int(x) for x in contract["layers"]["draft_layer_ids"]],
            "dflash2": {
                "conv_kernel_size": int(dflash2["conv_kernel_size"]),
                "conv_group_size": int(dflash2["conv_group_size"]),
                "selector_rank": int(dflash2["selector_rank"]),
                "selector_top_k": int(dflash2["selector_top_k"]),
                "selector_loss_alpha": float(dflash2["selector_loss_alpha"]),
                "kd_chunk_size": int(dflash2["kd_chunk_size"]),
                # Upstream does not publish a training objective; this flag marks
                # the recorded run as the experimental split-objective port.
                "objective": "experimental-split (unary forward-KL + selector K-way CE)",
            },
        }

    # ------------------------------------------------------------------ #
    # inference side
    # ------------------------------------------------------------------ #

    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        from ..dflash2.factory import load_checkpoint

        return load_checkpoint(checkpoint_path, map_location=map_location)

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from ..dflash2.model import Qwen3DFlash2DraftModel

        config = Qwen3Config.from_dict(draft_config_dict)
        config._attn_implementation = attention_backend or "flex_attention"
        return Qwen3DFlash2DraftModel(config)


register(DFlash2Algorithm())
