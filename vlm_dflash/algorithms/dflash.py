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

from .base import AlgorithmMetadata, DrafterAlgorithm, EvidenceRecord, TrainingSignals
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
        capabilities={"image": True, "multi_image": True, "video": False, "eval": True, "export": True},
        targets={"qwen25_vl": "validated", "qwen3_vl": "validated"},
        training_signals=TrainingSignals(
            hard_label=(
                "Yes - decayed block CE on regenerated targets; slots 1..block_size-1 "
                "(slot 0 is the clean anchor, unsupervised)."
            ),
            context_layers=(
                "Contract capture layers (drafter.capture_layer_ids; fused [T*H] context "
                "projection shared by every draft layer)."
            ),
            teacher_final_hidden=(
                "Not used by the default objective; required by train_recipe: upstream_two_stage."
            ),
            distribution_objective=(
                "None by default; LK (hybrid KL/TV) + end-to-end multi-step TV under the recipe."
            ),
            confidence_objective="None.",
            stage=(
                "Single stage by default; the recipe switches LK cold start -> e2e TV at "
                "cold_start_fraction."
            ),
            block_convention=(
                "DFlash convention: slot 0 = clean anchor (unsupervised), slots 1..S-1 "
                "supervised; S-1 proposals per round."
            ),
        ),
        evidence=(
            EvidenceRecord(
                family="qwen25_vl",
                layout="single_image",
                feature_mode="offline",
                state="canonical_validated",
                evidence=(
                    "RELEASE_VALIDATION.md canonical benchmark (frozen 128p baseline AL 1.6298, 128/128 exact; tests/reference_dflash/manifest.json); zoo-v0.3.0 export reload"
                ),
            ),
            EvidenceRecord(
                family="qwen3_vl",
                layout="single_image",
                feature_mode="offline",
                state="canonical_validated",
                evidence=(
                    "RELEASE_VALIDATION.md targets (full 2K pipeline AL 1.665, 32/32 exact; zoo-v0.3.1 8B export reload)"
                ),
            ),
            EvidenceRecord(
                family="qwen25_vl",
                layout="multi_image+multi_turn",
                feature_mode="offline",
                state="gpu_smoke",
                evidence=(
                    "R-20261014-08 E2 smoke (2-image/2-turn rows, 16 train + 4 val, token-exact eval; configs/smoke_multiturn.yaml; recorded before the R3/R4 review remediations)"
                ),
            ),
            EvidenceRecord(
                family="smolvlm",
                layout="single_image",
                feature_mode="offline",
                state="gpu_smoke",
                evidence=(
                    "R-20261014-17 (SmolVLM2-2.2B smoke: 16 train + 4 val synthetic rows, token-exact 4/4; single-image only; no quality claim)"
                ),
            ),
        ),
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
        from ..objectives import assert_recipe_capture, resolve_recipe

        assert_recipe_capture(resolve_recipe(cfg, self.name), cfg, algorithm_name=self.name)

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
        from ..objectives import resolve_recipe

        return build_trainer_module(
            draft_config,
            contract,
            cfg,
            embed_weight,
            lm_head_weight=lm_head_weight,
            module_kwargs={"train_recipe": resolve_recipe(cfg, self.name)},
        )

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        from ..objectives import resolve_recipe

        drafter = cfg["drafter"]
        recipe = resolve_recipe(cfg, self.name)
        return {
            "draft_config": draft_config.to_dict(),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": float(drafter.get("loss_decay_gamma", 7.0)),
            "target_layer_ids": [int(x) for x in contract["layers"]["draft_layer_ids"]],
            "train_recipe": None if recipe is None else recipe.as_dict(),
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
