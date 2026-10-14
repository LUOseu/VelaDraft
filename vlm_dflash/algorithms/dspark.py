"""DSpark: DFlash plus a low-rank Markov logit bias and a confidence head.

Reference implementations (see ``docs/upstream/dspark.md``):
``vllm-project/speculators`` @ ``55ebbd5b`` (Apache-2.0) and
``NVIDIA-NeMo/Automodel`` @ ``bb5b34ce`` (Apache-2.0). The draft backbone, block
CE objective and verification protocol are the frozen DFlash ones; DSpark adds
the semi-autoregressive in-block dependency the block-parallel drafter lacks.
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata, DrafterAlgorithm, EvidenceRecord
from .registry import register


class DSparkAlgorithm(DrafterAlgorithm):
    name = "dspark"
    metadata = AlgorithmMetadata(
        name="dspark",
        description=(
            "DSpark: DFlash block-parallel drafter plus a low-rank Markov "
            "logit-bias head (semi-autoregressive in-block sampling) and a "
            "per-slot acceptance-confidence head."
        ),
        maturity={
            "architecture": "validated",
            "training": "validated",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "upstream_reference": "vllm-project/speculators@55ebbd5b, NVIDIA-NeMo/Automodel@bb5b34ce (Apache-2.0)",
            "port_notes": "docs/upstream/dspark.md",
            "algorithm_doc": "docs/algorithms/dspark.md",
        },
        capabilities={"image": True, "multi_image": True, "video": False, "eval": True, "export": True},
        targets={"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"},
        evidence=(
            EvidenceRecord(
                family="qwen25_vl",
                layout="single_image",
                feature_mode="offline",
                state="canonical_validated",
                evidence=(
                    "RELEASE_VALIDATION.md canonical benchmark (AL 1.735, 128/128; TextVQA 1.253) + zoo-v0.3.1 archive reload"
                ),
            ),
            EvidenceRecord(
                family="qwen3_vl",
                layout="single_image",
                feature_mode="offline",
                state="reduced_validation",
                evidence=(
                    "RELEASE_VALIDATION.md Qwen3-VL-8B reduced validation (256-row slice, 16-prompt eval AL 1.2212, 16/16 exact, export + reload)"
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
                f"algorithm=dspark requires a 'drafter' config block, got {type(drafter).__name__}."
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
        from ..dspark.factory import resolve_dspark_config

        dspark = resolve_dspark_config(cfg)
        markov_rank = int(dspark["markov_rank"])
        if markov_rank < 0:
            raise ValueError(f"algorithm.dspark.markov_rank must be >= 0, got {markov_rank}.")
        head_type = str(dspark["markov_head_type"]).lower()
        if head_type not in {"vanilla", "gated", "rnn"}:
            raise ValueError(
                f"algorithm.dspark.markov_head_type must be vanilla/gated/rnn, got {head_type!r}."
            )
        if bool(dspark["confidence_head_with_markov"]) and markov_rank == 0:
            raise ValueError(
                "algorithm.dspark.confidence_head_with_markov requires markov_rank > 0 "
                "(disable it or enable the Markov head)."
            )
        alpha = float(dspark["confidence_head_alpha"])
        if alpha < 0:
            raise ValueError(f"algorithm.dspark.confidence_head_alpha must be >= 0, got {alpha}.")

        from ..objectives import assert_recipe_capture, resolve_recipe

        recipe = resolve_recipe(cfg, self.name)
        assert_recipe_capture(recipe, cfg, algorithm_name=self.name)
        if recipe is not None:
            trains_confidence = any(float(stage.confidence_coef) > 0.0 for stage in recipe.stages)
            if trains_confidence:
                if not bool(dspark["enable_confidence_head"]):
                    raise ValueError(
                        f"algorithm=dspark train_recipe={recipe.name!r} trains the confidence head "
                        "against the analytic acceptance target; enable_confidence_head must be "
                        "true (or drop the recipe)."
                    )
                if float(dspark["confidence_head_alpha"]) != 1.0:
                    raise ValueError(
                        "algorithm.dspark.confidence_head_alpha must stay at the upstream default "
                        f"1.0 under train_recipe={recipe.name!r}; set train_recipe.confidence_coef "
                        "to change the confidence weight."
                    )
            elif bool(dspark["enable_confidence_head"]):
                raise ValueError(
                    f"algorithm=dspark train_recipe={recipe.name!r} has no confidence term "
                    "(confidence_coef=0), so the enabled confidence head would never receive "
                    "gradient; set algorithm.dspark.enable_confidence_head=false or "
                    "train_recipe.confidence_coef > 0."
                )

    # ------------------------------------------------------------------ #
    # training side
    # ------------------------------------------------------------------ #

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..dspark.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..dspark.factory import build_trainer_module

        return build_trainer_module(
            draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight
        )

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        drafter = cfg["drafter"]
        # The built draft config is the source of truth for the head knobs
        # (checkpoint metadata must record what was actually constructed).
        dflash_config = dict(getattr(draft_config, "dflash_config", {}) or {})
        from ..dspark.factory import DEFAULTS, resolve_dspark_config
        from ..objectives import resolve_recipe

        resolved = resolve_dspark_config(cfg)
        dspark = {key: dflash_config.get(key, resolved[key]) for key in DEFAULTS}
        dspark["confidence_head_alpha"] = float(resolved["confidence_head_alpha"])
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
            "dspark": {
                "markov_rank": int(dspark["markov_rank"]),
                "markov_head_type": str(dspark["markov_head_type"]),
                "enable_confidence_head": bool(dspark["enable_confidence_head"]),
                "confidence_head_with_markov": bool(dspark["confidence_head_with_markov"]),
                "confidence_head_alpha": float(dspark["confidence_head_alpha"]),
                "sample_from_anchor": bool(dspark["sample_from_anchor"]),
            },
        }

    # ------------------------------------------------------------------ #
    # inference side
    # ------------------------------------------------------------------ #

    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        from ..dspark.factory import load_checkpoint

        return load_checkpoint(checkpoint_path, map_location=map_location)

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from ..dspark.model import Qwen3DSparkDraftModel

        config = Qwen3Config.from_dict(draft_config_dict)
        config._attn_implementation = attention_backend or "flex_attention"
        return Qwen3DSparkDraftModel(config)


register(DSparkAlgorithm())
