"""JetSpec: causal block-parallel drafting distilled against target soft logits.

Reference implementation (see ``docs/upstream/jetspec.md``): NVIDIA-NeMo's
``nemo_automodel/components/speculative/dflash/jetspec_core.py`` and
``components/loss/kd_loss.py`` (Apache-2.0). The unmodified DFlash draft
backbone is trained with (1) a block-causal in-block attention mask and (2) a
temperature-scaled forward-KL objective against the target's per-position soft
distribution. The production parallel-tree *runtime* is not ported: eval runs
the shared block-parallel decoder as a proposal reference (see the audit doc).
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata, EvidenceRecord, TrainingSignals
from .dflash import DFlashAlgorithm
from .registry import register


class JetSpecAlgorithm(DFlashAlgorithm):
    """DFlash backbone + causal in-block mask + forward-KL distillation."""

    name = "jetspec"
    metadata = AlgorithmMetadata(
        name="jetspec",
        description=(
            "JetSpec: causal in-block block-parallel drafting trained by "
            "temperature-scaled forward-KL distillation against the target's "
            "per-position soft distribution (parallel-tree runtime not ported; "
            "eval uses the shared block decoder as a proposal reference)."
        ),
        maturity={
            "architecture": "validated",
            "training": "validated",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "upstream_reference": "NVIDIA-NeMo/Automodel (Apache-2.0)",
            "port_notes": "docs/upstream/jetspec.md",
            "algorithm_doc": "docs/algorithms/jetspec.md",
        },
        capabilities={
            "image": True,
            "multi_image": True,
            "video": False,
            "parallel_tree_runtime": False,
            "eval": True,
            "export": True,
        },
        targets={"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"},
        training_signals=TrainingSignals(
            hard_label=(
                "No hard-label CE - the trainer replaces it with the teacher-distribution KL "
                "(it never degenerates to CE)."
            ),
            context_layers="Contract capture layers (unmodified DFlash backbone; no new parameters).",
            teacher_final_hidden=(
                "Required (capture.store_final_hidden=true): teacher logits are the frozen "
                "output-head projection of the stored final hidden states at the gathered "
                "positions (bit-exact reproduction of the target's own logits)."
            ),
            distribution_objective=(
                "Yes - temperature-scaled forward KL(P_target || Q_draft), T^2 scaling, uniform "
                "weighting over slots 1..S-1 (no depth decay)."
            ),
            confidence_objective="None.",
            stage="Single stage.",
            block_convention=(
                "DFlash slot convention (slot 0 anchor clean, unsupervised; S-1 proposals) with "
                "the block-causal in-block attention mask."
            ),
        ),
        evidence=(
            EvidenceRecord(
                family="qwen25_vl",
                layout="single_image",
                feature_mode="offline",
                state="canonical_validated",
                evidence=(
                    "RELEASE_VALIDATION.md canonical benchmark (AL 1.599, 128/128; causal-mask serving - tree runtime not ported) + zoo-v0.3.1 archive reload"
                ),
            ),
            EvidenceRecord(
                family="qwen3_vl",
                layout="single_image",
                feature_mode="offline",
                state="reduced_validation",
                evidence=(
                    "RELEASE_VALIDATION.md Qwen3-VL-8B reduced validation (256-row slice, 16-prompt eval AL 1.1185, 16/16 exact, export + reload)"
                ),
            ),
        ),
    )

    def validate_config(self, cfg: dict) -> None:
        drafter = cfg.get("drafter")
        if not isinstance(drafter, dict):
            raise ValueError(
                f"algorithm=jetspec requires a 'drafter' config block, got {type(drafter).__name__}."
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
            raise ValueError("algorithm=jetspec supervises block offsets 1..block_size-1; sample_from_anchor must be false.")
        if drafter.get("loss_decay_gamma", None) is not None:
            raise ValueError(
                "algorithm=jetspec uses uniform position weighting (forward KL); set "
                "drafter.loss_decay_gamma to null."
            )
        from ..jetspec.factory import resolve_jetspec_config

        jetspec = resolve_jetspec_config(cfg)
        temperature = float(jetspec["kd_temperature"])
        if temperature <= 0:
            raise ValueError(f"algorithm.jetspec.kd_temperature must be > 0, got {temperature}.")
        chunk = int(jetspec["kd_chunk_size"])
        if chunk < 0:
            raise ValueError(f"algorithm.jetspec.kd_chunk_size must be >= 0, got {chunk}.")
        capture = cfg.get("capture") or {}
        if not bool(capture.get("store_final_hidden", False)):
            raise ValueError(
                "algorithm=jetspec requires capture.store_final_hidden=true: the forward-KL teacher "
                "logits are projected from the target's captured final hidden states."
            )

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..jetspec.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..jetspec.factory import build_trainer_module

        return build_trainer_module(draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight)

    def checkpoint_metadata(self, contract: dict, cfg: dict, draft_config) -> dict:
        drafter = cfg["drafter"]
        from ..jetspec.factory import resolve_jetspec_config

        jetspec = resolve_jetspec_config(cfg)
        return {
            "draft_config": draft_config.to_dict(),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": None,
            "target_layer_ids": [int(x) for x in contract["layers"]["draft_layer_ids"]],
            "jetspec": {
                "kd_temperature": float(jetspec["kd_temperature"]),
                "kd_chunk_size": int(jetspec["kd_chunk_size"]),
                "causal_in_block": True,
            },
        }

    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        from ..jetspec.factory import load_checkpoint

        draft, checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
        # Serve the causal in-block mask the trainer used. New checkpoints carry
        # this in their draft config; checkpoints written before the field
        # existed (still JetSpec weights) are corrected here so eval/export
        # never silently fall back to the non-causal block forward.
        draft.inference_in_block_causal = True
        return draft, checkpoint

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        from ..dflash.draft_model import Qwen3DFlashDraftModel

        model = super().build_inference_module(draft_config_dict, attention_backend)
        model.inference_in_block_causal = bool(
            getattr(model, "inference_in_block_causal", False)
        ) or bool((draft_config_dict.get("dflash_config") or {}).get("inference_in_block_causal", True))
        return model


register(JetSpecAlgorithm())
