"""DFly: DFlash FC context + DFlare fusion residual + optional hidden correction.

Upstream: Tencent AngelSpec ``angelspec/models/draft/dfly.py``,
``angelspec/models/dfly.py``, ``angelspec/training/dfly_trainer.py`` @
``d3412bed231025ea85d35dbf5e0af44f1ac62a5b`` (Apache-2.0); report arXiv:2607.25852.
Audit, adaptation notes and the upstream recipe split: ``docs/upstream/dfly.md``.
"""

from __future__ import annotations

import torch

from .base import AlgorithmMetadata, DrafterAlgorithm, EvidenceRecord
from .registry import register


class DFlyAlgorithm(DrafterAlgorithm):
    name = "dfly"
    metadata = AlgorithmMetadata(
        name="dfly",
        description=(
            "DFlash shared-FC context plus a per-layer fusion residual and an optional zero-init "
            "SwiGLU hidden-states correction conditioned on the previous token (DFlareV2)."
        ),
        maturity={
            "architecture": "validated",
            # Upstream's two-stage objective (LK/D-PACE -> e2e TV) is ported as
            # the config-selectable ``algorithm.train_recipe`` (objectives.py,
            # CPU-tested end to end) and its canonical 2K run is recorded
            # (R-20261014-11: AL 1.839 vs controlled_ce 1.682, 128/128 exact),
            # so training is validated like the other recorded runs.
            "training": "validated",
            "multimodal": "validated",
            "runtime": "experimental",
        },
        references={
            "paper": "arXiv:2607.25852 (AngelSpec / DFly)",
            "upstream_reference": (
                "Tencent/AngelSpec@d3412bed231025ea85d35dbf5e0af44f1ac62a5b "
                "angelspec/models/draft/dfly.py (Apache-2.0)"
            ),
            "port_notes": "docs/upstream/dfly.md",
            "algorithm_doc": "docs/algorithms/dfly.md",
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
                    "R-20261014-11 E4 canonical 2K upstream_two_stage recipe (AL 1.839 vs controlled_ce 1.682, 128/128 exact; configs/dfly/qwen2_5_vl_3b_2k_recipe.yaml); zoo-v0.3.2 archive + demo reload"
                ),
            ),
            EvidenceRecord(
                family="qwen3_vl",
                layout="single_image",
                feature_mode="offline",
                state="reduced_validation",
                evidence=(
                    "docs/algorithms/dfly.md Qwen3-VL-8B reduced validation (configs/dfly/qwen3_vl_8b_reduced.yaml)"
                ),
            ),
        ),
    )

    def validate_config(self, cfg: dict) -> None:
        drafter = cfg.get("drafter")
        if not isinstance(drafter, dict):
            raise ValueError(
                f"algorithm=dfly requires a 'drafter' config block, got {type(drafter).__name__}."
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
        from ..dfly.factory import resolve_dfly_config

        dfly = resolve_dfly_config(cfg)
        if str(dfly["fusion_init"]) not in {"distance", "uniform"}:
            raise ValueError(
                f"algorithm.dfly.fusion_init must be 'distance' (upstream) or 'uniform', "
                f"got {dfly['fusion_init']!r}."
            )
        intermediate = dfly.get("hidden_correction_intermediate_size")
        if intermediate is not None and int(intermediate) < 1:
            raise ValueError(
                "algorithm.dfly.hidden_correction_intermediate_size must be >= 1, "
                f"got {intermediate!r}."
            )
        if dfly.get("context_layer_ids") is not None and not isinstance(dfly["context_layer_ids"], (list, tuple)):
            raise ValueError("algorithm.dfly.context_layer_ids must be a list of target layer indices.")
        from ..objectives import assert_recipe_capture, resolve_recipe

        assert_recipe_capture(resolve_recipe(cfg, self.name), cfg, algorithm_name=self.name)

    def build_draft_config(self, contract: dict, cfg: dict):
        from ..dfly.factory import build_draft_config

        return build_draft_config(contract, cfg)

    def build_trainer_module(
        self,
        draft_config,
        contract: dict,
        cfg: dict,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
    ):
        from ..dfly.factory import build_trainer_module
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
        from ..contract import contract_capture_layer_ids
        from ..dfly.factory import OBJECTIVE_PROFILE, resolve_dfly_config
        from ..objectives import resolve_recipe

        drafter = cfg["drafter"]
        resolved = resolve_dfly_config(cfg)
        recipe = resolve_recipe(cfg, self.name)
        dflash_config = dict(getattr(draft_config, "dflash_config", {}) or {})
        context = dflash_config.get("context_layer_ids") or contract_capture_layer_ids(contract)
        return {
            "draft_config": draft_config.to_dict(),
            # Machine-readable objective: the selected recipe name, defaulting
            # to the shared DFlash block-CE profile (controlled_ce).
            "objective_profile": str(dflash_config.get("objective_profile", OBJECTIVE_PROFILE)),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "mask_token_id": int(drafter["mask_token_id"]),
            "block_size": int(drafter["block_size"]),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "loss_decay_gamma": float(drafter.get("loss_decay_gamma", 7.0)),
            "target_layer_ids": [int(x) for x in context],
            "capture_layer_ids": contract_capture_layer_ids(contract),
            "dfly": {
                "context_layer_ids": [int(x) for x in context],
                "fusion_trainable": bool(
                    dflash_config.get("fusion_trainable", resolved["fusion_trainable"])
                ),
                "fusion_init": str(dflash_config.get("fusion_init", resolved["fusion_init"])),
                "enable_hidden_correction": bool(
                    dflash_config.get(
                        "enable_hidden_correction", resolved["enable_hidden_correction"]
                    )
                ),
                "hidden_correction_intermediate_size": dflash_config.get(
                    "hidden_correction_intermediate_size",
                    resolved["hidden_correction_intermediate_size"],
                ),
            },
            "train_recipe": None if recipe is None else recipe.as_dict(),
        }

    def load_decoder(self, checkpoint_path, map_location: str = "cpu"):
        from ..dfly.factory import load_checkpoint

        return load_checkpoint(checkpoint_path, map_location=map_location)

    def plan_knobs(self, cfg: dict, contract: dict | None = None) -> dict:
        """Resolved DFly knobs for the training-plan digest.

        ``context_layer_ids: null`` resolves to every captured layer (contract
        required), and the correction width normalizes the same way the model
        builds it: ``null`` -> the draft/target hidden size when the correction
        is enabled, and no width at all when it is disabled (the knob has no
        effect then, so changing it must not fake a plan change).
        """
        if contract is None:
            raise ValueError(
                "dfly training-plan knobs depend on the capture layout; pass the frozen contract "
                "so context_layer_ids: null resolves to the actual ordered layer list."
            )
        from ..contract import contract_capture_layer_ids
        from ..dfly.factory import resolve_dfly_config
        from ..dflare.factory import resolve_context_layer_ids

        resolved = resolve_dfly_config(cfg)
        context = resolve_context_layer_ids(resolved, contract_capture_layer_ids(contract))
        correction_enabled = bool(resolved["enable_hidden_correction"])
        intermediate = resolved["hidden_correction_intermediate_size"]
        if not correction_enabled:
            correction_width = None
        elif intermediate is None:
            correction_width = int(contract["text"]["hidden_size"])
        else:
            correction_width = int(intermediate)
        from ..objectives import plan_recipe_knobs

        knobs = {
            "context_layer_ids": [int(layer) for layer in context],
            "separate_context_kv": False,
            "fusion_trainable": bool(resolved["fusion_trainable"]),
            "fusion_init": str(resolved["fusion_init"]),
            "enable_hidden_correction": correction_enabled,
            "hidden_correction_intermediate_size": correction_width,
        }
        knobs.update(plan_recipe_knobs(cfg, self.name))
        return knobs

    def build_inference_module(self, draft_config_dict: dict, attention_backend: str):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from ..dfly.model import Qwen3DFlyDraftModel

        config = Qwen3Config.from_dict(draft_config_dict)
        config._attn_implementation = attention_backend or "flex_attention"
        return Qwen3DFlyDraftModel(config)


register(DFlyAlgorithm())

