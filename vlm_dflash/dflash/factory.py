"""Build the Qwen3-style DFlash draft config/model from a target contract.

Mirrors the reference ``_build_draft_config``: the draft inherits the target's
text config shape, keeps three full-attention layers, and converts the target's
MRoPE rope scaling into the draft's plain 1-D rope (the spatial conditioning is
carried by explicit ``spatial_rope_*`` DFlash settings instead).
"""

from __future__ import annotations

import torch
from torch import nn


def model_attention_implementation(attention_backend: str | None) -> str:
    """Map a VelaDraft training backend to a transformers attention implementation.

    Chunked/fused DFlash backends are implemented inside this repository's
    attention module and need a materialised mask spec, not a transformers
    backend. They therefore use the ordinary ``sdpa`` implementation for the
    projection/backend dispatch while ``DFlashTrainerModule.attention_backend``
    keeps the requested path for mask construction and diagnostics.
    """
    requested = str(attention_backend or "flex_attention")
    if requested == "flex_attention":
        return "flex_attention"
    if requested in ("sdpa", "sdpa_chunked", "npu_fusion"):
        return "sdpa"
    raise ValueError(
        "attention_backend must be one of flex_attention, sdpa, sdpa_chunked or npu_fusion; "
        f"got {requested!r}."
    )


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
    # The draft is always a dense Qwen3 stack. MoE targets (Qwen3.5/3.6-MoE)
    # carry no dense ``intermediate_size`` in their text config, so the draft
    # MLP width must come from the drafter block instead of a Qwen3Config
    # default; an explicit value also overrides a dense target's width.
    if drafter.get("intermediate_size") is not None:
        draft["intermediate_size"] = int(drafter["intermediate_size"])
    elif draft.get("intermediate_size") is None:
        raise ValueError(
            "Target text config has no dense intermediate_size (MoE target); set drafter.intermediate_size."
        )
    draft["num_target_layers"] = int(contract["layers"]["num_target_layers"])
    # Per-layer width of the captured target features (the fusion models need it
    # to validate their stacked context; equal to the draft width in this stack).
    # Optional: legacy/synthetic contracts without a ``text`` block fall back to
    # the draft hidden size, matching the old single-layout stores.
    text_hidden = contract.get("text", {}).get("hidden_size")
    if text_hidden is not None:
        draft["target_hidden_size"] = int(text_hidden)
    draft["block_size"] = int(drafter["block_size"])
    draft["dflash_config"] = {
        "mask_token_id": int(drafter["mask_token_id"]),
        "target_layer_ids": [int(layer) for layer in contract["layers"]["draft_layer_ids"]],
        "spatial_rope_enabled": bool(drafter.get("spatial_rope_enabled", True)),
        "spatial_rope_mode": str(drafter.get("spatial_rope_mode", "replace")),
        "spatial_rope_sections": (
            [int(section) for section in contract["rope"]["sections"]]
            if contract["rope"].get("sections") is not None
            else None
        ),
        "spatial_rope_interleaved": bool(contract["rope"].get("interleaved", False)),
        "visual_num_query_tokens": 0,
    }
    rope = draft.get("rope_scaling") or draft.get("rope_parameters")
    if isinstance(rope, dict) and "mrope_section" in rope:
        rope_theta = float(rope.get("rope_theta", 1000000.0))
        draft.pop("rope_scaling", None)
        draft.pop("rope_parameters", None)
        draft["rope_parameters"] = {"rope_type": "default", "rope_theta": rope_theta}
        # Keep the target's partial rotary as resolved by the contract: the
        # draft rotates the same leading channels and passes the rest through.
        partial_rotary_factor = float(contract["rope"].get("partial_rotary_factor") or 1.0)
        if partial_rotary_factor != 1.0:
            draft["rope_parameters"]["partial_rotary_factor"] = partial_rotary_factor
    config = Qwen3Config.from_dict(draft)
    config._attn_implementation = model_attention_implementation(
        str(drafter.get("attention_backend", "flex_attention"))
    )
    return config


def build_trainer_module(
    draft_config,
    contract: dict,
    cfg: dict,
    embed_weight: torch.Tensor,
    lm_head_weight: torch.Tensor | None = None,
    *,
    model_class=None,
    module_class=None,
    module_kwargs: dict | None = None,
):
    """Trainer module with the target's frozen embed and output head.

    The output head must be the one inference actually samples through
    (``target.lm_head`` in ``spec_generate``): tied targets share the embedding
    matrix, untied targets (e.g. Qwen3-VL-8B) pass their own ``lm_head_weight``
    captured alongside the features. Passing ``None`` for an untied target would
    silently train against the embeddings and zero out acceptance at eval.

    ``model_class`` / ``module_class`` / ``module_kwargs`` let algorithm plugins
    (e.g. DSpark) reuse this frozen-embed/lm_head wiring with their own draft
    backbone model and trainer module.
    """
    from .draft_model import Qwen3DFlashDraftModel
    from .trainer_core import DFlashTrainerModule

    model_class = model_class or Qwen3DFlashDraftModel
    module_class = module_class or DFlashTrainerModule
    drafter = cfg["drafter"]
    draft = model_class(draft_config)
    vocab_size, hidden_size = int(embed_weight.shape[0]), int(embed_weight.shape[1])
    embed = nn.Embedding(vocab_size, hidden_size, dtype=embed_weight.dtype)
    with torch.no_grad():
        embed.weight.copy_(embed_weight)
    embed.weight.requires_grad_(False)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=embed_weight.dtype)
    if lm_head_weight is None:
        lm_head.weight = embed.weight
    else:
        if tuple(lm_head_weight.shape) != (vocab_size, hidden_size):
            raise ValueError(
                "lm_head_weight must have shape "
                f"{(vocab_size, hidden_size)}, got {tuple(lm_head_weight.shape)}."
            )
        with torch.no_grad():
            lm_head.weight.copy_(lm_head_weight)
        lm_head.weight.requires_grad_(False)
    # ``null`` disables depth decay entirely (JetSpec's forward-KL objective
    # weights every position uniformly); algorithms that need decay keep a float.
    gamma_raw = drafter.get("loss_decay_gamma", 7.0)
    loss_decay_gamma = None if gamma_raw is None else float(gamma_raw)
    # Recipe stages whose objective routes through the fp32 TV/KL chunk
    # arithmetic (``objectives._tv_kl``) can compile that arithmetic as an
    # opt-in execution optimization (``drafter.recipe_compile_tv_kl``, default
    # off). The gate is objective-aware inside ``recipe_loss``: stages that do
    # not call ``_tv_kl`` (e.g. ``e2e_tv``) are exempt by construction, so the
    # knob can stay on across a full two-stage run. The win is
    # hardware/workload dependent -- measured on the 48 GB A6000 audit
    # harness; re-measure per hardware rather than assume it transfers.
    recipe_compile_tv_kl = bool(drafter.get("recipe_compile_tv_kl", False))
    module = module_class(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=int(drafter["mask_token_id"]),
        block_size=int(drafter["block_size"]),
        attention_backend=str(drafter.get("attention_backend", "flex_attention")),
        num_anchors=int(drafter.get("num_anchors", 256)),
        loss_decay_gamma=loss_decay_gamma,
        loss_type="dflash",
        # ``None`` (absent) = auto: enable wherever the algorithm declares a
        # row-local scoring tail; an explicit false always disables.
        recipe_row_compaction=drafter.get("recipe_row_compaction"),
        recipe_compile_tv_kl=recipe_compile_tv_kl,
        attention_chunk_size=int(drafter.get("attention_chunk_size", 128)),
        attention_memory_budget_gb=drafter.get("attention_memory_budget_gb"),
        npu_fusion_fallback=bool(drafter.get("npu_fusion_fallback", False)),
        **(module_kwargs or {}),
    )
    return module


def load_checkpoint(path, map_location="cpu", model_class=None):
    """Load a vlm-dflash draft checkpoint (own format; trusted local file)."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from .draft_model import Qwen3DFlashDraftModel

    model_class = model_class or Qwen3DFlashDraftModel
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "vlm-dflash-ckpt-v1":
        raise ValueError(f"Unrecognized checkpoint format in {path}: {checkpoint.get('format')!r}")
    config = Qwen3Config.from_dict(checkpoint["draft_config"])
    config._attn_implementation = model_attention_implementation(
        checkpoint.get("attention_backend", "flex_attention")
    )
    draft = model_class(config)
    draft.load_state_dict(checkpoint["model"])
    draft.eval()
    return draft, checkpoint
