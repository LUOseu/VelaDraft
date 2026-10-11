"""Qwen2.5-VL adapter (the v0.1 reference family)."""

from __future__ import annotations

from .base import AdapterInfo, VLMAdapter


class Qwen25VLAdapter(VLMAdapter):
    info = AdapterInfo(
        name="qwen25_vl",
        family="qwen-vl",
        model_types=("qwen2_5_vl",),
        model_class="Qwen2_5_VLForConditionalGeneration",
        notes=(
            "MRoPE sections come from text_config.rope_parameters.mrope_section "
            "(e.g. [16, 24, 24] for Qwen2.5-VL); transformers >= 5 requires explicit "
            "mm_token_type_ids for multimodal forwards."
        ),
    )
