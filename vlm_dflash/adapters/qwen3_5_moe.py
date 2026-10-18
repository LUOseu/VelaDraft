"""Qwen3.5/3.6-MoE adapter (hybrid Gated DeltaNet + full-attention MoE target).

Same processor and staged-capture interface as Qwen3-VL. The family's partial
rotary, patch size and missing dense ``intermediate_size`` are resolved from
the target config (see ``docs/REFERENCE.md``).
"""

from __future__ import annotations

from .base import AdapterInfo, VLMAdapter


class Qwen35MoeAdapter(VLMAdapter):
    info = AdapterInfo(
        name="qwen3_5_moe",
        family="qwen-vl",
        model_types=("qwen3_5_moe",),
        model_class="Qwen3_5MoeForConditionalGeneration",
        cache_crop_rollback=False,
        notes=(
            "Hybrid Gated DeltaNet / full-attention MoE target with partial rotary (factor 0.25) and "
            "interleaved MRoPE [11, 11, 10]. Install flash-linear-attention for the GDN chunk kernel; "
            "the torch fallback is too slow for training. Pin image_max_pixels explicitly: the processor "
            "default (16M pixels) allows up to 16384 visual tokens per image."
        ),
    )
