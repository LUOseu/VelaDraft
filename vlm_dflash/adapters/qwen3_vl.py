"""Qwen3-VL adapter.

Shares the capture/position-id interface with Qwen2.5-VL (the model exposes
``compute_3d_position_ids`` and the processor emits ``mm_token_type_ids``), but
keeps its own registry entry so family quirks stay isolated. The MRoPE sections
are resolved from the target config at runtime (Qwen3-VL defaults to
``[24, 20, 20]``); the YAML default is only a validation target.
"""

from __future__ import annotations

from .base import AdapterInfo, VLMAdapter


class Qwen3VLAdapter(VLMAdapter):
    info = AdapterInfo(
        name="qwen3_vl",
        family="qwen-vl",
        model_types=("qwen3_vl",),
        model_class="Qwen3VLForConditionalGeneration",
        notes=(
            "Same staged-capture interface as Qwen2.5-VL (compute_3d_position_ids + "
            "mm_token_type_ids). Verify the resolved mask token id and mrope sections "
            "against the downloaded checkpoint via `vlm-dflash doctor` before training."
        ),
    )
