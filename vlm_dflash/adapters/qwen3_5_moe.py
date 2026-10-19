"""Qwen3.5/3.6-MoE adapter (hybrid Gated DeltaNet + full-attention MoE target).

Same processor and staged-capture interface as Qwen3-VL. The family's partial
rotary, patch size and missing dense ``intermediate_size`` are resolved from
the target config (see ``docs/REFERENCE.md``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

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

    def load_model(
        self,
        model_path: str,
        *,
        dtype,
        attn_implementation: str,
        local_files_only: bool,
        backend_hint: str | None = None,
        **load_kwargs,
    ):
        """Load the target, selecting the Gated DeltaNet execution path per backend.

        Upstream transformers selects FLA/causal-conv1d solely by package
        availability, before the model is moved to its device. On NPU the FLA
        Triton kernels are not assumed to exist and the RMSNorm-gated module's
        constructor calls ``torch.cuda.current_device()``. We therefore force the
        tracked, upstream PyTorch reference functions while constructing the
        model on NPU. CUDA keeps the fast path; CPU can opt in with
        ``VLM_DFLASH_QWEN35_TORCH_REFERENCE=1``.
        """
        force_reference = backend_hint == "npu" or os.environ.get(
            "VLM_DFLASH_QWEN35_TORCH_REFERENCE", ""
        ).lower() in ("1", "true", "yes")
        if not force_reference:
            return super().load_model(
                model_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
                local_files_only=local_files_only,
                backend_hint=backend_hint,
                **load_kwargs,
            )
        with _torch_reference_gated_delta():
            model = super().load_model(
                model_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
                local_files_only=local_files_only,
                backend_hint=backend_hint,
                **load_kwargs,
            )
        setattr(model, "_vlm_dflash_gated_delta_backend", "torch_reference")
        return model


@contextmanager
def _torch_reference_gated_delta():
    """Temporarily make upstream Qwen3.5-MoE use its tracked PyTorch fallbacks.

    Instances capture the selected functions at construction time, so restoring
    the module globals afterwards does not alter the loaded model. No file in
    ``site-packages`` is modified.
    """
    module = __import__(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        fromlist=["modeling_qwen3_5_moe"],
    )
    names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
        "is_fast_path_available",
    )
    saved = {name: getattr(module, name, None) for name in names}
    try:
        for name in names[:-1]:
            setattr(module, name, None)
        module.is_fast_path_available = False
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


__all__ = ["Qwen35MoeAdapter"]
