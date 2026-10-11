"""Target-family adapter contract.

Everything model-family specific (model class, processor conventions, MRoPE
position derivation) lives behind this interface so the pipeline stages never
branch on a concrete Qwen-VL generation. The contract/provenance layers stay
generic: adapters only own *how* to talk to a family, never *what* the
pipeline requires from it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch

def _mapping_value(source, key):
    """Read ``key`` from a dict-like or attribute-like source (SizeDict etc.)."""
    if isinstance(source, dict):
        return source.get(key)
    value = getattr(source, key, None)
    if value is not None:
        return value
    try:
        return source[key]
    except Exception:
        return None


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    family: str
    model_types: tuple[str, ...]
    model_class: str
    notes: str = ""


class VLMAdapter:
    """Base adapter; subclasses set :attr:`info` (and may override hooks)."""

    info: AdapterInfo

    # ---- model / processor loading -------------------------------------
    def load_model(self, model_path: str, *, dtype, attn_implementation: str, local_files_only: bool):
        import transformers

        model_class = getattr(transformers, self.info.model_class, None)
        if model_class is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} has no {self.info.model_class}; "
                f"upgrade transformers to use the {self.info.name} adapter."
            )
        return model_class.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )

    def processor_kwargs(self, contract: dict) -> dict:
        """Extra ``AutoProcessor.from_pretrained`` kwargs (e.g. pixel bounds)."""
        return {
            "min_pixels": contract["image"]["min_pixels"],
            "max_pixels": contract["image"]["max_pixels"],
        }

    def processor_pixel_bounds(self, processor, model_dir) -> dict:
        """Resolve the image pixel bounds a processor family actually uses.

        HF processors disagree on the layout, so both layouts are supported and
        the resolved source is recorded in the contract:

        * ``image_processor.min_pixels`` / ``max_pixels`` (Qwen2.5-VL style);
        * ``image_processor.size.shortest_edge`` / ``longest_edge`` (Qwen3-VL
          style, where ``min_pixels`` mirrors ``shortest_edge``).

        ``preprocessor_config.json`` is consulted when the loaded processor does
        not expose either layout.

        Returns:
            ``{"min_pixels": int, "max_pixels": int, "source": str}``.
        """
        import json
        from pathlib import Path

        candidates: list[tuple[str, object]] = []
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            candidates.append(("processor.image_processor", image_processor))
        pre_path = Path(model_dir) / "preprocessor_config.json"
        if pre_path.exists():
            candidates.append((pre_path.name, json.loads(pre_path.read_text(encoding="utf-8"))))
        for name, source in candidates:
            min_pixels, max_pixels = _mapping_value(source, "min_pixels"), _mapping_value(source, "max_pixels")
            if min_pixels is not None and max_pixels is not None:
                return {
                    "min_pixels": int(min_pixels),
                    "max_pixels": int(max_pixels),
                    "source": f"{name}.min_pixels/max_pixels",
                }
        for name, source in candidates:
            size = _mapping_value(source, "size")
            if size is None:
                continue
            shortest, longest = _mapping_value(size, "shortest_edge"), _mapping_value(size, "longest_edge")
            if shortest is not None and longest is not None:
                return {
                    "min_pixels": int(shortest),
                    "max_pixels": int(longest),
                    "source": f"{name}.size.shortest_edge/longest_edge",
                }
        raise RuntimeError(
            "Could not resolve processor min_pixels/max_pixels: neither the loaded image processor nor "
            "preprocessor_config.json exposes 'min_pixels'/'max_pixels' or 'size.{shortest_edge,longest_edge}'."
        )

    def finalize_processor(self, processor, contract: dict) -> None:
        """Post-construction processor tweaks shared by the encoder-decoder path."""
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    # ---- multimodal position ids ---------------------------------------
    def call_position_builder(self, model, **kwargs):
        """Call the target's own ``compute_3d_position_ids`` with only supported kwargs.

        Families differ in the accepted keyword set (for example Qwen2.5-VL takes
        ``second_per_grid_ts`` while Qwen3-VL does not), so the call is filtered
        by the resolved signature instead of assuming one family's API.

        Returns:
            The raw builder output. ``None`` is a legitimate result: the model
            cannot always derive multimodal positions (e.g. on a cached sequence
            without the token-type map), and callers decide the fallback.
        """
        builder = getattr(getattr(model, "model", None), "compute_3d_position_ids", None)
        if not callable(builder):
            raise RuntimeError(
                f"Target model does not expose compute_3d_position_ids; the {self.info.name} "
                "adapter requires it (Qwen2.5-VL / Qwen3-VL families provide it)."
            )
        signature = inspect.signature(builder)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        call_kwargs = {key: value for key, value in kwargs.items() if accepts_var_kw or key in signature.parameters}
        return builder(**call_kwargs)

    def compute_multimodal_positions(
        self,
        model,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
    ) -> torch.Tensor:
        """Call the target's MRoPE position builder and normalize the batch-1 output.

        The capture path is batch-1: 4-D builder outputs keep the spatial axis
        and 3-D outputs are squeezed to ``[3, sequence]``.
        """
        positions = self.call_position_builder(
            model,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            second_per_grid_ts=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        if positions is None:
            raise RuntimeError("compute_3d_position_ids returned None; image/token-type inputs missing.")
        if positions.dim() == 4:
            positions = positions[..., 0, :]
        if positions.dim() == 3:
            positions = positions[:, 0, :]
        return positions

