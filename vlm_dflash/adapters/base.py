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

from ..contract import ContractError, contract_chat_template_kwargs
from ..target.prompts import CHAT_TEMPLATE_KWARGS_ATTR

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
    # Position semantics: "mrope" families expose ``compute_3d_position_ids``
    # and a ``mrope_section`` in their rope parameters; "1d" families use plain
    # sequence positions for the whole (multimodal) sequence.
    position_mode: str = "mrope"
    # Image-token semantics: "grid" families derive the image-token count from
    # a ``image_grid_thw`` patch grid; "tiles" families emit a fixed number of
    # tokens per vision tile.
    image_token_mode: str = "grid"
    # Processor-output tensors forwarded to the target forward / generation
    # calls (plus attention_mask, which is always forwarded when present).
    forward_input_keys: tuple[str, ...] = ("pixel_values", "image_grid_thw", "mm_token_type_ids")



def locate_image_token_spans(input_ids, image_token_id: int, *, bridge_ids=(), is_bridge=None):
    """Contiguous image-token regions, merged across family bridge tokens.

    Runs of ``image_token_id`` that are separated only by bridge tokens (an
    optional explicit id set plus an optional per-token predicate) belong to
    the same image region. Text tokens end a region, so images separated by
    text stay separate spans.
    """
    matches = (input_ids == image_token_id).nonzero().flatten().tolist()
    if not matches:
        raise ValueError("No image tokens found in the sequence.")
    bridge = set(int(value) for value in bridge_ids)

    def gap_is_bridge(lo: int, hi: int) -> bool:
        if hi < lo:
            return False
        for position in range(lo, hi + 1):
            token = int(input_ids[position])
            if token in bridge:
                continue
            if is_bridge is not None and is_bridge(token):
                continue
            return False
        return True

    spans: list[tuple[int, int]] = []
    start = previous = matches[0]
    for position in matches[1:]:
        if position == previous + 1 or gap_is_bridge(previous + 1, position - 1):
            previous = position
            continue
        spans.append((start, previous))
        start = previous = position
    spans.append((start, previous))
    return spans

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
        """Post-construction processor tweaks shared by the encoder-decoder path.

        Overrides must call ``super()``: this also attaches the contract's
        chat-template kwargs that every prompt render reads.
        """
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        setattr(processor, CHAT_TEMPLATE_KWARGS_ATTR, contract_chat_template_kwargs(contract))

    # ---- special tokens / mask policy ----------------------------------
    def resolve_tokens(self, tokenizer, config, text_config) -> dict[str, int | None]:
        """Family special tokens, validated against the target config.

        The default implements the Qwen-VL set; families override with their own
        token names and cross-checks. Only ``image_token_id`` is consumed by the
        pipeline stages; the rest is recorded in the contract for provenance.
        """

        def tid(token: str) -> int:
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or value < 0:
                raise ContractError(f"Tokenizer does not define required special token {token!r}.")
            return int(value)

        tokens = {
            "image_token_id": tid("<|image_pad|>"),
            "vision_start_token_id": tid("<|vision_start|>"),
            "vision_end_token_id": tid("<|vision_end|>"),
            "im_start_token_id": tid("<|im_start|>"),
            "im_end_token_id": tid("<|im_end|>"),
            "endoftext_token_id": tid("<|endoftext|>"),
        }
        for cfg_key in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
            cfg_value = getattr(text_config, cfg_key, None)
            if cfg_value is None and text_config is not config:
                cfg_value = getattr(config, cfg_key, None)
            if cfg_value is not None and int(cfg_value) != tokens[cfg_key]:
                raise ContractError(
                    f"Token config disagrees with tokenizer for {cfg_key}: "
                    f"config={cfg_value} tokenizer={tokens[cfg_key]}"
                )
        return tokens

    def review_mask_token(self, token_str: str | None) -> str:
        """Accept or reject the resolved mask-token string; returns the policy note.

        Default policy (MRoPE families): the mask must resolve to *no* tokenizer
        token at all, i.e. live on an embedding row the tokenizer can never
        emit. Families may override when their checkpoints reserve explicit
        never-emitted special rows instead.
        """
        if token_str is not None:
            raise ContractError(
                f"mask_token_id resolves to tokenizer token {token_str!r}; "
                "the mask must live on a reserved embedding row that the tokenizer can never emit. "
                "Pick an id >= tokenizer.vocab_size."
            )
        return "reserved embedding row outside tokenizer vocabulary"

    # ---- rope -----------------------------------------------------------
    def resolve_rope(self, config, text_config, head_dim: int, defaults: dict) -> dict:
        """Resolve the target's rope/position configuration.

        Returns a dict with ``type``, ``mode`` (``"mrope"`` or ``"1d"``),
        ``sections`` (``None`` for 1-D families), ``interleaved`` and ``source``.
        The default implements the Qwen MRoPE contract; 1-D families override.
        """
        rope, rope_source = None, None
        for holder, holder_name, attr in (
            (text_config, "text_config", "rope_parameters"),
            (text_config, "text_config", "rope_scaling"),
            (config, "config", "rope_scaling"),
            (config, "config", "rope_parameters"),
        ):
            value = getattr(holder, attr, None)
            if isinstance(value, dict) and value:
                rope, rope_source = value, f"{holder_name}.{attr}"
                break
        if not isinstance(rope, dict):
            raise ContractError("Target config has no rope_scaling/rope_parameters mapping.")
        sections = rope.get("mrope_section")
        if sections is None:
            raise ContractError("Target config rope scaling has no mrope_section.")
        sections = [int(s) for s in sections]
        default_sections = defaults.get("spatial_rope_sections")
        if default_sections is not None and [int(s) for s in default_sections] != sections:
            raise ContractError(
                f"YAML spatial_rope_sections {default_sections} disagree with target mrope_section {sections}."
            )
        if 2 * sum(sections) != head_dim:
            raise ContractError(
                f"mrope_section {sections} doubled sum must equal head_dim {head_dim} for spatial-rope DFlash."
            )
        interleaved = bool(rope.get("mrope_interleaved", False))
        default_interleaved = defaults.get("spatial_rope_interleaved")
        if default_interleaved is not None and bool(default_interleaved) != interleaved:
            raise ContractError(
                f"YAML spatial_rope_interleaved {bool(default_interleaved)} disagrees with target "
                f"mrope_interleaved {interleaved}."
            )
        if interleaved:
            half = head_dim // 2
            for axis in (1, 2):
                if 3 * sections[axis] > half:
                    raise ContractError(
                        f"Interleaved MRoPE needs 3*mrope_section[axis] <= head_dim/2; axis {axis} has "
                        f"3*{sections[axis]} > {half}."
                    )
        return {
            "type": rope.get("type") or rope.get("rope_type"),
            "mode": "mrope",
            "sections": sections,
            "interleaved": interleaved,
            "source": f"target {rope_source}.mrope_section + mrope_interleaved (absent = False)",
        }

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

    def image_token_spans(self, input_ids, image_token_id: int, *, tokenizer=None) -> list[tuple[int, int]]:
        """Contiguous image regions in one token row (the family's span rule).

        The default treats every run of ``image_token_id`` as its own region
        (Qwen-VL lays each image out as one pure run). Families whose processors
        interleave structure tokens between tile runs override this; they may
        use ``tokenizer`` to recognise their own bridge tokens.
        """
        return locate_image_token_spans(input_ids, image_token_id)

    def batch_processor_images(self, images: list) -> list:
        """Image argument for a multi-row ``processor(text=[...], images=...)`` call.

        Default passes the flat per-row image list (Qwen-VL accepts it);
        families whose processors demand one nested list per text override this.
        """
        return images

    # ---- forward plumbing ----------------------------------------------
    def forward_kwargs_from_inputs(self, inputs: dict) -> dict:
        """Pick the family's tensors out of a processor-output dict (eval path)."""
        kwargs = {
            key: inputs[key]
            for key in self.info.forward_input_keys
            if inputs.get(key) is not None
        }
        if inputs.get("attention_mask") is not None:
            kwargs["attention_mask"] = inputs["attention_mask"]
        return kwargs

    def capture_forward_inputs(self, prepared_list: list[dict], device) -> dict:
        """Family image tensors for a batched capture forward (rows already padded).

        The default implements the Qwen-VL layout (grids, video tensors); other
        families override with their own processor outputs. The token-type map
        stays in the capture stage because the per-row position derivation
        consumes it too.
        """
        forward_kwargs: dict = {}
        pixel_values = [item["inputs"].get("pixel_values") for item in prepared_list]
        grids = [item["inputs"].get("image_grid_thw") for item in prepared_list]
        for key, values in (("pixel_values", pixel_values), ("image_grid_thw", grids)):
            present = [value is not None for value in values]
            if any(present) and not all(present):
                raise ValueError(f"Batch capture mixes rows with and without {key}; split the batch.")
            if all(present):
                forward_kwargs[key] = torch.cat(values, dim=0).to(device)
        for key in ("pixel_values_videos", "video_grid_thw"):
            values = [item["inputs"].get(key) for item in prepared_list]
            if all(value is not None for value in values):
                forward_kwargs[key] = torch.cat(values, dim=0).to(device)
        return forward_kwargs

    def expected_image_token_counts(self, inputs: dict, *, merge_size: int | None) -> list[int] | None:
        """Per-image expected image-token counts for one processor row (or None).

        Grid families derive the count from the patch grid; tile families
        override with their fixed tokens-per-tile convention. ``None`` means the
        family cannot predict the count and consumers skip the check.
        """
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return None
        if merge_size is None:
            raise ValueError("merge_size is required to derive image-token counts from a patch grid.")
        return [int(row.prod().item()) // (merge_size * merge_size) for row in grid]

