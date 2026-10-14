"""SmolVLM2 adapter (first non-Qwen family).

Differences from the Qwen-VL families that this adapter resolves:

* Plain 1-D sequence positions (Llama-style RoPE in the SmolLM2 backbone); no
  ``compute_3d_position_ids`` and no ``mrope_section``. The pipeline's
  ``position3`` feature field is still populated with three identical axes so
  the store schema is unchanged; a 1-D-family draft trains with
  ``spatial_rope_enabled=false`` and never consumes it.
* Images tokenize to a contiguous run of ``<image>`` ids, but the count is not
  grid-derived: each vision tile emits a fixed ``image_seq_len`` (81 for
  SmolVLM2), so a row's count is ``tiles x 81``.
* Processor outputs ``pixel_values``/``pixel_attention_mask`` (no
  ``image_grid_thw`` / ``mm_token_type_ids``).
* The checkpoint reserves explicit never-emitted special rows
  ``<|reserved_special_token_N|>`` used as the mask token (default policy in
  the contract still prefers an id outside the vocabulary; this family's
  embedding table has no extra rows, so a reserved row is the deterministic
  alternative).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ..contract import ContractError
from .base import AdapterInfo, VLMAdapter, locate_image_token_spans

_RESERVED_TOKEN = re.compile(r"^<\|reserved_special_token_\d+\|>$")
_ROW_COL_TOKEN = re.compile(r"^<row_\d+_col_\d+>$")


class SmolVLMAdapter(VLMAdapter):
    info = AdapterInfo(
        name="smolvlm",
        family="smolvlm",
        model_types=("smolvlm",),
        model_class="SmolVLMForConditionalGeneration",
        notes=(
            "1-D positions (no MRoPE): train/eval DFlash drafts for this family with "
            "drafter.spatial_rope_enabled=false. Images are emitted as fixed-size tile runs "
            "(image_seq_len tokens per tile); mask token must be a reserved special row."
        ),
        position_mode="1d",
        image_token_mode="tiles",
        forward_input_keys=("pixel_values", "pixel_attention_mask"),
    )

    # SmolVLM2 vision tiles always decode to 81 <image> tokens (processor_config.json).
    IMAGE_SEQ_LEN = 81

    def processor_kwargs(self, contract: dict) -> dict:
        # The SmolVLM processor scores its own resize/split budget; no pixel bounds are injected.
        return {}

    def processor_pixel_bounds(self, processor, model_dir) -> dict:
        pre_path = Path(model_dir) / "preprocessor_config.json"
        data = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.exists() else {}
        size = data.get("size") or {}
        max_image_size = data.get("max_image_size") or {}
        longest = size.get("longest_edge")
        shortest = max_image_size.get("longest_edge")
        if longest is None or shortest is None:
            raise ContractError(
                "SmolVLM preprocessor_config.json must define size.longest_edge and "
                "max_image_size.longest_edge to resolve image bounds."
            )
        return {
            "min_pixels": int(shortest) ** 2,
            "max_pixels": int(longest) ** 2,
            "source": "preprocessor_config.json size.longest_edge/max_image_size.longest_edge",
        }

    def resolve_tokens(self, tokenizer, config, text_config) -> dict[str, int | None]:
        def tid(token: str, *, required: bool) -> int | None:
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or value < 0:
                if required:
                    raise ContractError(f"Tokenizer does not define required special token {token!r}.")
                return None
            return int(value)

        tokens = {
            "image_token_id": tid("<image>", required=True),
            "fake_token_around_image_id": tid("<fake_token_around_image>", required=True),
            "im_start_token_id": tid("<|im_start|>", required=False),
            "im_end_token_id": tid("<|im_end|>", required=False),
            "endoftext_token_id": tid("<|endoftext|>", required=False),
        }
        cfg_image = getattr(config, "image_token_id", None)
        if cfg_image is not None and int(cfg_image) != tokens["image_token_id"]:
            raise ContractError(
                f"Token config disagrees with tokenizer for image_token_id: "
                f"config={cfg_image} tokenizer={tokens['image_token_id']}"
            )
        return tokens

    def review_mask_token(self, token_str: str | None) -> str:
        # Preferred: ids outside the tokenizer vocabulary (base policy). This family's
        # checkpoint has no spare embedding rows, so a reserved special row is accepted
        # instead - deterministic, present in the checkpoint, and never emitted by the
        # SmolVLM chat template.
        if token_str is None:
            return "reserved embedding row outside tokenizer vocabulary"
        if _RESERVED_TOKEN.match(token_str):
            return f"reserved special-token row {token_str!r} (never emitted by the chat template)"
        raise ContractError(
            f"mask_token_id resolves to tokenizer token {token_str!r}; for SmolVLM pick an id outside the "
            "vocabulary or a <|reserved_special_token_N|> row."
        )

    def resolve_rope(self, config, text_config, head_dim: int, defaults: dict) -> dict:
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
        rope = rope or {}
        default_sections = defaults.get("spatial_rope_sections")
        if default_sections is not None:
            raise ContractError(
                f"YAML spatial_rope_sections {default_sections} set for a family without mrope_section."
            )
        if defaults.get("spatial_rope_interleaved"):
            raise ContractError("YAML spatial_rope_interleaved set for a family without MRoPE.")
        return {
            "type": rope.get("type") or rope.get("rope_type") or "default",
            "mode": "1d",
            "sections": None,
            "interleaved": False,
            "source": (
                f"target {rope_source} (no mrope_section)"
                if rope_source
                else "target config has no rope mapping; plain sequence positions"
            ),
        }

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
        """Three identical axes of 1-D positions (schema-compatible stand-in).

        A 1-D-family draft trains with ``spatial_rope_enabled=false``, so these
        values are never consumed; they keep the feature-store schema unchanged
        and are exactly what an equal-axes MRoPE merge would produce.
        """
        length = int(input_ids.shape[-1])
        row = torch.arange(length, device=input_ids.device, dtype=torch.long)
        return row.unsqueeze(0).expand(3, -1).contiguous()

    def batch_processor_images(self, images: list) -> list:
        # SmolVLM's processor matches the number of images in the text against a
        # nested per-text image list; a flat batch list raises.
        return [[image] for image in images]

    def image_token_spans(self, input_ids, image_token_id: int, *, tokenizer=None) -> list[tuple[int, int]]:
        # One SmolVLM2 image expands (per the processor) to tile runs of
        # ``image_seq_len`` <image> tokens separated by <row_i_col_j>/
        # <global-img>/<fake_token_around_image> markers and newlines; the
        # markers and whitespace bridge the runs back into one image region.
        bridge_ids: tuple[int, ...] = ()
        is_bridge = None
        if tokenizer is not None:
            structure = []
            for token in ("<global-img>", "<fake_token_around_image>"):
                value = tokenizer.convert_tokens_to_ids(token)
                if value is not None and int(value) >= 0:
                    structure.append(int(value))
            for candidate in range(max(0, image_token_id - 64), image_token_id):
                text = tokenizer.convert_ids_to_tokens(candidate)
                if isinstance(text, str) and _ROW_COL_TOKEN.match(text):
                    structure.append(candidate)
            bridge_ids = tuple(structure)

            def is_bridge(token_id: int) -> bool:
                return not tokenizer.decode([int(token_id)]).strip()

        return locate_image_token_spans(
            input_ids, image_token_id, bridge_ids=bridge_ids, is_bridge=is_bridge
        )

    def capture_forward_inputs(self, prepared_list: list[dict], device) -> dict:
        forward_kwargs: dict = {}
        for key in ("pixel_values", "pixel_attention_mask"):
            values = [item["inputs"].get(key) for item in prepared_list]
            present = [value is not None for value in values]
            if any(present) and not all(present):
                raise ValueError(f"Batch capture mixes rows with and without {key}; split the batch.")
            if not all(present):
                continue
            shapes = {tuple(value.shape) for value in values}
            if len(shapes) > 1:
                raise ValueError(
                    f"Batch capture rows disagree on {key} shapes {sorted(shapes)}; "
                    "use one sample per capture group for this family."
                )
            forward_kwargs[key] = torch.cat([value.to(device) for value in values], dim=0)
        return forward_kwargs

    def expected_image_token_counts(self, inputs: dict, *, merge_size: int | None) -> list[int] | None:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return None
        tiles = int(pixel_values.shape[1])
        return [tiles * self.IMAGE_SEQ_LEN]
