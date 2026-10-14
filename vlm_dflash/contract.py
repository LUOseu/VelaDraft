"""Runtime model contract resolution for VLM-family targets.

Every value the DFlash pipeline depends on (mask token id, MRoPE sections,
draft layer routing, image pixel bounds, special tokens) is resolved from the
target model directory at runtime and validated against the YAML defaults.
Family specifics (token names, rope layout, image-token semantics) come from
the target-family adapter; nothing is hard-coded downstream: a YAML default
that disagrees with the resolved model contract raises
:class:`ContractError`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import now_iso, sha256_file, sha256_json, write_json


class ContractError(RuntimeError):
    """Raised when a YAML default cannot be honored by the target model."""


def contract_defaults_from_config(cfg: dict) -> dict:
    drafter = cfg.get("drafter", {})
    target = cfg.get("target", {})
    # Fusion algorithms (DFlare/DFly) consume an explicit target-layer subset
    # instead of the draft routing; any algorithm block that declares
    # ``context_layer_ids`` names the layers that must exist in the capture.
    algorithm_context = None
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        for block in algorithm_cfg.values():
            if isinstance(block, dict) and block.get("context_layer_ids") is not None:
                algorithm_context = block["context_layer_ids"]
                break
    return {
        "mask_token_id": drafter.get("mask_token_id"),
        "spatial_rope_sections": drafter.get("spatial_rope_sections"),
        "spatial_rope_interleaved": drafter.get("spatial_rope_interleaved"),
        "draft_num_hidden_layers": drafter.get("num_hidden_layers"),
        "capture_layer_ids": drafter.get("capture_layer_ids"),
        "algorithm_context_layer_ids": algorithm_context,
        "block_size": drafter.get("block_size"),
        "image_min_pixels": target.get("image_min_pixels"),
        "image_max_pixels": target.get("image_max_pixels"),
    }


def ensure_contract(cfg: dict, run_dir):
    """Load ``<run>/contract.json`` or resolve it once, then validate the YAML."""
    from pathlib import Path

    from .utils import read_json

    run_dir = Path(run_dir)
    path = run_dir / "contract.json"
    if path.exists():
        contract = read_json(path)
        if contract["target"]["model_path"] != str(Path(cfg["target"]["model"]).resolve()):
            frozen = contract["target"]["model_path"]
            raise ContractError(
                f"Run directory was created for target {frozen} but config points at "
                f"{cfg['target']['model']}; use a new run directory when switching targets."
            )
    else:
        contract = resolve_contract(cfg["target"]["model"], contract_defaults_from_config(cfg), out_path=path)
    validate_yaml_against_contract(cfg, contract, "config")
    return contract


_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
]
_PROCESSOR_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "chat_template.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
]
_WEIGHT_SHARD_GLOBS = ("model.safetensors", "model-*.safetensors")


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Absolute target hidden-state indices routed into the draft, DFlash style.

    Mirrors the reference ``build_target_layer_ids``: start at layer 1 and
    uniformly cover up to ``num_target_layers - 3`` (the last three layers are
    left out because their residual stream is closest to the final logits).
    """
    if num_target_layers < 4:
        raise ContractError(f"Target text stack is too shallow for DFlash routing: {num_target_layers} layers.")
    if num_draft_layers < 1:
        raise ContractError("draft_num_hidden_layers must be >= 1.")
    start, end = 1, num_target_layers - 3
    if num_draft_layers == 1:
        return [start]
    step = (end - start) / (num_draft_layers - 1)
    return [int(start + round(i * step)) for i in range(num_draft_layers)]


def contract_capture_layer_ids(contract: dict) -> list[int]:
    """The captured layer list of a contract or feature manifest (ordered).

    ``capture_layer_ids`` records which target layers a store actually holds and
    in which order. Contracts/stores written before the field existed captured
    exactly the draft routing (the old code's single layout), so falling back to
    ``draft_layer_ids`` recovers their true layout rather than guessing one.
    """
    layers = contract.get("layers") or {}
    raw = layers.get("capture_layer_ids")
    if raw is None:
        raw = layers["draft_layer_ids"]
    return [int(layer) for layer in raw]


def validate_capture_layer_ids(capture_layer_ids, num_target_layers: int, where: str) -> list[int]:
    """Validate an explicit capture list: unique, strictly increasing, in range."""
    ids = [int(layer) for layer in capture_layer_ids]
    if not ids:
        raise ContractError(f"{where}: capture_layer_ids must not be empty.")
    if ids != sorted(set(ids)):
        raise ContractError(
            f"{where}: capture_layer_ids must be unique and strictly increasing "
            f"(capture order is the layout order), got {ids}."
        )
    depth = int(num_target_layers)
    out_of_range = [layer for layer in ids if layer < 0 or layer >= depth]
    if out_of_range:
        raise ContractError(
            f"{where}: capture_layer_ids {out_of_range} outside the target's {depth} text layers."
        )
    return ids


def _file_fingerprint(model_dir: Path, names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        path = model_dir / name
        out[name] = sha256_file(path) if path.exists() else None
    return out


def _weight_shard_signature(model_dir: Path) -> dict[str, int]:
    sig: dict[str, int] = {}
    for pattern in _WEIGHT_SHARD_GLOBS:
        for path in sorted(model_dir.glob(pattern)):
            if path.is_file():
                sig[path.name] = path.stat().st_size
    return sig


def resolve_contract(
    model_path: str | os.PathLike,
    defaults: dict[str, Any] | None = None,
    out_path: str | os.PathLike | None = None,
) -> dict:
    """Resolve and validate the runtime contract for a local target model.

    Args:
        model_path: Local directory of a supported Qwen-VL-family checkpoint.
        defaults: YAML defaults to validate against. Supported keys:
            ``mask_token_id``, ``spatial_rope_sections``,
            ``spatial_rope_interleaved``, ``draft_num_hidden_layers``,
            ``block_size``, ``image_min_pixels``, ``image_max_pixels``.
        out_path: When given, the contract JSON is written there.

    Returns:
        Contract dictionary (see the module docstring of the repository README
        for the schema); every entry carries a ``source`` marker.
    """
    defaults = dict(defaults or {})
    model_dir = Path(model_path)
    if not (model_dir / "config.json").exists():
        raise ContractError(f"Target model directory has no config.json: {model_dir}")

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)

    from .adapters import get_adapter

    adapter = get_adapter(str(getattr(config, "model_type", "")))  # fail loud on unsupported families
    model_type = str(config.model_type)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)

    text_config = getattr(config, "text_config", None) or config
    vocab_size = int(getattr(text_config, "vocab_size"))
    hidden_size = int(getattr(text_config, "hidden_size"))
    num_target_layers = int(getattr(text_config, "num_hidden_layers"))
    head_dim = int(getattr(text_config, "head_dim", hidden_size // int(text_config.num_attention_heads)))

    # ---- special tokens -------------------------------------------------
    # Family-specific resolution (token names + config cross-checks) lives in
    # the adapter; only ``image_token_id`` is consumed downstream.
    tokens = adapter.resolve_tokens(tokenizer, config, text_config)

    # ---- mask token -----------------------------------------------------
    mask_token_id = defaults.get("mask_token_id")
    if mask_token_id is None:
        raise ContractError("defaults.mask_token_id is required (no implicit default).")
    mask_token_id = int(mask_token_id)
    if not 0 <= mask_token_id < vocab_size:
        raise ContractError(f"mask_token_id {mask_token_id} outside embedding range [0, {vocab_size}).")
    mask_token_str = tokenizer.convert_ids_to_tokens(mask_token_id)
    mask_policy = adapter.review_mask_token(mask_token_str)
    for name, value in tokens.items():
        if value is not None and mask_token_id == value:
            raise ContractError(f"mask_token_id collides with {name}={value}.")

    # ---- rope -----------------------------------------------------------
    # Family-specific: MRoPE families validate ``mrope_section`` against
    # head_dim (and the YAML defaults); 1-D families report mode="1d" with
    # sections=None. transformers >=5 moved rope settings onto the text
    # sub-config; adapters search all known locations.
    rope_info = adapter.resolve_rope(config, text_config, head_dim, defaults)
    sections = rope_info["sections"]

    # ---- draft layer routing -------------------------------------------
    draft_layers = int(defaults.get("draft_num_hidden_layers", 3))
    draft_layer_ids = build_target_layer_ids(num_target_layers, draft_layers)
    capture_default = defaults.get("capture_layer_ids")
    if capture_default is None:
        capture_layer_ids = list(draft_layer_ids)
        capture_source = "default: the draft layer routing (capture == draft_layer_ids)"
    else:
        capture_layer_ids = validate_capture_layer_ids(
            capture_default, num_target_layers, "drafter.capture_layer_ids"
        )
        capture_source = "explicit drafter.capture_layer_ids"
    # Every layer the *configured* algorithm consumes must be in the capture:
    # the draft routing for DFlash-family, or the explicit context subset the
    # fusion algorithms (DFlare/DFly) declare.
    consumed_default = defaults.get("algorithm_context_layer_ids")
    if consumed_default is None:
        consumed_layers = list(draft_layer_ids)
        consumed_source = f"draft routing {consumed_layers}"
    else:
        consumed_layers = [int(layer) for layer in consumed_default]
        consumed_source = f"algorithm context_layer_ids {consumed_layers}"
    missing_consumed = [layer for layer in consumed_layers if layer not in capture_layer_ids]
    if missing_consumed:
        raise ContractError(
            f"drafter.capture_layer_ids {capture_layer_ids} does not include layers "
            f"{missing_consumed} that the configured algorithm consumes "
            f"({consumed_source}); the capture must be a superset of every consumed layer."
        )
    block_size = int(defaults.get("block_size", 8))
    if block_size < 2:
        raise ContractError(f"block_size must be >= 2, got {block_size}.")

    # ---- image pixel bounds --------------------------------------------
    try:
        bounds = adapter.processor_pixel_bounds(processor, model_dir)
    except RuntimeError as error:
        raise ContractError(str(error)) from error
    proc_min, proc_max = int(bounds["min_pixels"]), int(bounds["max_pixels"])
    min_pixels = int(defaults.get("image_min_pixels") or proc_min)
    max_pixels = int(defaults.get("image_max_pixels") or proc_max)
    if not 0 < min_pixels <= max_pixels:
        raise ContractError(f"Invalid image pixel bounds: min={min_pixels} max={max_pixels}.")
    grid_tokens = adapter.info.image_token_mode == "grid"
    if grid_tokens:
        image_processor = getattr(processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 14))
        merge_size = int(getattr(image_processor, "merge_size", 2))
        if min_pixels % (patch_size * patch_size) != 0 or max_pixels % (patch_size * patch_size) != 0:
            raise ContractError(
                f"Image pixel bounds must be multiples of patch_size^2={patch_size * patch_size} "
                f"(got min={min_pixels}, max={max_pixels})."
            )
        tokens_per_patch = (patch_size * merge_size) ** 2
        image_token_bounds = {
            "min_tokens": min_pixels // tokens_per_patch,
            "max_tokens": max_pixels // tokens_per_patch,
        }
        sequence_length_per_tile = None
    else:
        # Tile-encoded families: no patch grid; token count = tiles x fixed length.
        patch_size = merge_size = tokens_per_patch = None
        image_token_bounds = {"min_tokens": None, "max_tokens": None}
        sequence_length_per_tile = int(getattr(processor, "image_seq_len", 0) or 0) or None
        if sequence_length_per_tile is None:
            raise ContractError(
                "Tile-encoded family without processor.image_seq_len; cannot derive image-token counts."
            )

    # ---- fingerprints ---------------------------------------------------
    model_files = _file_fingerprint(model_dir, _MODEL_FILES)
    processor_files = _file_fingerprint(model_dir, _PROCESSOR_FILES)
    shard_sig = _weight_shard_signature(model_dir)
    model_fingerprint = sha256_json({"files": model_files, "weights": shard_sig})
    processor_fingerprint = sha256_json({"files": processor_files})

    contract = {
        "contract_version": 1,
        "resolved_at": now_iso(),
        "target": {
            "model_path": str(model_dir.resolve()),
            "model_type": model_type,
            "model_fingerprint": model_fingerprint,
            "processor_fingerprint": processor_fingerprint,
            "fingerprint_source": "sha256 of config/generation/index + weight shard sizes; processor files",
            "model_files": model_files,
            "processor_files": processor_files,
            "weight_shards": shard_sig,
        },
        "text": {
            "num_target_layers": num_target_layers,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "source": "target config.text_config",
        },
        "mask": {
            "token_id": mask_token_id,
            "token_str": mask_token_str,
            "policy": mask_policy,
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "source": "config default validated against tokenizer",
        },
        "rope": {
            "type": rope_info["type"],
            "mode": rope_info["mode"],
            "sections": sections,
            "interleaved": rope_info["interleaved"],
            "source": rope_info["source"],
        },
        "layers": {
            "num_target_layers": num_target_layers,
            "num_draft_layers": draft_layers,
            "draft_layer_ids": draft_layer_ids,
            "capture_layer_ids": capture_layer_ids,
            "capture_layer_ids_source": capture_source,
            "block_size": block_size,
            "source": (
                "config default (start=1, end=num_target_layers-3, uniform) validated against target "
                "depth; capture_layer_ids is the ordered layer list every feature store holds"
            ),
        },
        "image": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "processor_min_pixels": proc_min,
            "processor_max_pixels": proc_max,
            "processor_bounds_source": bounds["source"],
            "token_count_mode": adapter.info.image_token_mode,
            "patch_size": patch_size,
            "merge_size": merge_size,
            "tokens_per_patch": tokens_per_patch,
            "min_tokens": image_token_bounds["min_tokens"],
            "max_tokens": image_token_bounds["max_tokens"],
            "sequence_length_per_tile": sequence_length_per_tile,
            "source": "config override validated against processor; processor defaults recorded",
        },
        "tokens": {**tokens, "source": "tokenizer, cross-checked against target config"},
        "head": {
            "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", True)),
            "source": "target config.json (tie_word_embeddings; absent = True)",
        },
    }
    if out_path is not None:
        write_json(out_path, contract)
    return contract


def validate_yaml_against_contract(cfg: dict, contract: dict, where: str = "config") -> None:
    """Raise when YAML defaults that were used to build the contract drift.

    The capture/train/eval scripts call this so a config edited after capture
    cannot silently train against inconsistent defaults.
    """
    def dotted_get(node, dotted):
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    checks = [
        ("drafter.mask_token_id", contract["mask"]["token_id"]),
        ("drafter.block_size", contract["layers"]["block_size"]),
        ("drafter.num_hidden_layers", contract["layers"]["num_draft_layers"]),
    ]
    for dotted, expected in checks:
        value = dotted_get(cfg, dotted)
        if value != expected:
            raise ContractError(f"{where}: {dotted}={value!r} disagrees with frozen contract value {expected!r}.")
    capture_cfg = (cfg.get("drafter", {}) or {}).get("capture_layer_ids")
    if capture_cfg is not None:
        expected_capture = contract_capture_layer_ids(contract)
        actual_capture = validate_capture_layer_ids(capture_cfg, contract["layers"]["num_target_layers"], where)
        if actual_capture != expected_capture:
            raise ContractError(
                f"{where}: drafter.capture_layer_ids {actual_capture} disagree with frozen contract "
                f"{expected_capture}."
            )
    sections = [int(s) for s in (cfg.get("drafter", {}).get("spatial_rope_sections") or [])]
    contract_sections = contract["rope"].get("sections")
    if contract_sections is None:
        if sections:
            raise ContractError(
                f"{where}: drafter.spatial_rope_sections {sections} set but the target family has no "
                "mrope_section (1-D positions)."
            )
    elif sections and sections != [int(s) for s in contract_sections]:
        raise ContractError(
            f"{where}: drafter.spatial_rope_sections {sections} disagree with contract {contract_sections}."
        )
    interleaved = cfg.get("drafter", {}).get("spatial_rope_interleaved")
    if interleaved is not None and bool(interleaved) != bool(contract["rope"].get("interleaved", False)):
        raise ContractError(
            f"{where}: drafter.spatial_rope_interleaved {bool(interleaved)} disagrees with contract "
            f"{bool(contract['rope'].get('interleaved', False))}."
        )
    if contract["rope"].get("mode") == "1d" and (cfg.get("drafter", {}) or {}).get("spatial_rope_enabled") is not False:
        raise ContractError(
            f"{where}: drafter.spatial_rope_enabled must be false for a 1-D-position family "
            "(the target has no mrope_section)."
        )
    for key in ("image_min_pixels", "image_max_pixels"):
        value = cfg.get("target", {}).get(key)
        expected = contract["image"]["min_pixels" if key.endswith("min_pixels") else "max_pixels"]
        if value is not None and int(value) != int(expected):
            raise ContractError(f"{where}: target.{key}={value} disagrees with contract {expected}.")
