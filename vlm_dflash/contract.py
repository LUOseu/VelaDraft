"""Runtime model contract resolution for the Qwen2.5-VL target.

Every value the DFlash pipeline depends on (mask token id, MRoPE sections,
draft layer routing, image pixel bounds, special tokens) is resolved from the
target model directory at runtime and validated against the YAML defaults.
Nothing is hard-coded downstream: a YAML default that disagrees with the
resolved model contract raises :class:`ContractError`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .utils import now_iso, sha256_file, sha256_json, write_json


class ContractError(RuntimeError):
    """Raised when a YAML default cannot be honored by the target model."""


def contract_defaults_from_config(cfg: dict) -> dict:
    drafter = cfg.get("drafter", {})
    target = cfg.get("target", {})
    return {
        "mask_token_id": drafter.get("mask_token_id"),
        "spatial_rope_sections": drafter.get("spatial_rope_sections"),
        "draft_num_hidden_layers": drafter.get("num_hidden_layers"),
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
        model_path: Local directory of a Qwen2.5-VL-style checkpoint.
        defaults: YAML defaults to validate against. Supported keys:
            ``mask_token_id``, ``spatial_rope_sections``,
            ``draft_num_hidden_layers``, ``block_size``,
            ``image_min_pixels``, ``image_max_pixels``.
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
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)

    text_config = getattr(config, "text_config", None) or config
    vocab_size = int(getattr(text_config, "vocab_size"))
    hidden_size = int(getattr(text_config, "hidden_size"))
    num_target_layers = int(getattr(text_config, "num_hidden_layers"))
    head_dim = int(getattr(text_config, "head_dim", hidden_size // int(text_config.num_attention_heads)))

    # ---- special tokens -------------------------------------------------
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
    for cfg_key, cfg_value in (
        ("image_token_id", getattr(text_config, "image_token_id", None)),
        ("vision_start_token_id", getattr(text_config, "vision_start_token_id", None)),
        ("vision_end_token_id", getattr(text_config, "vision_end_token_id", None)),
    ):
        if cfg_value is not None and int(cfg_value) != tokens[cfg_key]:
            raise ContractError(
                f"Token config disagrees with tokenizer for {cfg_key}: config={cfg_value} tokenizer={tokens[cfg_key]}"
            )

    # ---- mask token -----------------------------------------------------
    mask_token_id = defaults.get("mask_token_id")
    if mask_token_id is None:
        raise ContractError("defaults.mask_token_id is required (no implicit default).")
    mask_token_id = int(mask_token_id)
    if not 0 <= mask_token_id < vocab_size:
        raise ContractError(f"mask_token_id {mask_token_id} outside embedding range [0, {vocab_size}).")
    mask_token_str = tokenizer.convert_ids_to_tokens(mask_token_id)
    if mask_token_str is not None:
        raise ContractError(
            f"mask_token_id {mask_token_id} resolves to tokenizer token {mask_token_str!r}; "
            "the mask must live on a reserved embedding row that the tokenizer can never emit. "
            "Pick an id >= tokenizer.vocab_size."
        )
    for name, value in tokens.items():
        if mask_token_id == value:
            raise ContractError(f"mask_token_id collides with {name}={value}.")

    # ---- MRoPE sections -------------------------------------------------
    # transformers >=5 moved rope settings onto the text sub-config
    # (``text_config.rope_parameters``); older layouts keep ``rope_scaling`` at
    # the top level. Search all known locations and record where it came from.
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

    # ---- draft layer routing -------------------------------------------
    draft_layers = int(defaults.get("draft_num_hidden_layers", 3))
    draft_layer_ids = build_target_layer_ids(num_target_layers, draft_layers)
    block_size = int(defaults.get("block_size", 8))
    if block_size < 2:
        raise ContractError(f"block_size must be >= 2, got {block_size}.")

    # ---- image pixel bounds --------------------------------------------
    image_processor = getattr(processor, "image_processor", None)
    proc_min = getattr(image_processor, "min_pixels", None)
    proc_max = getattr(image_processor, "max_pixels", None)
    if proc_min is None or proc_max is None:
        pre = json.loads((model_dir / "preprocessor_config.json").read_text())
        proc_min = proc_min if proc_min is not None else pre.get("min_pixels")
        proc_max = proc_max if proc_max is not None else pre.get("max_pixels")
    if proc_min is None or proc_max is None:
        raise ContractError("Could not resolve processor min_pixels/max_pixels.")
    proc_min, proc_max = int(proc_min), int(proc_max)
    min_pixels = int(defaults.get("image_min_pixels") or proc_min)
    max_pixels = int(defaults.get("image_max_pixels") or proc_max)
    if not 0 < min_pixels <= max_pixels:
        raise ContractError(f"Invalid image pixel bounds: min={min_pixels} max={max_pixels}.")
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
            "model_type": str(getattr(config, "model_type", "unknown")),
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
            "policy": "reserved embedding row outside tokenizer vocabulary",
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "source": "config default validated against tokenizer",
        },
        "rope": {
            "type": rope.get("type") or rope.get("rope_type"),
            "sections": sections,
            "source": f"target {rope_source}.mrope_section",
        },
        "layers": {
            "num_target_layers": num_target_layers,
            "num_draft_layers": draft_layers,
            "draft_layer_ids": draft_layer_ids,
            "block_size": block_size,
            "source": "config default (start=1, end=num_target_layers-3, uniform) validated against target depth",
        },
        "image": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "processor_min_pixels": proc_min,
            "processor_max_pixels": proc_max,
            "patch_size": patch_size,
            "merge_size": merge_size,
            "tokens_per_patch": tokens_per_patch,
            "min_tokens": image_token_bounds["min_tokens"],
            "max_tokens": image_token_bounds["max_tokens"],
            "source": "config override validated against processor; processor defaults recorded",
        },
        "tokens": {**tokens, "source": "tokenizer, cross-checked against target config"},
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
    sections = [int(s) for s in (cfg.get("drafter", {}).get("spatial_rope_sections") or [])]
    if sections and sections != [int(s) for s in contract["rope"]["sections"]]:
        raise ContractError(
            f"{where}: drafter.spatial_rope_sections {sections} disagree with contract {contract['rope']['sections']}."
        )
    for key in ("image_min_pixels", "image_max_pixels"):
        value = cfg.get("target", {}).get(key)
        expected = contract["image"]["min_pixels" if key.endswith("min_pixels") else "max_pixels"]
        if value is not None and int(value) != int(expected):
            raise ContractError(f"{where}: target.{key}={value} disagrees with contract {expected}.")
