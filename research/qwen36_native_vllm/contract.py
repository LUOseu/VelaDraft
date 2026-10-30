"""Frozen Qwen3.6-35B-A3B draft contract shared by every method adapter.

CPU only: imports Torch-free values so exporters and preflights can run on a
login node. Values come from the finished 400K training checkpoints
(<task>/runs/<method>/train/ckpt_final.pt).
"""
from __future__ import annotations

from typing import Any

MARKER = "qwen36_spatial_adapter_v1"
TARGET_MODEL_TYPE = "qwen3_5_moe"
TARGET_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TARGET_NUM_LAYERS = 40
TARGET_STOP_TOKEN_IDS = [248046, 248044]  # generation_config.eos_token_id: <|im_end|>, <|endoftext|>
TARGET_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
IMAGE_MIN_PIXELS = 65536
IMAGE_MAX_PIXELS = 1048576

BLOCK_SIZE = 8
SPECULATIVE_TOKENS = BLOCK_SIZE - 1
MASK_TOKEN_ID = 248077
TARGET_LAYER_IDS = [1, 19, 37]
MROPE_SECTION = [11, 11, 10]
ROPE_THETA = 10000000.0
PARTIAL_ROTARY_FACTOR = 0.25

DRAFT_FIELDS = {
    "hidden_size": 2048, "head_dim": 256, "num_hidden_layers": 3,
    "num_attention_heads": 16, "num_key_value_heads": 2,
    "vocab_size": 248320, "intermediate_size": 6144,
}
DFLASH_FIELDS = {
    "mask_token_id": MASK_TOKEN_ID, "target_layer_ids": TARGET_LAYER_IDS,
    "spatial_rope_enabled": True, "spatial_rope_mode": "replace",
    "spatial_rope_sections": MROPE_SECTION, "spatial_rope_interleaved": True,
}
SUPPORTED_TENSOR_PARALLEL = (1, 2)


def as_dict(config: Any) -> dict:
    return config if isinstance(config, dict) else config.to_dict()


def validate_draft_config(config: Any, *, require_marker: bool = True) -> dict:
    """Fail closed unless the config is the frozen Qwen3.6 3-layer draft."""
    cfg = as_dict(config)
    for field, value in DRAFT_FIELDS.items():
        if cfg.get(field) != value:
            raise ValueError(f"Frozen draft requires {field}={value}, got {cfg.get(field)!r}")
    if cfg.get("target_hidden_size", cfg["hidden_size"]) != 2048:
        raise ValueError("Frozen draft requires target_hidden_size 2048")
    if cfg.get("block_size") != BLOCK_SIZE or cfg.get("attention_bias", False):
        raise ValueError("Frozen draft requires block_size=8 and attention_bias=False")
    dflash = cfg.get("dflash_config", {})
    for field, value in DFLASH_FIELDS.items():
        if dflash.get(field) != value:
            raise ValueError(f"Frozen draft requires dflash_config.{field}={value!r}, got {dflash.get(field)!r}")
    rope = cfg.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default" or rope.get("rope_theta") != ROPE_THETA:
        raise ValueError("Frozen draft requires default frequencies with theta=1e7")
    if rope.get("partial_rotary_factor") != PARTIAL_ROTARY_FACTOR:
        raise ValueError("Frozen draft requires partial_rotary_factor=0.25 (rotary dim 64)")
    if require_marker:
        if dflash.get(MARKER) is not True:
            raise ValueError(f"Exported config must explicitly enable dflash_config.{MARKER}")
        if rope.get("mrope_section") != MROPE_SECTION or rope.get("mrope_interleaved") is not True:
            raise ValueError("Exported config must select official interleaved MRotaryEmbedding")
    return cfg


def validate_launch(vllm_config: Any) -> None:
    """Engine-level checks shared by every adapter (called in each worker)."""
    from vllm.config import CompilationMode, CUDAGraphMode
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size not in SUPPORTED_TENSOR_PARALLEL or (pc.pipeline_parallel_size, pc.data_parallel_size) != (1, 1):
        raise ValueError("Qwen3.6 adapter supports TP 1 or 2 with PP=DP=1")
    if vllm_config.scheduler_config.max_num_seqs != 1:
        raise ValueError("Qwen3.6 adapter requires max_num_seqs=1")
    if vllm_config.model_config.hf_config.model_type != TARGET_MODEL_TYPE:
        raise ValueError("Qwen3.6 adapter requires a qwen3_5_moe target")
    spec = vllm_config.speculative_config
    if spec.num_speculative_tokens != SPECULATIVE_TOKENS:
        raise ValueError("Frozen block=8 requires num_speculative_tokens=7")
    if not spec.draft_model_config.uses_mrope:
        raise ValueError("Draft model config did not enable official MRoPE")
    if vllm_config.model_config.enforce_eager or spec.enforce_eager:
        raise ValueError("Graph adapter requires enforce_eager=False for target and draft")
    comp = vllm_config.compilation_config
    if comp.mode != CompilationMode.VLLM_COMPILE:
        raise ValueError("Graph adapter requires the official VLLM_COMPILE path")
    if comp.cudagraph_mode.mixed_mode() not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        raise ValueError("Graph adapter requires a mixed graph mode enabling draft PIECEWISE graphs")
    sizes = comp.cudagraph_capture_sizes
    if sizes is not None and BLOCK_SIZE not in sizes:
        raise ValueError("Graph adapter requires capture size eight")
