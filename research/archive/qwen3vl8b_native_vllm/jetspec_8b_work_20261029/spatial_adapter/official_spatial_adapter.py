"""Process-local vLLM 0.22.1 adapter for the frozen Qwen2.5-VL DFlash draft.

No installed files are changed and no kernels are supplied. The official
DFlash expansion kernel still owns token IDs, physical cache slots, rejection
handling and sampling indices. Three-axis positions are used only for RoPE.

Call install() before constructing vllm.LLM (in every worker process). Initial
integration scope is TP=PP=DP=1, max_num_seqs=1, enforce_eager=True, block=8.
CPU contract helpers import only Torch; vLLM imports are lazy inside install().
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

MARKER = "official_spatial_adapter_v1"
_INSTALLED = False


def _as_dict(config: Any) -> dict:
    return config if isinstance(config, dict) else config.to_dict()


def validate_draft_config(config: Any, *, require_marker: bool = True) -> dict:
    cfg = _as_dict(config)
    dflash = cfg.get("dflash_config", {})
    expected = {
        "hidden_size": 4096, "head_dim": 128, "num_hidden_layers": 3,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "target_hidden_size": 4096, "vocab_size": 151936, "intermediate_size": 12288,
    }
    for field, value in expected.items():
        if cfg.get(field) != value:
            raise ValueError(f"Frozen draft requires {field}={value}, got {cfg.get(field)!r}")
    dflash_expected = {
        "mask_token_id": 151669, "target_layer_ids": [1, 17, 33],
        "spatial_rope_enabled": True, "spatial_rope_mode": "replace",
        "spatial_rope_sections": [24, 20, 20], "spatial_rope_interleaved": True,
    }
    for field, value in dflash_expected.items():
        if dflash.get(field) != value:
            raise ValueError(f"Frozen draft requires dflash_config.{field}={value!r}")
    if cfg.get("block_size") != 8 or cfg.get("attention_bias", False):
        raise ValueError("Frozen draft requires block_size=8 and attention_bias=False")
    rope = cfg.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default" or rope.get("rope_theta") != 5000000.0:
        raise ValueError("Frozen draft requires default frequencies with theta=5000000")
    if require_marker:
        if dflash.get(MARKER) is not True:
            raise ValueError(f"Exported config must explicitly enable dflash_config.{MARKER}")
        if rope.get("mrope_section") != [24, 20, 20] or rope.get("mrope_interleaved") is not True:
            raise ValueError("Exported config must select official interleaved MRotaryEmbedding")
    return cfg


def absolute_context_positions(seq_lens: torch.Tensor, query_start_loc: torch.Tensor,
                               num_context: int) -> torch.Tensor:
    """Absolute token indices, independent of compressed MRoPE coordinates.

    seq_lens includes padded/rejected positions, as does query_start_loc.
    The official expansion kernel applies rejection offsets to query slots.
    Explicit output_size prevents a repeat_interleave GPU-to-host size sync.
    """
    batch_size = query_start_loc.numel() - 1
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    request = torch.repeat_interleave(
        torch.arange(batch_size, device=query_start_loc.device), lengths,
        output_size=num_context)
    offsets = torch.arange(num_context, device=query_start_loc.device)
    return (offsets - query_start_loc[:-1].index_select(0, request)
            + seq_lens.index_select(0, request) - lengths.index_select(0, request)).long()


def spatial_query_positions(target_axes: torch.Tensor, query_start_loc: torch.Tensor,
                            num_rejected: torch.Tensor | None,
                            num_query_per_req: int) -> torch.Tensor:
    """Keep T/H/W independently and continue from each last accepted token."""
    if target_axes.ndim != 2 or target_axes.shape[0] != 3:
        raise ValueError("Spatial RoPE requires three axes with shape [3, context_tokens]")
    valid_end = query_start_loc[1:].long()
    if num_rejected is not None:
        valid_end = valid_end - num_rejected.long()
    last_valid = target_axes.index_select(1, valid_end - 1)
    offsets = torch.arange(1, num_query_per_req + 1, device=target_axes.device,
                           dtype=target_axes.dtype)
    return (last_valid[:, :, None] + offsets).flatten(1)


def rotate_context_keys(rotary_emb: Any, context_axes: torch.Tensor,
                        flat_keys: torch.Tensor, num_layers: int,
                        *, native: bool = False) -> torch.Tensor:
    """Use the official MRoPE operator for the existing fused layer-major K.

    MRotaryEmbedding requires both Q and K. Treat context K as Q and supply a
    separate one-head zero tensor as the unused K. This avoids pointer aliasing
    and uses the official operator without modifying its kernel/API.
    """
    if context_axes.ndim == 1:
        # Only profiling/dummy_run uses the upstream all-zero 1D buffer.
        context_axes = context_axes.unsqueeze(0).expand(3, -1)
    if context_axes.ndim != 2 or context_axes.shape[0] != 3:
        raise ValueError("Context RoPE requires three axes")
    positions = context_axes.repeat(1, num_layers)
    dummy_key = torch.zeros((flat_keys.shape[0], rotary_emb.head_size),
                           dtype=flat_keys.dtype, device=flat_keys.device)
    operator = rotary_emb.forward_native if native else rotary_emb
    rotated, _ = operator(positions, flat_keys, dummy_key)
    return rotated


def _enabled(config: Any) -> bool:
    return getattr(config, "dflash_config", {}).get(MARKER) is True


def _check_source_bindings(modules: dict[str, Any]) -> dict[str, str]:
    """Fail closed if this adapter is used with different official source."""
    manifest = __import__("json").loads(Path(__file__).with_name("source_bindings.json").read_text())
    actual = {}
    for name, module in modules.items():
        digest = hashlib.sha256(Path(module.__file__).read_bytes().rstrip(b"\n")).hexdigest()
        if digest != manifest[name]["sha256_without_final_newlines"]:
            raise RuntimeError(f"vLLM source binding changed for {name}; adapter requires re-review")
        actual[name] = digest
    return actual


def install() -> dict[str, str]:
    """Install guarded hooks in this process before constructing vllm.LLM."""
    global _INSTALLED
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.models import qwen3_dflash as model_module
    from vllm.model_executor.layers.rotary_embedding import mrope as mrope_module
    from vllm.v1.spec_decode import dflash as proposer_module
    from vllm.v1.spec_decode import llm_base_proposer as base_module
    if vllm.__version__ != "0.22.1":
        raise RuntimeError(f"Adapter requires official vLLM 0.22.1, got {vllm.__version__}")
    bindings = _check_source_bindings({"qwen3_dflash": model_module, "dflash_proposer": proposer_module,
                                      "llm_base_proposer": base_module, "mrope": mrope_module})
    if _INSTALLED:
        return bindings
    Proposer = proposer_module.DFlashProposer
    Model = model_module.DFlashQwen3Model
    original_init = Proposer.__init__
    original_raise = Proposer._raise_if_mrope
    original_set = Proposer.set_inputs_first_pass
    original_build = Proposer.build_model_inputs_first_pass
    original_context = Model.precompute_and_store_context_kv

    def guarded_raise(self):
        if _enabled(self.draft_model_config.hf_config):
            validate_draft_config(self.draft_model_config.hf_config)
            return
        return original_raise(self)

    def init(self, vllm_config, device, runner=None):
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        if _enabled(cfg):
            validate_draft_config(cfg)
            pc = vllm_config.parallel_config
            if (pc.tensor_parallel_size, pc.pipeline_parallel_size, pc.data_parallel_size) != (1, 1, 1):
                raise ValueError("Initial spatial adapter requires TP=PP=DP=1")
            if vllm_config.scheduler_config.max_num_seqs != 1:
                raise ValueError("Initial spatial adapter requires max_num_seqs=1")
            if not vllm_config.model_config.enforce_eager:
                raise ValueError("Initial spatial adapter requires enforce_eager=True")
            if vllm_config.model_config.hf_config.model_type != "qwen3_vl":
                raise ValueError("Spatial adapter requires Qwen3-VL target")
            if vllm_config.speculative_config.num_speculative_tokens != 7:
                raise ValueError("Frozen block=8 requires num_speculative_tokens=7")
            if not vllm_config.speculative_config.draft_model_config.uses_mrope:
                raise ValueError("Draft model config did not enable official MRoPE")
        original_init(self, vllm_config, device, runner=runner)
        if _enabled(cfg):
            # Keep upstream self.positions and _context_positions_buffer 1D:
            # the official expansion kernel writes their physical positions.
            self.mrope_positions = torch.zeros((3, self.max_query_tokens + 1),
                                               dtype=torch.int64, device=device)
            self._context_mrope_positions = torch.zeros((3, self.max_num_tokens),
                                                        dtype=torch.int64, device=device)

    def set_inputs(self, target_token_ids, next_token_ids, target_positions,
                   target_hidden_states, token_indices_to_sample, cad,
                   num_rejected_tokens_gpu):
        if not _enabled(self.draft_model_config.hf_config):
            return original_set(self, target_token_ids, next_token_ids, target_positions,
                                target_hidden_states, token_indices_to_sample, cad,
                                num_rejected_tokens_gpu)
        n = target_token_ids.shape[0]
        if target_positions.shape != (3, n):
            raise ValueError(f"Spatial adapter requires target positions [3,{n}]")
        physical = absolute_context_positions(cad.seq_lens, cad.query_start_loc, n)
        result = original_set(self, target_token_ids, next_token_ids, physical,
                              target_hidden_states, token_indices_to_sample, cad,
                              num_rejected_tokens_gpu)
        axes = spatial_query_positions(target_positions, cad.query_start_loc,
                                       num_rejected_tokens_gpu, 1 + self.num_speculative_tokens)
        self._context_mrope_positions[:, :n].copy_(target_positions)
        self.mrope_positions.zero_()
        self.mrope_positions[:, :axes.shape[1]].copy_(axes)
        return result

    def build_inputs(self, num_tokens, num_input_tokens, mm_embed_inputs):
        if not _enabled(self.draft_model_config.hf_config):
            return original_build(self, num_tokens, num_input_tokens, mm_embed_inputs)
        n = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states, self._context_mrope_positions[:, :n],
            self._context_slot_mapping_buffer[:n])
        return dict(input_ids=self.input_ids[:num_input_tokens],
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None), num_input_tokens

    def precompute(self, context_states, context_positions, context_slot_mapping=None):
        if not _enabled(self.config):
            return original_context(self, context_states, context_positions, context_slot_mapping)
        if context_slot_mapping is not None and (
                context_positions.ndim != 2 or context_positions.shape[0] != 3):
            raise ValueError("Spatial adapter requires three axes for real context cache writes")
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        # Unchanged official fused projection, normalization and cache writes.
        # Only the 1D rotary_embedding call is replaced by official MRoPE.
        num_ctx, L, kv, hd, nkv = (context_states.shape[0], self._num_attn_layers,
                                  self._kv_size, self._head_dim, self._num_kv_heads)
        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, self._hidden_norm_weight, self._rms_norm_eps)
        all_kv_flat = F.linear(normed, self._fused_kv_weight, self._fused_kv_bias)
        all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        all_k, all_v = all_kv[0], all_kv[1]
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            ops.rms_norm(all_k_normed[i], all_k[i], self._k_norm_weights[i], self._rms_norm_eps)
        rotary = self.layers[0].self_attn.rotary_emb
        if not isinstance(rotary, mrope_module.MRotaryEmbedding):
            raise RuntimeError("Spatial adapter requires official MRotaryEmbedding")
        all_k_flat = rotate_context_keys(rotary, context_positions,
                                         all_k_normed.view(L * num_ctx, kv), L)
        if context_slot_mapping is None:
            return
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        for i in range(L):
            attn = self._attn_layers[i]
            attn.impl.do_kv_cache_update(attn, all_k_final[i], all_v[i],
                                        attn.kv_cache, context_slot_mapping)

    Proposer.__init__ = init
    Proposer._raise_if_mrope = guarded_raise
    Proposer.set_inputs_first_pass = set_inputs
    Proposer.build_model_inputs_first_pass = build_inputs
    Model.precompute_and_store_context_kv = precompute
    _INSTALLED = True
    return bindings
