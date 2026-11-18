"""Separate fixed-eight-query graph candidate; frozen eager math is retained.

Official compilation/capture and their dynamic metadata are unchanged. No new
kernels. This candidate is not graph-validated until the root GPU smoke passes.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
import official_spatial_adapter as frozen
from official_spatial_adapter import (
    MARKER, absolute_context_positions, spatial_query_positions,
    rotate_context_keys, validate_draft_config, _enabled, _check_source_bindings,
)

_INSTALLED = False


def validate_target_head_binding(draft: Any, target_language_model: Any) -> None:
    embedding = target_language_model.model.embed_tokens
    head = target_language_model.lm_head
    if head is embedding or draft.model.embed_tokens is not embedding or draft.lm_head is not head:
        raise ValueError("8B draft must share the target's distinct embedding and output head")


def validate_graph_launch_config(vllm_config: Any) -> None:
    from vllm.config import CUDAGraphMode, CompilationMode
    if vllm_config.model_config.enforce_eager:
        raise ValueError("Graph candidate requires target enforce_eager=False")
    if vllm_config.speculative_config.enforce_eager:
        raise ValueError("Graph candidate requires speculative enforce_eager=False")
    comp = vllm_config.compilation_config
    if comp.mode != CompilationMode.VLLM_COMPILE:
        raise ValueError("Graph candidate requires the official VLLM_COMPILE path")
    if comp.cudagraph_mode.mixed_mode() not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        raise ValueError("Graph candidate requires a mixed graph mode enabling draft PIECEWISE graphs")
    sizes = comp.cudagraph_capture_sizes
    if sizes is not None and 8 not in sizes:
        raise ValueError("Graph candidate requires capture size eight; padding to sixteen is unsupported")
    if vllm_config.scheduler_config.max_num_batched_tokens < 8:
        raise ValueError("Graph candidate requires capacity for eight query tokens")


def validate_query_batch(num_tokens: int, num_padded: int, mode_name: str,
                         *, graphs_requested: bool, keys_initialized: bool) -> None:
    if num_tokens != 8:
        raise ValueError(f"Graph candidate supports exactly eight query tokens, got {num_tokens}")
    if num_padded != 8:
        raise ValueError(f"Graph candidate rejects query padding eight to {num_padded}")
    if graphs_requested and keys_initialized and mode_name != "PIECEWISE":
        raise ValueError(f"Graph candidate requires actual draft PIECEWISE dispatch, got {mode_name}")


def _install_diagnostic_trace(Proposer: Any) -> None:
    """Optional Python-only passthrough; disable for every performance run."""
    if os.environ.get("VLLM_SPATIAL_GRAPH_TRACE") != "1":
        return
    from contextvars import ContextVar
    from functools import wraps
    import json
    import sys
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    scope = ContextVar("official_spatial_draft_graph_scope", default=None)
    for name in ("dummy_run", "propose"):
        original = getattr(Proposer, name)
        def make_scoped(function):
            @wraps(function)
            def scoped(self, *args, **kwargs):
                if not _enabled(self.draft_model_config.hf_config):
                    return function(self, *args, **kwargs)
                names = tuple(sorted(getattr(self, "_draft_attn_layer_names", ())))
                token = scope.set(names)
                try:
                    return function(self, *args, **kwargs)
                finally:
                    scope.reset(token)
            return scoped
        setattr(Proposer, name, make_scoped(original))

    original_call = CUDAGraphWrapper.__call__
    counts = {}
    @wraps(original_call)
    def traced_call(self, *args, **kwargs):
        layers = scope.get()
        if layers is None or not is_forward_context_available():
            return original_call(self, *args, **kwargs)
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE or self.runtime_mode != CUDAGraphMode.PIECEWISE:
            return original_call(self, *args, **kwargs)
        descriptor = context.batch_descriptor
        before = self.concrete_cudagraph_entries.get(descriptor)
        replay = before is not None and before.cudagraph is not None
        output = original_call(self, *args, **kwargs)
        after = self.concrete_cudagraph_entries.get(descriptor)
        if after is not None and after.cudagraph is not None:
            key = (id(self), "replay" if replay else "capture")
            counts[key] = counts.get(key, 0) + 1
            if not replay or counts[key] <= 3:
                print("SPATIAL_DRAFT_GRAPH " + json.dumps({
                    "event": key[1], "wrapper_id": id(self), "count": counts[key],
                    "draft_attention_layers": layers, "descriptor": str(descriptor),
                    "diagnostic_only": True}), file=sys.stderr, flush=True)
        return output
    CUDAGraphWrapper.__call__ = traced_call


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
    if frozen._INSTALLED:
        raise RuntimeError("Graph candidate must run in a separate process from the frozen eager adapter")
    Proposer = proposer_module.DFlashProposer
    Model = model_module.DFlashQwen3Model
    original_init = Proposer.__init__
    original_raise = Proposer._raise_if_mrope
    original_set = Proposer.set_inputs_first_pass
    original_build = Proposer.build_model_inputs_first_pass
    original_context = Model.precompute_and_store_context_kv
    original_determine = Proposer._determine_batch_execution_and_padding
    original_initialize_keys = Proposer.initialize_cudagraph_keys
    original_load_model = Proposer.load_model

    def checked_load_model(self, target_model):
        result = original_load_model(self, target_model)
        if _enabled(self.draft_model_config.hf_config):
            validate_target_head_binding(self.model, target_model.get_language_model())
        return result

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
            validate_graph_launch_config(vllm_config)
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

    def determine(self, num_tokens, use_cudagraphs=True):
        result = original_determine(self, num_tokens, use_cudagraphs=use_cudagraphs)
        if _enabled(self.draft_model_config.hf_config):
            mode, padded, _ = result
            validate_query_batch(num_tokens, padded, mode.name,
                graphs_requested=use_cudagraphs,
                keys_initialized=self.cudagraph_dispatcher.keys_initialized)
        return result

    def initialize_keys(self, cudagraph_mode):
        original_initialize_keys(self, cudagraph_mode)
        if _enabled(self.draft_model_config.hf_config):
            mode, descriptor = self.cudagraph_dispatcher.dispatch(8)
            validate_query_batch(8, descriptor.num_tokens, mode.name,
                                  graphs_requested=True, keys_initialized=True)

    Proposer.__init__ = init
    Proposer.load_model = checked_load_model
    Proposer._raise_if_mrope = guarded_raise
    Proposer.set_inputs_first_pass = set_inputs
    Proposer.build_model_inputs_first_pass = build_inputs
    Proposer._determine_batch_execution_and_padding = determine
    Proposer.initialize_cudagraph_keys = initialize_keys
    Model.precompute_and_store_context_kv = precompute
    _install_diagnostic_trace(Proposer)
    _INSTALLED = True
    return bindings
