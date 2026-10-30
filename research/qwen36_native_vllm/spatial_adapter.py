"""Process-local vLLM 0.22.1 spatial-MRoPE adapter for the Qwen3.6 DFlash family.

Port of research/qwen3vl8b_native_vllm/spatial_adapter (eager v1 math, graph v1
guards, graph v2 compile-metadata repair, graph v3 dummy/real scopes) to the
frozen Qwen3.6 draft: hidden 2048, 16/2 heads, head_dim 256, partial rotary 64,
interleaved MRoPE sections [11, 11, 10], TP 1 or 2.

No installed files change and no kernels are supplied. The official DFlash
expansion kernel still owns token IDs, physical cache slots, rejection handling
and sampling indices; three-axis positions are used only for RoPE. Method
adapters (DFly, DFlash2, DFlare, Domino, JetSpec, DSpark) install this first and
then add their own module hooks.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

import contract
from contract import MARKER, validate_draft_config, validate_launch

_INSTALLED = False
_BINDINGS: dict[str, str] = {}
_DUMMY_SCOPE = ContextVar("qwen36_spatial_dummy_scope", default=False)

# Same official vLLM 0.22.1 sources the 8B campaign reviewed (sha256 of the file
# without trailing newlines); verified byte-identical on the eval environment.
SOURCE_BINDINGS = {
    "vllm.model_executor.models.qwen3_dflash": "822279152563e53f80643148b61252de230e00d3037e7fe7784045168a13b9fa",
    "vllm.v1.spec_decode.dflash": "d632a8a7e6bba4fd5dd6b07cb53891d74f5d4dee272d8efae0b57265c1e14b90",
    "vllm.v1.spec_decode.llm_base_proposer": "b8cd732a305e48119275f786bb9043dd0e4bddd0db759d44a9e39a15f5f51a89",
    "vllm.model_executor.layers.rotary_embedding.mrope": "c5a7397040955daded984d66001a1445a2a0a8e90c5a35599752c9cfb59a0b80",
}


def enabled(config: Any) -> bool:
    return getattr(config, "dflash_config", {}).get(MARKER) is True


def in_dummy_scope() -> bool:
    return _DUMMY_SCOPE.get()


def absolute_context_positions(seq_lens: torch.Tensor, query_start_loc: torch.Tensor,
                               num_context: int) -> torch.Tensor:
    """Absolute token indices, independent of compressed MRoPE coordinates.

    seq_lens includes padded/rejected positions, as does query_start_loc. The
    official expansion kernel applies rejection offsets to query slots.
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
    """Official MRoPE operator on the fused layer-major context K.

    MRotaryEmbedding requires both Q and K: context K is passed as Q with a
    separate one-head zero tensor as the unused K (no pointer aliasing).
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


def draft_graph_modes() -> tuple[str, ...]:
    """Admitted draft graph dispatch for real proposals (PIECEWISE unless the frozen
    launch also admits FULL via Q36_DRAFT_GRAPH_MODES=PIECEWISE,FULL)."""
    modes = tuple(m for m in os.environ.get("Q36_DRAFT_GRAPH_MODES", "PIECEWISE").split(",") if m)
    if not modes or set(modes) - {"PIECEWISE", "FULL"}:
        raise ValueError(f"Q36_DRAFT_GRAPH_MODES must list PIECEWISE and/or FULL, got {modes}")
    return modes


def validate_scoped_query_batch(num_tokens: int, num_padded: int, mode_name: str,
                                *, dummy: bool, keys_initialized: bool) -> None:
    block = contract.BLOCK_SIZE
    modes = draft_graph_modes()
    if dummy:
        if not 1 <= num_tokens <= block or not num_tokens <= num_padded <= block:
            raise ValueError(f"Unsafe dummy query batch {num_tokens} padded to {num_padded}")
        if mode_name not in ("NONE", *modes):
            raise ValueError(f"Unsupported dummy graph dispatch {mode_name}")
        return
    if num_tokens != block or num_padded != block:
        raise ValueError(f"Real spatial proposals require exactly eight query tokens, got {num_tokens}->{num_padded}")
    if not keys_initialized or mode_name not in modes:
        raise ValueError(f"Real spatial proposals require initialized {modes} dispatch, got {mode_name}")


def validate_target_head_binding(draft: Any, target_language_model: Any) -> None:
    embedding = target_language_model.model.embed_tokens
    head = target_language_model.lm_head
    if head is embedding or draft.model.embed_tokens is not embedding or draft.lm_head is not head:
        raise ValueError("Qwen3.6 draft must share the target's distinct embedding and untied output head")


def check_source_bindings() -> dict[str, str]:
    import importlib
    actual = {}
    for name, expected in SOURCE_BINDINGS.items():
        module = importlib.import_module(name)
        digest = hashlib.sha256(Path(module.__file__).read_bytes().rstrip(b"\n")).hexdigest()
        if digest != expected:
            raise RuntimeError(f"vLLM source binding changed for {name}; adapter requires re-review")
        actual[name] = digest
    return actual


def _compile_metadata_binding(model_class: type) -> tuple[dict, Any, Any]:
    """Locate the official support_torch_compile dynamic-dims closure (graph v2)."""
    from vllm.compilation import decorators
    initializer, call_function = model_class.__init__, model_class.__call__
    try:
        mapping = inspect.getclosurevars(initializer).nonlocals["dynamic_arg_dims"]
        mark_function = inspect.getclosurevars(call_function).nonlocals["_mark_dynamic_inputs"]
        mark_mapping = inspect.getclosurevars(mark_function).nonlocals["dynamic_arg_dims"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Official compile metadata closure contract changed") from error
    decorator_file = Path(decorators.__file__).resolve()
    for function in (initializer, call_function, mark_function):
        if Path(inspect.getsourcefile(function)).resolve() != decorator_file:
            raise RuntimeError("Compile metadata functions no longer come from the official decorator")
    if mapping is not mark_mapping:
        raise RuntimeError("Official initializer/marker metadata identity mismatch")
    if mapping != {"input_ids": 0, "positions": 0, "input_embeds": 0}:
        raise RuntimeError(f"Official compile metadata has unexpected initial values: {mapping!r}")
    return mapping, initializer, call_function


# Exact official Qwen2Model declaration for tokens and MRoPE positions; the inner
# DFlash model's embedding argument uses the singular name input_embeds.
CORRECTED_METADATA = {"input_ids": {0: "b"}, "positions": {-1: "b"}, "input_embeds": {0: "b"}}


def install() -> dict[str, str]:
    """Install guarded hooks in this process before constructing vllm.LLM."""
    global _INSTALLED, _BINDINGS
    import vllm
    if vllm.__version__ != "0.22.1":
        raise RuntimeError(f"Adapter requires official vLLM 0.22.1, got {vllm.__version__}")
    bindings = check_source_bindings()
    if _INSTALLED:
        return dict(_BINDINGS)
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.rotary_embedding import mrope as mrope_module
    from vllm.model_executor.models import qwen3_dflash as model_module
    from vllm.v1.spec_decode import dflash as proposer_module
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    Proposer = proposer_module.DFlashProposer
    Model = model_module.DFlashQwen3Model
    mapping, model_initializer, model_call = _compile_metadata_binding(Model)
    original_init = Proposer.__init__
    original_raise = Proposer._raise_if_mrope
    original_set = Proposer.set_inputs_first_pass
    original_build = Proposer.build_model_inputs_first_pass
    original_initialize_keys = Proposer.initialize_cudagraph_keys
    original_load_model = Proposer.load_model
    original_context = Model.precompute_and_store_context_kv
    official_determine = SpecDecodeBaseProposer._determine_batch_execution_and_padding
    if Proposer._determine_batch_execution_and_padding is not official_determine:
        raise RuntimeError("DFlash proposer no longer inherits the official dispatcher")

    @wraps(model_initializer)
    def model_init(self, *args, **kwargs):
        model_initializer(self, *args, **kwargs)
        if not enabled(self.config):
            return
        # Graph v2: fix MRoPE positions' dynamic axis in place, before first compile.
        if not (self._dynamic_arg_dims is mapping) or self.do_not_compile or self.compiled:
            raise RuntimeError("Metadata repair must run after compiled init and before the first compile")
        if Model.__call__ is not model_call:
            raise RuntimeError("Official compiled call function identity changed")
        if mapping not in ({"input_ids": 0, "positions": 0, "input_embeds": 0}, CORRECTED_METADATA):
            raise RuntimeError(f"Refusing unexpected compile metadata values: {mapping!r}")
        mapping.clear()
        mapping.update(CORRECTED_METADATA)

    def checked_load_model(self, target_model):
        result = original_load_model(self, target_model)
        if enabled(self.draft_model_config.hf_config):
            validate_target_head_binding(self.model, target_model.get_language_model())
        return result

    def guarded_raise(self):
        if enabled(self.draft_model_config.hf_config):
            validate_draft_config(self.draft_model_config.hf_config)
            return
        return original_raise(self)

    def init(self, vllm_config, device, runner=None):
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        if enabled(cfg):
            validate_draft_config(cfg)
            validate_launch(vllm_config)
        original_init(self, vllm_config, device, runner=runner)
        if enabled(cfg):
            # Upstream self.positions / _context_positions_buffer stay 1D: the
            # official expansion kernel writes their physical positions.
            self.mrope_positions = torch.zeros((3, self.max_query_tokens + 1),
                                               dtype=torch.int64, device=device)
            self._context_mrope_positions = torch.zeros((3, self.max_num_tokens),
                                                        dtype=torch.int64, device=device)

    def set_inputs(self, target_token_ids, next_token_ids, target_positions,
                   target_hidden_states, token_indices_to_sample, cad,
                   num_rejected_tokens_gpu):
        if not enabled(self.draft_model_config.hf_config):
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
        if not enabled(self.draft_model_config.hf_config):
            return original_build(self, num_tokens, num_input_tokens, mm_embed_inputs)
        n = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states, self._context_mrope_positions[:, :n],
            self._context_slot_mapping_buffer[:n])
        return dict(input_ids=self.input_ids[:num_input_tokens],
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None), num_input_tokens

    def precompute(self, context_states, context_positions, context_slot_mapping=None):
        if not enabled(self.config):
            return original_context(self, context_states, context_positions, context_slot_mapping)
        if context_slot_mapping is not None and (
                context_positions.ndim != 2 or context_positions.shape[0] != 3):
            raise ValueError("Spatial adapter requires three axes for real context cache writes")
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        # Unchanged official fused projection, normalization and cache writes;
        # only the 1D rotary_embedding call is replaced by official MRoPE.
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

    trace_keys = set()

    def determine(self, num_tokens, use_cudagraphs=True):
        result = official_determine(self, num_tokens, use_cudagraphs=use_cudagraphs)
        if enabled(self.draft_model_config.hf_config):
            mode, padded, _ = result
            dummy = in_dummy_scope()
            validate_scoped_query_batch(num_tokens, padded, mode.name, dummy=dummy,
                                        keys_initialized=self.cudagraph_dispatcher.keys_initialized)
            if os.environ.get("VLLM_SPATIAL_GRAPH_TRACE") == "1":
                key = (dummy, num_tokens, padded, mode.name)
                if key not in trace_keys:
                    trace_keys.add(key)
                    print("SPATIAL_DRAFT_QUERY_SCOPE " + json.dumps({
                        "scope": "dummy" if dummy else "real_propose", "requested": num_tokens,
                        "padded": padded, "mode": mode.name, "diagnostic_only": True}),
                        file=sys.stderr, flush=True)
        return result

    def initialize_keys(self, cudagraph_mode):
        if enabled(self.draft_model_config.hf_config) and os.environ.get("Q36_DRAFT_CUDAGRAPH") == "PIECEWISE":
            # Target keeps the engine's graph mode; the draft dispatches PIECEWISE only.
            from vllm.config import CUDAGraphMode
            cudagraph_mode = CUDAGraphMode.PIECEWISE
        original_initialize_keys(self, cudagraph_mode)
        if enabled(self.draft_model_config.hf_config):
            mode, descriptor = self.cudagraph_dispatcher.dispatch(contract.BLOCK_SIZE)
            validate_scoped_query_batch(contract.BLOCK_SIZE, descriptor.num_tokens, mode.name,
                                        dummy=False, keys_initialized=True)

    for name, dummy in (("dummy_run", True), ("propose", False)):
        def make_scoped(function, is_dummy):
            @wraps(function)
            def scoped(self, *args, **kwargs):
                if not enabled(self.draft_model_config.hf_config):
                    return function(self, *args, **kwargs)
                token = _DUMMY_SCOPE.set(is_dummy)
                try:
                    return function(self, *args, **kwargs)
                finally:
                    _DUMMY_SCOPE.reset(token)
            return scoped
        setattr(Proposer, name, make_scoped(getattr(Proposer, name), dummy))

    Model.__init__ = model_init
    Model.precompute_and_store_context_kv = precompute
    Proposer.__init__ = init
    Proposer.load_model = checked_load_model
    Proposer._raise_if_mrope = guarded_raise
    Proposer.set_inputs_first_pass = set_inputs
    Proposer.build_model_inputs_first_pass = build_inputs
    Proposer._determine_batch_execution_and_padding = determine
    Proposer.initialize_cudagraph_keys = initialize_keys
    _BINDINGS = bindings
    _INSTALLED = True
    return dict(_BINDINGS)
