"""Source-bound public DFlash S1/S3/S7 policy with frozen spatial math.

The checkpoint remains trained B8. Runtime queries B=1+S change the noncausal
proposal distribution. This module supplies no kernels and patches no target
forward or sampler. Import is CPU-safe; call install before engine creation.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
_CAMPAIGN = _ROOT.parent
_MANIFEST = json.loads((_ROOT / "source_bindings.json").read_text())


def _check_frozen_sources() -> dict[str, str]:
    result = {}
    for name, expected in _MANIFEST["frozen"].items():
        digest = hashlib.sha256((_CAMPAIGN / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Frozen dependency changed: {name}; re-review required")
        result[name] = digest
    return result


_check_frozen_sources()
sys.path.insert(0, str(_CAMPAIGN / "spatial_adapter"))
import official_spatial_adapter as frozen
import official_spatial_graph_adapter as graph_v1
import official_spatial_graph_adapter_v2 as graph_v2
from official_spatial_adapter import (
    MARKER, _enabled, absolute_context_positions, spatial_query_positions,
    rotate_context_keys, validate_draft_config,
)

for _module in (frozen, graph_v1, graph_v2):
    if Path(_module.__file__).resolve() != (_CAMPAIGN / "spatial_adapter" / (_module.__name__ + ".py")).resolve():
        raise RuntimeError(f"Frozen import resolved outside the pinned campaign: {_module.__name__}")

_INSTALLED = False
_BINDINGS: dict[str, str] = {}
_DUMMY_SCOPE = ContextVar("official_spatial_bounded_dummy_scope", default=False)


def query_block(num_speculative_tokens: int) -> int:
    if type(num_speculative_tokens) is not int or num_speculative_tokens not in (1, 3, 7):
        raise ValueError("Only public lookahead1/3/7 is validated")
    return 1 + num_speculative_tokens


def validate_query_batch(num_speculative_tokens: int, num_tokens: int,
                         num_padded: int, mode_name: str, *, dummy: bool,
                         keys_initialized: bool) -> None:
    block = query_block(num_speculative_tokens)
    if dummy:
        if not 1 <= num_tokens <= num_padded <= block:
            raise ValueError(f"Unsafe dummy query count/padding {num_tokens} to {num_padded}; bound={block}")
        if mode_name not in ("NONE", "PIECEWISE"):
            raise ValueError(f"Unsupported dummy graph dispatch {mode_name}")
        return
    if num_tokens != block:
        raise ValueError(f"Real spatial proposals require exactly {block} query tokens, got {num_tokens}")
    if num_padded != block:
        raise ValueError(f"Real spatial proposals reject query padding {block} to {num_padded}")
    if not keys_initialized or mode_name != "PIECEWISE":
        raise ValueError(f"Real spatial proposals require initialized PIECEWISE dispatch, got {mode_name}")


def validate_launch_config(vllm_config: Any) -> int:
    from vllm.config import CUDAGraphMode, CompilationMode
    spec = vllm_config.speculative_config
    block = query_block(spec.num_speculative_tokens)
    validate_draft_config(spec.draft_model_config.hf_config)
    if spec.method != "dflash":
        raise ValueError("Bounded spatial adapter requires method=dflash")
    pc = vllm_config.parallel_config
    if (pc.tensor_parallel_size, pc.pipeline_parallel_size, pc.data_parallel_size) != (1, 1, 1):
        raise ValueError("Bounded spatial adapter requires TP=PP=DP=1")
    if vllm_config.scheduler_config.max_num_seqs != 1:
        raise ValueError("Bounded spatial adapter requires max_num_seqs=1")
    if vllm_config.model_config.hf_config.model_type != "qwen3_vl":
        raise ValueError("Bounded spatial adapter requires Qwen3-VL target")
    if not spec.draft_model_config.uses_mrope:
        raise ValueError("Draft model config did not enable official MRoPE")
    if vllm_config.model_config.enforce_eager or spec.enforce_eager:
        raise ValueError("Bounded spatial adapter requires target and draft enforce_eager=False")
    comp = vllm_config.compilation_config
    if comp.mode != CompilationMode.VLLM_COMPILE:
        raise ValueError("Bounded spatial adapter requires official VLLM_COMPILE")
    if comp.cudagraph_mode.mixed_mode() not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        raise ValueError("Bounded spatial adapter requires mixed graph mode enabling draft PIECEWISE")
    sizes = comp.cudagraph_capture_sizes
    if sizes is not None and block not in sizes:
        raise ValueError(f"Bounded spatial adapter requires capture size {block}; query padding is unsupported")
    if vllm_config.scheduler_config.max_num_batched_tokens < block:
        raise ValueError(f"Bounded spatial adapter requires capacity for {block} query tokens")
    return block


def _runtime_block(proposer: Any) -> int:
    spec = proposer._bounded_num_speculative_tokens
    if (proposer.num_speculative_tokens != spec
            or proposer.speculative_config.num_speculative_tokens != spec
            or proposer.vllm_config.speculative_config.num_speculative_tokens != spec):
        raise ValueError("Public lookahead changed after buffer/graph initialization; use a new engine")
    return query_block(spec)


def in_dummy_scope() -> bool:
    return _DUMMY_SCOPE.get()


def install_scoped_query_guards(Proposer: Any, official_determine: Callable) -> None:
    previous_determine = Proposer._determine_batch_execution_and_padding
    for name, dummy in (("dummy_run", True), ("propose", False)):
        original = getattr(Proposer, name)
        def make_scoped(function, is_dummy):
            @wraps(function)
            def scoped(self, *args, **kwargs):
                if not _enabled(self.draft_model_config.hf_config):
                    return function(self, *args, **kwargs)
                _runtime_block(self)
                token = _DUMMY_SCOPE.set(is_dummy)
                try:
                    return function(self, *args, **kwargs)
                finally:
                    _DUMMY_SCOPE.reset(token)
            return scoped
        setattr(Proposer, name, make_scoped(original, dummy))

    def determine(self, num_tokens, use_cudagraphs=True):
        if not _enabled(self.draft_model_config.hf_config):
            return previous_determine(self, num_tokens, use_cudagraphs=use_cudagraphs)
        _runtime_block(self)
        result = official_determine(self, num_tokens, use_cudagraphs=use_cudagraphs)
        mode, padded, _ = result
        validate_query_batch(self._bounded_num_speculative_tokens, num_tokens, padded, mode.name,
            dummy=in_dummy_scope(), keys_initialized=self.cudagraph_dispatcher.keys_initialized)
        return result
    Proposer._determine_batch_execution_and_padding = determine


def _check_source_bindings() -> dict[str, str]:
    result = _check_frozen_sources()
    for name, entry in _MANIFEST["upstream"].items():
        module = importlib.import_module(entry["module"])
        contents = Path(module.__file__).read_bytes()
        if entry["strip_final_newlines"]:
            contents = contents.rstrip(b"\n")
        digest = hashlib.sha256(contents).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Official source binding changed for {name}; re-review required")
        result[name] = digest
    return result


def _require_official_function(function: Callable, module: Any) -> None:
    if Path(inspect.getsourcefile(function)).resolve() != Path(module.__file__).resolve():
        raise RuntimeError(f"Function was already replaced outside pinned official source: {function.__name__}")


def _require_official_dummy_run(function: Callable, module: Any) -> None:
    """Accept exactly the stock inference-mode wrapper and pinned official leaf."""
    reference = torch.inference_mode()(lambda: None)
    if getattr(function, "__code__", None) is not reference.__code__:
        raise RuntimeError("dummy_run does not have the stock Torch inference_mode wrapper")
    captured = inspect.getclosurevars(function).nonlocals
    expected = inspect.getclosurevars(reference).nonlocals
    wrapped = getattr(function, "__wrapped__", None)
    factory = captured.get("ctx_factory")
    expected_factory = expected["ctx_factory"]
    context = getattr(factory, "__self__", None)
    if (set(captured) != {"ctx_factory", "func"}
            or captured["func"] is not wrapped
            or getattr(factory, "__func__", None) is not expected_factory.__func__
            or type(context) is not type(expected_factory.__self__)
            or getattr(context, "mode", None) is not True
            or hasattr(wrapped, "__wrapped__")
            or getattr(getattr(wrapped, "__code__", None), "co_qualname", None) != "DFlashProposer.dummy_run"):
        raise RuntimeError("dummy_run inference_mode wrapper chain differs from the stock official callable")
    _require_official_function(wrapped, module)


def install() -> dict[str, str]:
    """Install this separate chain before constructing an engine in each worker."""
    global _INSTALLED, _BINDINGS
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.models import qwen3_dflash as model_module
    from vllm.model_executor.layers.rotary_embedding import mrope as mrope_module
    from vllm.v1.spec_decode import dflash as proposer_module
    from vllm.v1.spec_decode import llm_base_proposer as base_module
    if vllm.__version__ != "0.22.1":
        raise RuntimeError(f"Adapter requires official vLLM 0.22.1, got {vllm.__version__}")
    bindings = _check_source_bindings()
    if _INSTALLED:
        return dict(_BINDINGS)
    for name in ("official_spatial_adapter", "official_spatial_graph_adapter",
                 "official_spatial_graph_adapter_v2", "official_spatial_graph_adapter_v3"):
        module = sys.modules.get(name)
        if module is not None and module._INSTALLED:
            raise RuntimeError("Bounded adapter requires a fresh process without frozen adapter installation")
    Proposer = proposer_module.DFlashProposer
    Model = model_module.DFlashQwen3Model
    original_init = Proposer.__init__
    original_raise = Proposer._raise_if_mrope
    original_set = Proposer.set_inputs_first_pass
    original_build = Proposer.build_model_inputs_first_pass
    original_context = Model.precompute_and_store_context_kv
    original_determine = Proposer._determine_batch_execution_and_padding
    original_initialize_keys = Proposer.initialize_cudagraph_keys
    for function in (original_init, original_set, original_build):
        _require_official_function(function, proposer_module)
    _require_official_dummy_run(Proposer.dummy_run, proposer_module)
    for function in (original_raise, original_determine, original_initialize_keys, Proposer.propose):
        _require_official_function(function, base_module)
    _require_official_function(original_context, model_module)
    binding = graph_v2.inspect_metadata_binding(Model)

    def guarded_raise(self):
        if _enabled(self.draft_model_config.hf_config):
            validate_draft_config(self.draft_model_config.hf_config)
            return
        return original_raise(self)

    def init(self, vllm_config, device, runner=None):
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        enabled = _enabled(cfg)
        block = validate_launch_config(vllm_config) if enabled else None
        # Real S and trained block_size=8 reach the official constructor unchanged.
        original_init(self, vllm_config, device, runner=runner)
        if enabled:
            if self.max_query_tokens != block:
                raise RuntimeError("Official query-buffer capacity differs from bounded public lookahead")
            self._bounded_num_speculative_tokens = self.num_speculative_tokens
            self.mrope_positions = torch.zeros((3, block + 1), dtype=torch.int64, device=device)
            self._context_mrope_positions = torch.zeros((3, self.max_num_tokens), dtype=torch.int64, device=device)

    def set_inputs(self, target_token_ids, next_token_ids, target_positions,
                   target_hidden_states, token_indices_to_sample, cad,
                   num_rejected_tokens_gpu):
        if not _enabled(self.draft_model_config.hf_config):
            return original_set(self, target_token_ids, next_token_ids, target_positions,
                                target_hidden_states, token_indices_to_sample, cad,
                                num_rejected_tokens_gpu)
        block = _runtime_block(self)
        if cad.batch_size() != 1:
            raise ValueError("Bounded spatial proposals require batch size one")
        n = target_token_ids.shape[0]
        if target_positions.shape != (3, n):
            raise ValueError(f"Spatial adapter requires target positions [3,{n}]")
        physical = absolute_context_positions(cad.seq_lens, cad.query_start_loc, n)
        result = original_set(self, target_token_ids, next_token_ids, physical,
                              target_hidden_states, token_indices_to_sample, cad,
                              num_rejected_tokens_gpu)
        axes = spatial_query_positions(target_positions, cad.query_start_loc,
                                       num_rejected_tokens_gpu, block)
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

    def initialize_keys(self, cudagraph_mode):
        original_initialize_keys(self, cudagraph_mode)
        if _enabled(self.draft_model_config.hf_config):
            block = _runtime_block(self)
            mode, descriptor = self.cudagraph_dispatcher.dispatch(block)
            validate_query_batch(self._bounded_num_speculative_tokens, block, descriptor.num_tokens,
                mode.name, dummy=False, keys_initialized=self.cudagraph_dispatcher.keys_initialized)

    @wraps(binding.original_initializer)
    def model_init(self, *args, **kwargs):
        binding.original_initializer(self, *args, **kwargs)
        if _enabled(self.config):
            graph_v2.repair_metadata_mapping(self, binding)

    Proposer.__init__ = init
    Proposer._raise_if_mrope = guarded_raise
    Proposer.set_inputs_first_pass = set_inputs
    Proposer.build_model_inputs_first_pass = build_inputs
    Proposer.initialize_cudagraph_keys = initialize_keys
    Model.precompute_and_store_context_kv = precompute
    Model.__init__ = model_init
    # Reuse the frozen optional passthrough trace; never enable it for timing.
    graph_v1._install_diagnostic_trace(Proposer)
    install_scoped_query_guards(Proposer, original_determine)
    _BINDINGS = bindings
    _INSTALLED = True
    return dict(_BINDINGS)
