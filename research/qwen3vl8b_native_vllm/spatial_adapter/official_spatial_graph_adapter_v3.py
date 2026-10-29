"""Graph v3: allow bounded official dummy batches, keep real proposals strict.

Preserves graph-v2 metadata repair and all existing math, kernels and traces.
The official base dispatcher is called directly; only adapter guard placement
changes, using explicit dummy/propose scopes that reset even on exceptions.
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
from typing import Any, Callable

import official_spatial_graph_adapter_v2 as graph_v2
from official_spatial_adapter import _enabled

_INSTALLED = False
_BINDINGS: dict[str, str] = {}
_DUMMY_SCOPE = ContextVar("official_spatial_graph_dummy_scope", default=False)


def in_dummy_scope() -> bool:
    return _DUMMY_SCOPE.get()


def validate_scoped_query_batch(num_tokens: int, num_padded: int, mode_name: str,
                                *, dummy: bool, graphs_requested: bool,
                                keys_initialized: bool) -> None:
    if dummy:
        if not 1 <= num_tokens <= 8:
            raise ValueError(f"Official dummy query count must be within one to eight, got {num_tokens}")
        if not num_tokens <= num_padded <= 8:
            raise ValueError(f"Unsafe dummy query padding {num_tokens} to {num_padded}")
        if mode_name not in ("NONE", "PIECEWISE"):
            raise ValueError(f"Unsupported dummy graph dispatch {mode_name}")
        return
    if num_tokens != 8:
        raise ValueError(f"Real spatial proposals require exactly eight query tokens, got {num_tokens}")
    if num_padded != 8:
        raise ValueError(f"Real spatial proposals reject query padding eight to {num_padded}")
    if not keys_initialized or mode_name != "PIECEWISE":
        raise ValueError(f"Real spatial proposals require initialized PIECEWISE dispatch, got {mode_name}")


def _check_v3_source_bindings() -> dict[str, str]:
    from vllm.v1.spec_decode import llm_base_proposer
    manifest = json.loads(Path(__file__).with_name("graph_v3_source_bindings.json").read_text())
    v2_digest = hashlib.sha256(Path(graph_v2.__file__).read_bytes()).hexdigest()
    base_path = Path(llm_base_proposer.__file__)
    base_digest = hashlib.sha256(base_path.read_bytes().rstrip(b"\n")).hexdigest()
    if v2_digest != manifest["graph_v2_adapter"]["sha256"]:
        raise RuntimeError("Frozen graph-v2 adapter source binding changed")
    if base_digest != manifest["official_base_proposer"]["sha256_without_final_newlines"]:
        raise RuntimeError("Official base proposer source binding changed")
    determine = llm_base_proposer.SpecDecodeBaseProposer._determine_batch_execution_and_padding
    if Path(inspect.getsourcefile(determine)).resolve() != base_path.resolve():
        raise RuntimeError("Base dispatcher function no longer comes from its pinned official source")
    return {"graph_v2_adapter": v2_digest, "official_base_proposer": base_digest}


def install_scoped_query_guards(Proposer: Any, official_determine: Callable) -> None:
    previous_determine = Proposer._determine_batch_execution_and_padding
    for name, dummy in (("dummy_run", True), ("propose", False)):
        original = getattr(Proposer, name)
        def make_scoped(function, is_dummy):
            @wraps(function)
            def scoped(self, *args, **kwargs):
                if not _enabled(self.draft_model_config.hf_config):
                    return function(self, *args, **kwargs)
                token = _DUMMY_SCOPE.set(is_dummy)
                try:
                    return function(self, *args, **kwargs)
                finally:
                    _DUMMY_SCOPE.reset(token)
            return scoped
        setattr(Proposer, name, make_scoped(original, dummy))

    trace_keys = set()
    def determine(self, num_tokens, use_cudagraphs=True):
        if not _enabled(self.draft_model_config.hf_config):
            return previous_determine(self, num_tokens, use_cudagraphs=use_cudagraphs)
        # Call the unchanged official implementation, bypassing only v1's
        # adapter guard. No private closure is called or rebound.
        result = official_determine(self, num_tokens, use_cudagraphs=use_cudagraphs)
        mode, padded, _ = result
        dummy = in_dummy_scope()
        validate_scoped_query_batch(num_tokens, padded, mode.name, dummy=dummy,
            graphs_requested=use_cudagraphs,
            keys_initialized=self.cudagraph_dispatcher.keys_initialized)
        if os.environ.get("VLLM_SPATIAL_GRAPH_TRACE") == "1":
            key = (dummy, num_tokens, padded, mode.name)
            if key not in trace_keys:
                trace_keys.add(key)
                print("SPATIAL_DRAFT_QUERY_SCOPE " + json.dumps({
                    "scope": "dummy" if dummy else "real_propose", "requested": num_tokens,
                    "padded": padded, "mode": mode.name,
                    "diagnostic_only": True}), file=sys.stderr, flush=True)
        return result
    Proposer._determine_batch_execution_and_padding = determine


def install() -> dict[str, str]:
    global _INSTALLED, _BINDINGS
    extra_bindings = _check_v3_source_bindings()
    if _INSTALLED:
        return dict(_BINDINGS)
    if graph_v2._INSTALLED:
        raise RuntimeError("Graph v3 requires a fresh process without earlier adapter bootstrap installation")
    base_bindings = graph_v2.install()
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    official_determine = SpecDecodeBaseProposer._determine_batch_execution_and_padding
    # Pin identity of the bypassed v1 wrapper's underlying official operation.
    captured = inspect.getclosurevars(DFlashProposer._determine_batch_execution_and_padding).nonlocals
    if captured.get("original_determine") is not official_determine:
        raise RuntimeError("Frozen v1 dispatcher does not wrap the pinned official base function")
    install_scoped_query_guards(DFlashProposer, official_determine)
    _BINDINGS = dict(base_bindings, **extra_bindings)
    _INSTALLED = True
    return dict(_BINDINGS)
