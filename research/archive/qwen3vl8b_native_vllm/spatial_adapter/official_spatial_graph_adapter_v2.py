"""Graph v2: repair only the reproduced MRoPE compile-axis metadata failure.

Reuses the frozen graph-v1 adapter and every existing math/operator/graph guard.
The official decorator dictionary is mutated in place after marked model init,
only after source, value and object-identity checks. No closure is rebound.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import official_spatial_graph_adapter as graph_v1
from official_spatial_adapter import _enabled

_INSTALLED = False
_BINDINGS: dict[str, str] = {}


def original_metadata() -> dict:
    return {"input_ids": 0, "positions": 0, "input_embeds": 0}


def corrected_metadata() -> dict:
    # Exact official Qwen2Model declaration for tokens and positions; the inner
    # DFlash model's embedding argument uses the singular name input_embeds.
    return {"input_ids": {0: "b"}, "positions": {-1: "b"}, "input_embeds": {0: "b"}}


def _check_v2_source_bindings() -> dict[str, str]:
    from vllm.compilation import decorators
    manifest = json.loads(Path(__file__).with_name("graph_v2_source_bindings.json").read_text())
    files = {"compile_decorators": Path(decorators.__file__),
             "graph_v1_adapter": Path(graph_v1.__file__)}
    result = {}
    for name, path in files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest[name]["sha256"]:
            raise RuntimeError(f"Graph v2 source binding changed for {name}; re-review required")
        result[name] = digest
    return result


@dataclass(frozen=True)
class MetadataBinding:
    model_class: type
    original_initializer: Callable
    call_function: Callable
    mark_function: Callable
    mapping: dict


def inspect_metadata_binding(model_class: type) -> MetadataBinding:
    _check_v2_source_bindings()
    from vllm.compilation import decorators
    initializer = model_class.__init__
    call_function = model_class.__call__
    try:
        init_nonlocals = inspect.getclosurevars(initializer).nonlocals
        call_nonlocals = inspect.getclosurevars(call_function).nonlocals
        mark_function = call_nonlocals["_mark_dynamic_inputs"]
        mapping = init_nonlocals["dynamic_arg_dims"]
        mark_mapping = inspect.getclosurevars(mark_function).nonlocals["dynamic_arg_dims"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Official compile metadata closure contract changed") from error
    decorator_file = Path(decorators.__file__).resolve()
    for function in (initializer, call_function, mark_function):
        if Path(inspect.getsourcefile(function)).resolve() != decorator_file:
            raise RuntimeError("Compile metadata functions no longer come from the pinned official decorator")
    if mapping is not mark_mapping:
        raise RuntimeError("Official initializer/marker metadata identity mismatch")
    if mapping != original_metadata():
        raise RuntimeError(f"Official compile metadata has unexpected initial values: {mapping!r}")
    return MetadataBinding(model_class, initializer, call_function, mark_function, mapping)


def repair_metadata_mapping(instance: Any, binding: MetadataBinding) -> None:
    init_mapping = inspect.getclosurevars(binding.original_initializer).nonlocals["dynamic_arg_dims"]
    mark_mapping = inspect.getclosurevars(binding.mark_function).nonlocals["dynamic_arg_dims"]
    if not (instance._dynamic_arg_dims is binding.mapping is init_mapping is mark_mapping):
        raise RuntimeError("Instance/initializer/marker metadata identity mismatch")
    if binding.model_class.__call__ is not binding.call_function:
        raise RuntimeError("Official compiled call function identity changed")
    if instance.do_not_compile or instance.compiled:
        raise RuntimeError("Metadata repair must run after compiled init and before the first compile")
    if binding.mapping not in (original_metadata(), corrected_metadata()):
        raise RuntimeError(f"Refusing unexpected compile metadata values: {binding.mapping!r}")
    binding.mapping.clear()
    binding.mapping.update(corrected_metadata())
    if not (instance._dynamic_arg_dims is binding.mapping is init_mapping is mark_mapping):
        raise RuntimeError("Metadata identity changed during in-place repair")
    if binding.mapping != corrected_metadata():
        raise RuntimeError("Metadata repair did not establish the official shared-token declaration")


def install() -> dict[str, str]:
    """Install graph v1 plus the guarded metadata-only constructor wrapper."""
    global _INSTALLED, _BINDINGS
    extra_bindings = _check_v2_source_bindings()
    if _INSTALLED:
        return dict(_BINDINGS)
    if graph_v1._INSTALLED:
        raise RuntimeError("Graph v2 requires a fresh process, without graph-v1 bootstrap installation")
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    binding = inspect_metadata_binding(DFlashQwen3Model)
    base_bindings = graph_v1.install()

    @wraps(binding.original_initializer)
    def model_init(self, *args, **kwargs):
        binding.original_initializer(self, *args, **kwargs)
        if not _enabled(self.config):
            return
        repair_metadata_mapping(self, binding)
        if os.environ.get("VLLM_SPATIAL_GRAPH_TRACE") == "1":
            print("SPATIAL_DRAFT_COMPILE_METADATA " + json.dumps({
                "metadata": self._dynamic_arg_dims,
                "instance_initializer_marker_same_object": True,
                "mutation": "in_place", "forward_changed": False,
                "diagnostic_only": True}), file=sys.stderr, flush=True)

    DFlashQwen3Model.__init__ = model_init
    _BINDINGS = dict(base_bindings, **extra_bindings)
    _INSTALLED = True
    return dict(_BINDINGS)
