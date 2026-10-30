"""Local shared-block JetSpec on the Qwen3.6 target (block-causal draft queries).

Port of research/qwen3vl8b_native_vllm/jetspec_8b_work_20261029/jetspec_adapter to
the frozen Qwen3.6 draft and TP 1/2. Scope: the checkpoint's integrated
single-path reference (dflash_config.inference_in_block_causal=true), NOT the
upstream parallel-tree JetSpec runtime.

Everything is the spatial DFlash path except three policy hooks: the derived draft
attention config is causal, the query attention metadata returned by the
official expansion is causal=True, and the official base metadata builder is
called directly (stock DFlash's override asserts non-causal). With causal
FlashAttention over [context | 8 block queries], query i sees the whole context
and block tokens 0..i, the training-time in-block causal mask.
"""
from __future__ import annotations

from dataclasses import replace

import contract
import spatial_adapter as spatial

MARKER = "qwen36_jetspec_local_block_v1"
SCOPE = "local shared-block JetSpec reference; upstream parallel-tree runtime is not ported"
_INSTALLED = False


def enabled(config) -> bool:
    cfg = contract.as_dict(config)
    return (cfg.get("dflash_config") or {}).get(MARKER) is True


def validate_config(config) -> dict:
    cfg = contract.validate_draft_config(config)
    dflash = cfg["dflash_config"]
    if dflash.get("inference_in_block_causal") is not True or not enabled(cfg):
        raise ValueError("JetSpec requires inference_in_block_causal=true and its local-block marker")
    for key in ("sample_from_anchor", "shift_label", "projector_type", "layer_routing_enabled"):
        if dflash.get(key) not in (None, False):
            raise ValueError(f"JetSpec local-block export cannot set dflash_config.{key}")
    if cfg.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("JetSpec uses the registered DFlashDraftModel dispatch key")
    return cfg


def install():
    global _INSTALLED
    bindings = spatial.install()
    if _INSTALLED:
        return bindings
    from vllm.v1.spec_decode import dflash, llm_base_proposer
    P = dflash.DFlashProposer
    original_create = P._create_draft_vllm_config
    original_set = P.set_inputs_first_pass
    original_metadata = P.build_per_group_and_layer_attn_metadata
    base_metadata = llm_base_proposer.SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata

    def causal_draft_config(self):
        resolved = original_create(self)
        if not enabled(self.draft_model_config.hf_config):
            return resolved
        validate_config(self.draft_model_config.hf_config)
        return replace(resolved, attention_config=replace(resolved.attention_config, use_non_causal=False))

    def causal_query_inputs(self, *args, **kwargs):
        result = original_set(self, *args, **kwargs)
        if not enabled(self.draft_model_config.hf_config):
            return result
        num_queries, indices, query_cad = result
        return num_queries, indices, replace(query_cad, causal=True)

    def causal_layer_metadata(self, cad, draft_index=0):
        if not enabled(self.draft_model_config.hf_config):
            return original_metadata(self, cad, draft_index)
        if getattr(cad, "causal", None) is not True:
            raise ValueError("JetSpec query metadata must be causal=True before the official builder")
        per_group, per_layer = base_metadata(self, cad, draft_index)
        if not per_layer or any(getattr(m, "causal", None) is not True for m in per_layer.values()):
            raise RuntimeError("JetSpec backend metadata is not causal=True; unsupported backend")
        return per_group, per_layer

    P._create_draft_vllm_config = causal_draft_config
    P.set_inputs_first_pass = causal_query_inputs
    P.build_per_group_and_layer_attn_metadata = causal_layer_metadata
    _INSTALLED = True
    return bindings
