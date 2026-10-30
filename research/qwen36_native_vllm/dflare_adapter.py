"""DFlare on the Qwen3.6 target: per-layer fused context with dedicated context K/V.

Port of research/qwen3vl8b_native_vllm/dflare_8b_work_20261029 (same math) to the
frozen Qwen3.6 draft and TP 1/2. Matches vlm_dflash/dflare/model.py
(separate_context_kv=true, no FC):

  context_i  = hidden_norm(sum_t softmax(layer_fusion_weights)[i, t] * raw_t)
  K_i, V_i   = k_proj_target_i(context_i), v_proj_target_i(context_i)   (k_norm, spatial MRoPE)
  queries    = official DFlash layers (own q/k/v for the 8 block tokens), stock greedy head

The raw [C, 3*2048] payload is kept by combine_hidden_states. DFlare has no FC
base, so the tensor it returns (the first layer's slice) only satisfies the
proposer's width check and is never read: the context precompute uses the raw
payload. Context K/V projections are column-parallel over KV heads, like the
query-side k/v projections.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import contract
import spatial_adapter as spatial

ARCH = "DFlashDFlareQwen36ForCausalLM"
MARKER = "qwen36_dflare_v1"
TENSOR_COUNT = 42
HIDDEN = contract.DRAFT_FIELDS["hidden_size"]
HEAD_DIM = contract.DRAFT_FIELDS["head_dim"]
KV_HEADS = contract.DRAFT_FIELDS["num_key_value_heads"]
NUM_CONTEXT_LAYERS = len(contract.TARGET_LAYER_IDS)
RAW_WIDTH = NUM_CONTEXT_LAYERS * HIDDEN
DEDICATED = frozenset(f"layers.{i}.self_attn.{kind}_proj_target.weight" for i in range(3) for kind in ("k", "v"))
DFLARE_FIELDS = {"context_layer_ids": contract.TARGET_LAYER_IDS, "separate_context_kv": True,
                 "fusion_trainable": True}
_INSTALLED = False


def enabled(config) -> bool:
    cfg = contract.as_dict(config)
    return cfg.get("architectures") == [ARCH] and (cfg.get("dflash_config") or {}).get(MARKER) is True


def validate_config(config) -> dict:
    cfg = contract.validate_draft_config(config)
    dflash = cfg["dflash_config"]
    for field, value in DFLARE_FIELDS.items():
        if dflash.get(field) != value:
            raise ValueError(f"DFlare requires dflash_config.{field}={value!r}, got {dflash.get(field)!r}")
    if dflash.get("sample_from_anchor", False):
        raise ValueError("DFlare export requires the DFlash slot convention")
    if not enabled(cfg):
        raise ValueError("Missing DFlare architecture and marker")
    return cfg


def build_layer_contexts(raw: torch.Tensor, fusion_logits: torch.Tensor, hidden_norm) -> tuple[torch.Tensor, ...]:
    """Canonical Qwen3DFlareDraftModel._build_layer_contexts (probabilities cast to the raw dtype)."""
    if raw.ndim != 2 or raw.shape[1] != RAW_WIDTH or tuple(fusion_logits.shape) != (3, NUM_CONTEXT_LAYERS):
        raise ValueError("DFlare requires ordered raw context [C, 6144] and [3, 3] fusion logits")
    stacked = raw.view(raw.shape[0], NUM_CONTEXT_LAYERS, HIDDEN)
    probabilities = F.softmax(fusion_logits, dim=-1).to(raw.dtype)
    return tuple(hidden_norm(torch.einsum("t,ntd->nd", probabilities[i], stacked))
                 for i in range(probabilities.shape[0]))


def _build_classes():
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models import qwen3_dflash as official
    from vllm.model_executor.models.utils import maybe_prefix

    class DFlareDecoderLayer(official.DFlashQwen3DecoderLayer):
        def __init__(self, vllm_config, *, config, cache_config=None, quant_config=None, prefix=""):
            super().__init__(vllm_config, config=config, cache_config=cache_config,
                             quant_config=quant_config, prefix=prefix)
            dtype = vllm_config.model_config.dtype
            for kind in ("k", "v"):
                # Output rows are KV heads; each TP rank holds its own heads, matching
                # the sharded query-side attention and the per-rank KV cache.
                setattr(self.self_attn, kind + "_proj_target", ColumnParallelLinear(
                    HIDDEN, KV_HEADS * HEAD_DIM, bias=False, gather_output=False, params_dtype=dtype,
                    quant_config=None, prefix=f"{prefix}.self_attn.{kind}_proj_target", return_bias=False))

    @support_torch_compile(dynamic_arg_dims=dict(spatial.CORRECTED_METADATA))
    class DFlareBackbone(nn.Module):
        def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
            super().__init__()
            self.config = vllm_config.speculative_config.draft_model_config.hf_config
            validate_config(self.config)
            self.vocab_size = self.config.vocab_size
            self.quant_config = None
            self.use_aux_hidden_state = True
            current = get_current_vllm_config()
            eps = self.config.rms_norm_eps
            self.embed_tokens = official.VocabParallelEmbedding(self.config.vocab_size, HIDDEN,
                                                                prefix=maybe_prefix(prefix, "embed_tokens"))
            self.layers = nn.ModuleList([DFlareDecoderLayer(
                current, config=self.config, cache_config=current.cache_config, quant_config=None,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}")) for i in range(3)])
            self.hidden_norm = official.RMSNorm(HIDDEN, eps=eps)
            self.norm = official.RMSNorm(HIDDEN, eps=eps)
            self.layer_fusion_weights = nn.Parameter(
                torch.empty(3, NUM_CONTEXT_LAYERS, dtype=vllm_config.model_config.dtype), requires_grad=False)
            self.raw_context = None
            self._dummy_raw = None

        embed_input_ids = official.DFlashQwen3Model.embed_input_ids
        forward = official.DFlashQwen3Model.forward

        def load_weights(self, weights):
            state = dict(weights)
            params = dict(self.named_parameters())
            # Keep the dedicated context projections away from the stock .k_proj/.v_proj packer.
            for name in DEDICATED:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, state.pop(name))
            return official.DFlashQwen3Model.load_weights(self, state.items()) | set(DEDICATED)

        def precompute_and_store_context_kv(self, unused_base, context_positions, context_slot_mapping=None):
            n = unused_base.shape[0]
            raw = self.raw_context
            if raw is None or raw.shape[0] != n:
                if context_slot_mapping is not None:
                    raise ValueError("Real DFlare context writes require the raw payload of this proposal")
                if self._dummy_raw is None or self._dummy_raw.shape[0] < n:
                    self._dummy_raw = torch.zeros((max(n, 1), RAW_WIDTH), dtype=unused_base.dtype,
                                                  device=unused_base.device)
                raw = self._dummy_raw[:n]
            contexts = build_layer_contexts(raw, self.layer_fusion_weights, self.hidden_norm)
            keys, values = [], []
            for layer, context in zip(self.layers, contexts):
                attention = layer.self_attn
                key = F.linear(context, attention.k_proj_target.weight)
                value = F.linear(context, attention.v_proj_target.weight)
                keys.append(attention.k_norm(key.reshape(n, attention.num_kv_heads, attention.head_dim)))
                values.append(value.reshape(n, attention.num_kv_heads, attention.head_dim))
            first = self.layers[0].self_attn
            all_k = torch.stack(keys).contiguous()
            all_v = torch.stack(values).contiguous()
            all_k = spatial.rotate_context_keys(first.rotary_emb, context_positions,
                                                all_k.view(3 * n, first.kv_size), 3).view(3, n, first.num_kv_heads,
                                                                                         first.head_dim)
            self.raw_context = None
            if context_slot_mapping is None:
                return
            for i, layer in enumerate(self.layers):
                attn = layer.self_attn.attn
                attn.impl.do_kv_cache_update(attn, all_k[i], all_v[i], attn.kv_cache, context_slot_mapping)

    class DFlashDFlareQwen36ForCausalLM(official.DFlashQwen3ForCausalLM):
        def __init__(self, *, vllm_config, prefix=""):
            nn.Module.__init__(self)
            self.config = vllm_config.speculative_config.draft_model_config.hf_config
            validate_config(self.config)
            self.config.draft_vocab_size = self.config.vocab_size
            layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
            if layers != contract.TARGET_NUM_LAYERS:
                raise ValueError(f"DFlare requires the 40-layer Qwen3.6 target, got {layers}")
            self.config.target_layer_count = layers
            self.model = DFlareBackbone(vllm_config=vllm_config, start_layer_id=layers,
                                        prefix=maybe_prefix(prefix, "model"))
            self.lm_head = official.ParallelLMHead(self.config.vocab_size, HIDDEN,
                                                   prefix=maybe_prefix(prefix, "lm_head"))
            self.logits_processor = official.LogitsProcessor(self.config.vocab_size, scale=1.0)
            self.draft_id_to_target_id = None
            self.loaded_tensor_names = ()
            self.has_own_embed_tokens = False
            self.has_own_lm_head = False

        def combine_hidden_states(self, hidden_states):
            if hidden_states.ndim != 2 or hidden_states.shape[1] != RAW_WIDTH:
                raise ValueError("DFlare requires the ordered raw target payload [C, 6144]")
            self.model.raw_context = hidden_states
            return hidden_states[:, :HIDDEN]

        def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
            self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

        def load_weights(self, weights):
            state = {}
            for name, value in weights:
                if name in state:
                    raise ValueError("Duplicate DFlare tensor: " + name)
                state[name] = value
            if len(state) != TENSOR_COUNT or any("embed_tokens" in k or "lm_head" in k for k in state):
                raise ValueError(f"DFlare export must hold exactly {TENSOR_COUNT} trained tensors")
            self.model.load_weights(state.items())
            self.loaded_tensor_names = tuple(sorted(state))

    return DFlashDFlareQwen36ForCausalLM


def install():
    global _INSTALLED
    bindings = spatial.install()
    if _INSTALLED:
        return bindings
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.spec_decode.dflash import DFlashProposer
    import sys
    sys.modules[__name__].DFlashDFlareQwen36ForCausalLM = _build_classes()
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, f"{__name__}:{ARCH}")
    original_load = DFlashProposer.load_model

    def load_model(self, target_model):
        result = original_load(self, target_model)
        if enabled(self.draft_model_config.hf_config) and len(self.model.loaded_tensor_names) != TENSOR_COUNT:
            raise ValueError("Actual loader must load all 42 DFlare tensors")
        return result

    DFlashProposer.load_model = load_model
    _INSTALLED = True
    return bindings
