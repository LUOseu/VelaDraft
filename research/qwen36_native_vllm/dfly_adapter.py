"""DFly on the Qwen3.6 target: FC base context + per-layer fusion residual + hidden correction.

Port of research/qwen3vl8b_native_vllm/dfly_recovery_v2 (same math) to the frozen
Qwen3.6 draft and TP 1/2. Matches vlm_dflash/dfly/model.py:

  base       = fc(raw)                    raw = target layers [1,19,37] concatenated, [C, 3*2048]
  context_i  = hidden_norm(base + sum_t softmax(layer_fusion_weights)[i, t] * raw_t)
  K_i, V_i   = shared DFlash k/v projection of context_i (k_norm, spatial MRoPE)
  slot k     : h_k' = h_k + SwiGLU(norm(h_k) :: norm(embed(token_{k-1}))), greedy argmax,
               token_0 = the verifier's bonus token (sequential, real previous tokens)

Instead of copying the official propose (as the 8B adapter did), the raw payload is
kept by combine_hidden_states, which returns the unchanged official FC base; the
context precompute consumes both, and only _sample_draft_tokens is replaced.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import contract
import spatial_adapter as spatial

ARCH = "DFlashDFlyQwen36ForCausalLM"
MARKER = "qwen36_dfly_v1"
TENSOR_COUNT = 42
HIDDEN = contract.DRAFT_FIELDS["hidden_size"]
NUM_CONTEXT_LAYERS = len(contract.TARGET_LAYER_IDS)
RAW_WIDTH = NUM_CONTEXT_LAYERS * HIDDEN
CORRECTION_KEYS = frozenset("hidden_correction." + name + ".weight"
                            for name in ("hidden_norm", "embed_norm", "gate_proj", "up_proj", "down_proj"))
DFLY_FIELDS = {"context_layer_ids": contract.TARGET_LAYER_IDS, "separate_context_kv": False,
               "sample_from_anchor": False, "fusion_trainable": True, "enable_hidden_correction": True,
               "hidden_correction_intermediate_size": None}
_INSTALLED = False


def enabled(config) -> bool:
    cfg = contract.as_dict(config)
    return cfg.get("architectures") == [ARCH] and (cfg.get("dflash_config") or {}).get(MARKER) is True


def validate_config(config) -> dict:
    cfg = contract.validate_draft_config(config)
    dflash = cfg["dflash_config"]
    for field, value in DFLY_FIELDS.items():
        if dflash.get(field) != value:
            raise ValueError(f"DFly requires dflash_config.{field}={value!r}, got {dflash.get(field)!r}")
    if not enabled(cfg):
        raise ValueError("Missing DFly architecture and marker")
    return cfg


def build_layer_contexts(raw: torch.Tensor, base: torch.Tensor, fusion_logits: torch.Tensor,
                         hidden_norm) -> tuple[torch.Tensor, ...]:
    """Canonical cast and addition order of Qwen3DFlyDraftModel._build_layer_contexts."""
    if raw.ndim != 2 or raw.shape[1] != RAW_WIDTH or tuple(fusion_logits.shape) != (3, NUM_CONTEXT_LAYERS):
        raise ValueError("DFly requires ordered raw context [C, 6144] and [3, 3] fusion logits")
    stacked = raw.view(raw.shape[0], NUM_CONTEXT_LAYERS, HIDDEN).to(base.dtype)
    probabilities = F.softmax(fusion_logits, dim=-1).to(base.dtype)
    return tuple(hidden_norm(base + torch.einsum("t,ntd->nd", probabilities[i], stacked))
                 for i in range(probabilities.shape[0]))


class HiddenStatesCorrection(nn.Module):
    """Two-norm, three-matrix SwiGLU residual (replicated on every TP rank)."""

    def __init__(self, hidden_size, intermediate_size, *, dtype, rms_norm_eps, prefix=""):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import ReplicatedLinear
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.embed_norm = RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        for name, (in_size, out_size) in (("gate_proj", (2 * hidden_size, intermediate_size)),
                                          ("up_proj", (2 * hidden_size, intermediate_size)),
                                          ("down_proj", (intermediate_size, hidden_size))):
            setattr(self, name, ReplicatedLinear(in_size, out_size, bias=False, params_dtype=dtype,
                                                 quant_config=None, prefix=f"{prefix}.{name}",
                                                 return_bias=False, disable_tp=True))

    def forward(self, hidden_states, prev_token_embeds):
        joined = torch.cat([self.hidden_norm(hidden_states),
                            self.embed_norm(prev_token_embeds.to(hidden_states.dtype))], dim=-1)
        delta = self.down_proj(F.silu(self.gate_proj(joined)) * self.up_proj(joined))
        return hidden_states + delta.to(hidden_states.dtype)


def _build_classes():
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models import qwen3_dflash as official
    from vllm.model_executor.models.utils import maybe_prefix

    @support_torch_compile(dynamic_arg_dims=dict(spatial.CORRECTED_METADATA))
    class DFlyBackbone(nn.Module):
        def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
            super().__init__()
            self.config = vllm_config.speculative_config.draft_model_config.hf_config
            validate_config(self.config)
            self.vocab_size = self.config.vocab_size
            self.quant_config = None
            self.use_aux_hidden_state = True
            dtype = vllm_config.model_config.dtype
            current = get_current_vllm_config()
            eps = self.config.rms_norm_eps
            self.embed_tokens = official.VocabParallelEmbedding(self.config.vocab_size, HIDDEN,
                                                                prefix=maybe_prefix(prefix, "embed_tokens"))
            self.layers = nn.ModuleList([official.DFlashQwen3DecoderLayer(
                current, config=self.config, cache_config=current.cache_config, quant_config=None,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}")) for i in range(3)])
            self.fc = official.ReplicatedLinear(RAW_WIDTH, HIDDEN, bias=False, params_dtype=dtype, quant_config=None,
                                                prefix=maybe_prefix(prefix, "fc"), return_bias=False, disable_tp=True)
            self.hidden_norm = official.RMSNorm(HIDDEN, eps=eps)
            self.norm = official.RMSNorm(HIDDEN, eps=eps)
            self.layer_fusion_weights = nn.Parameter(torch.empty(3, NUM_CONTEXT_LAYERS, dtype=dtype),
                                                     requires_grad=False)
            self.hidden_correction = HiddenStatesCorrection(HIDDEN, HIDDEN, dtype=dtype, rms_norm_eps=eps,
                                                            prefix=maybe_prefix(prefix, "hidden_correction"))
            self.raw_context = None
            self._dummy_raw = None

        embed_input_ids = official.DFlashQwen3Model.embed_input_ids
        forward = official.DFlashQwen3Model.forward

        def load_weights(self, weights):
            state = dict(weights)
            params = dict(self.named_parameters())
            # Separate gate/up correction matrices: keep them away from the stock
            # substring packer, which would route them into a nonexistent gate_up.
            for name in CORRECTION_KEYS:
                default_weight_loader(params[name], state.pop(name))
            return official.DFlashQwen3Model.load_weights(self, state.items()) | set(CORRECTION_KEYS)

        def precompute_and_store_context_kv(self, base, context_positions, context_slot_mapping=None):
            n = base.shape[0]
            raw = self.raw_context
            if raw is None or raw.shape[0] != n:
                if context_slot_mapping is not None:
                    raise ValueError("Real DFly context writes require the raw payload of this proposal")
                if self._dummy_raw is None or self._dummy_raw.shape[0] < n:
                    self._dummy_raw = torch.zeros((max(n, 1), RAW_WIDTH), dtype=base.dtype, device=base.device)
                raw = self._dummy_raw[:n]
            contexts = build_layer_contexts(raw, base, self.layer_fusion_weights, self.hidden_norm)
            keys, values = [], []
            for layer, context in zip(self.layers, contexts):
                attention = layer.self_attn
                weight = attention.qkv_proj.weight[attention.q_size:]
                bias = attention.qkv_proj.bias
                bias = None if bias is None else bias[attention.q_size:]
                key, value = F.linear(context, weight, bias).split([attention.kv_size, attention.kv_size], dim=-1)
                key = attention.k_norm(key.reshape(n, attention.num_kv_heads, attention.head_dim))
                keys.append(key)
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

    class DFlashDFlyQwen36ForCausalLM(official.DFlashQwen3ForCausalLM):
        def __init__(self, *, vllm_config, prefix=""):
            nn.Module.__init__(self)
            self.config = vllm_config.speculative_config.draft_model_config.hf_config
            validate_config(self.config)
            self.config.draft_vocab_size = self.config.vocab_size
            layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
            if layers != contract.TARGET_NUM_LAYERS:
                raise ValueError(f"DFly requires the 40-layer Qwen3.6 target, got {layers}")
            self.config.target_layer_count = layers
            self.model = DFlyBackbone(vllm_config=vllm_config, start_layer_id=layers,
                                      prefix=maybe_prefix(prefix, "model"))
            self.lm_head = official.ParallelLMHead(self.config.vocab_size, HIDDEN,
                                                   prefix=maybe_prefix(prefix, "lm_head"))
            self.logits_processor = official.LogitsProcessor(self.config.vocab_size, scale=1.0)
            self.draft_id_to_target_id = None
            self.loaded_tensor_names = ()
            # The checkpoint has neither: the proposer shares the target's own objects.
            self.has_own_embed_tokens = False
            self.has_own_lm_head = False

        def combine_hidden_states(self, hidden_states):
            if hidden_states.ndim != 2 or hidden_states.shape[1] != RAW_WIDTH:
                raise ValueError("DFly requires the ordered raw target payload [C, 6144]")
            self.model.raw_context = hidden_states
            return self.model.fc(hidden_states)

        def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
            self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

        def load_weights(self, weights):
            state = {}
            for name, value in weights:
                if name in state:
                    raise ValueError("Duplicate DFly tensor: " + name)
                state[name] = value
            if len(state) != TENSOR_COUNT or any("embed_tokens" in k or "lm_head" in k for k in state):
                raise ValueError(f"DFly export must hold exactly {TENSOR_COUNT} trained tensors")
            self.model.load_weights(state.items())
            self.loaded_tensor_names = tuple(sorted(state))

    return DFlashDFlyQwen36ForCausalLM


def sequential_greedy(model, suffix_hidden: torch.Tensor, bonus: torch.Tensor) -> torch.Tensor:
    """Qwen3DFlyDraftModel._sample_draft_tokens for one request, greedy."""
    if suffix_hidden.ndim != 2 or suffix_hidden.shape[0] != contract.SPECULATIVE_TOKENS or bonus.shape != (1,):
        raise ValueError("DFly requires seven suffix rows and one bonus anchor")
    previous = bonus
    selected = []
    for i in range(contract.SPECULATIVE_TOKENS):
        embeds = model.embed_input_ids(previous).to(suffix_hidden.dtype)
        corrected = model.model.hidden_correction(suffix_hidden[i:i + 1], embeds)
        previous = model.compute_logits(corrected).argmax(dim=-1)
        selected.append(previous)
    return torch.cat(selected)


def install():
    global _INSTALLED
    bindings = spatial.install()
    if _INSTALLED:
        return bindings
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.spec_decode.dflash import DFlashProposer
    import sys
    module = sys.modules[__name__]
    module.DFlashDFlyQwen36ForCausalLM = _build_classes()
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, f"{__name__}:{ARCH}")
    original_set = DFlashProposer.set_inputs_first_pass
    original_sample = DFlashProposer._sample_draft_tokens
    original_load = DFlashProposer.load_model

    def set_inputs(self, target_token_ids, next_token_ids, *args, **kwargs):
        if enabled(self.draft_model_config.hf_config):
            if next_token_ids.shape != (1,):
                raise ValueError("DFly requires batch 1 with one bonus anchor")
            self._dfly_anchor = next_token_ids
        return original_set(self, target_token_ids, next_token_ids, *args, **kwargs)

    def sample(self, hidden_states, sampling_metadata):
        if not enabled(self.draft_model_config.hf_config):
            return original_sample(self, hidden_states, sampling_metadata)
        if not sampling_metadata.all_greedy:
            raise ValueError("DFly native mount requires greedy decoding")
        anchor, self._dfly_anchor = self._dfly_anchor, None
        return sequential_greedy(self.model, hidden_states, anchor), None

    def load_model(self, target_model):
        result = original_load(self, target_model)
        if enabled(self.draft_model_config.hf_config) and len(self.model.loaded_tensor_names) != TENSOR_COUNT:
            raise ValueError("Actual loader must load all 42 DFly tensors")
        return result

    DFlashProposer.set_inputs_first_pass = set_inputs
    DFlashProposer._sample_draft_tokens = sample
    DFlashProposer.load_model = load_model
    _INSTALLED = True
    return bindings
