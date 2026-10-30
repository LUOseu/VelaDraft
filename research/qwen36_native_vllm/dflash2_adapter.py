"""DFlash2 on the Qwen3.6 target: block-local dynamic convolutions + candidate selector.

Port of research/qwen3vl8b_native_vllm/dflash2_recovery_v2 (same math; the conv and
selector code is vendored from Z-Lab DFlash2 via speculators@55ebbd5b, MIT) to the
frozen Qwen3.6 draft and TP 1/2. Matches vlm_dflash/dflash2/model.py:

  layer:   x = norm(x); x, k = attention_conv.prepare(x); x = attn(x); x = attention_conv.finish(x, k)
           x = norm(x); x, k = mlp_conv.prepare(x);       x = mlp(x);  x = mlp_conv.finish(x, k)
  sample:  rows 1..7, unary = lm_head(h) top-16, score += <codebook_pred[prev] * P h, codebook_succ[cand]>,
           sequential with the real previous token (anchor first), greedy

The convolutions read and write full-width hidden states (sublayer input after the
norm, sublayer output after the row-parallel reduction), so they are replicated on
every TP rank; so is the selector. They are attached to the stock backbone's
decoder layers before the first compile.
"""
from __future__ import annotations

import types

import torch
from torch import nn

import contract
import spatial_adapter as spatial

ARCH = "DFlash2Qwen36ForCausalLM"
MARKER = "qwen36_dflash2_v1"
TENSOR_COUNT = 51
HIDDEN = contract.DRAFT_FIELDS["hidden_size"]
VOCAB = contract.DRAFT_FIELDS["vocab_size"]
OPTIONS = {"conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256, "selector_top_k": 16}
_INSTALLED = False


def enabled(config) -> bool:
    cfg = contract.as_dict(config)
    return cfg.get("architectures") == [ARCH] and (cfg.get("dflash_config") or {}).get(MARKER) is True


def validate_config(config) -> dict:
    cfg = contract.validate_draft_config(config)
    d = cfg["dflash_config"]
    options = d.get("dflash2") or {}
    if any(options.get(k) != v for k, v in OPTIONS.items()):
        raise ValueError(f"DFlash2 requires dflash_config.dflash2 {OPTIONS}, got {options}")
    if d.get("sample_from_anchor") or d.get("projector_type") is not None:
        raise ValueError("DFlash2 follows the DFlash slot convention without a projector")
    if not enabled(cfg):
        raise ValueError("Missing DFlash2 architecture and marker")
    return cfg


def grouped_dynamic_conv(hidden_states, delta_kernel, base_kernel, *, block_size, group_size):
    """Token-conditioned grouped causal convolution within draft blocks (Z-Lab DFlash2)."""
    hidden_size = hidden_states.shape[-1]
    kernel_size = base_kernel.shape[0]
    num_groups = hidden_size // group_size
    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, num_groups, group_size)
    flat_delta = delta_kernel.reshape(-1, kernel_size, num_groups)
    positions = torch.arange(flat_hidden.shape[0], device=hidden_states.device).remainder(block_size)
    output = torch.zeros_like(flat_hidden)
    for tap in range(kernel_size):
        if tap == 0:
            shifted = flat_hidden
        else:
            padding = flat_hidden.new_zeros(tap, num_groups, group_size)
            shifted = torch.cat([padding, flat_hidden[:-tap]], dim=0)
        coefficient = base_kernel[tap].to(flat_hidden.dtype).view(1, num_groups, group_size) + flat_delta[:, tap, :, None]
        valid = positions.ge(tap).view(-1, 1, 1)
        output = output + shifted * coefficient * valid
    return output.reshape(original_shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, *, block_size, kernel_size, group_size, dtype):
        super().__init__()
        self.block_size, self.kernel_size, self.group_size = block_size, kernel_size, group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size, dtype=dtype), requires_grad=False)
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * self.num_groups, bias=False, dtype=dtype)

    def prepare(self, hidden_states):
        kernels = self.kernel_projection(hidden_states).view(*hidden_states.shape[:-1], 2, self.kernel_size,
                                                            self.num_groups)
        prepared = grouped_dynamic_conv(hidden_states, kernels[..., 0, :, :], self.base_kernel[0],
                                        block_size=self.block_size, group_size=self.group_size)
        return prepared, kernels[..., 1, :, :]

    def finish(self, hidden_states, delta_kernel):
        return grouped_dynamic_conv(hidden_states, delta_kernel, self.base_kernel[1],
                                    block_size=self.block_size, group_size=self.group_size)


def decoder_forward(self, positions, hidden_states, residual):
    """Official deferred-residual DFlash layer with the four DFlash2 convolutions."""
    if residual is not None:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    else:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    hidden_states, kernel = self.attention_conv.prepare(hidden_states)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    hidden_states = self.attention_conv.finish(hidden_states, kernel)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states, kernel = self.mlp_conv.prepare(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.mlp_conv.finish(hidden_states, kernel)
    return hidden_states, residual


class CandidateSelector(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.top_k = OPTIONS["selector_top_k"]
        rank = OPTIONS["selector_rank"]
        self.predecessor_codebook = nn.Parameter(torch.empty(VOCAB, rank, dtype=dtype), requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(VOCAB, rank, dtype=dtype), requires_grad=False)
        self.hidden_projection = nn.Linear(HIDDEN, rank, bias=False, dtype=dtype)

    def select(self, unary_logits, hidden_states, predecessor_ids):
        candidate_ids = unary_logits.topk(self.top_k, dim=-1).indices
        context = self.predecessor_codebook[predecessor_ids.long()] * self.hidden_projection(hidden_states).to(
            self.predecessor_codebook.dtype)
        transition = (context.unsqueeze(-2) * self.successor_codebook[candidate_ids.long()]).sum(dim=-1)
        unary = unary_logits.gather(-1, candidate_ids)
        return candidate_ids, unary + transition.to(unary.dtype)


def selector_walk(model, hidden: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Qwen3DFlash2DraftModel._sample_draft_tokens for one request (rows 1..7), greedy."""
    unary = model.compute_logits(hidden).to(hidden.dtype)
    predecessor = anchor.long()
    selected = []
    for k in range(hidden.shape[0]):
        ids, scores = model.candidate_selector.select(unary[k:k + 1], hidden[k:k + 1], predecessor)
        predecessor = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        selected.append(predecessor)
    return torch.cat(selected)


def _build_classes():
    from vllm.model_executor.models import qwen3_dflash as official

    class DFlash2Qwen36ForCausalLM(official.DFlashQwen3ForCausalLM):
        def __init__(self, *, vllm_config, prefix=""):
            validate_config(vllm_config.speculative_config.draft_model_config.hf_config)
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if self.config.target_layer_count != contract.TARGET_NUM_LAYERS:
                raise ValueError("DFlash2 requires the 40-layer Qwen3.6 target")
            dtype = vllm_config.model_config.dtype
            for layer in self.model.layers:
                for name in ("attention_conv", "mlp_conv"):
                    setattr(layer, name, GroupedDynamicCausalConv(
                        HIDDEN, block_size=contract.BLOCK_SIZE, kernel_size=OPTIONS["conv_kernel_size"],
                        group_size=OPTIONS["conv_group_size"], dtype=dtype))
                layer.forward = types.MethodType(decoder_forward, layer)
            self.candidate_selector = CandidateSelector(dtype)
            self.loaded_tensor_names = ()

        def load_weights(self, weights):
            state = {}
            for name, value in weights:
                if name in state:
                    raise ValueError("Duplicate DFlash2 tensor: " + name)
                state[name] = value
            if len(state) != TENSOR_COUNT or any("embed_tokens" in k or "lm_head" in k for k in state):
                raise ValueError(f"DFlash2 export must hold exactly {TENSOR_COUNT} trained tensors")
            selector = {k: state.pop(k) for k in list(state) if k.startswith("candidate_selector.")}
            # Conv tensors go through the stock backbone loader (no packed-name collisions).
            super().load_weights(state.items())
            params = dict(self.named_parameters())
            for name, value in selector.items():
                if params[name].shape != value.shape or params[name].dtype != value.dtype:
                    raise ValueError("DFlash2 selector tensor layout differs: " + name)
                with torch.no_grad():
                    params[name].copy_(value)
            missing = [n for n in (f"model.layers.{i}.{c}.{p}" for i in range(3)
                                   for c in ("attention_conv", "mlp_conv")
                                   for p in ("base_kernel", "kernel_projection.weight")) if n not in params]
            if missing:
                raise ValueError(f"DFlash2 conv parameters missing: {missing}")
            self.loaded_tensor_names = tuple(sorted(state) + sorted(selector))

    return DFlash2Qwen36ForCausalLM


def install():
    global _INSTALLED
    bindings = spatial.install()
    if _INSTALLED:
        return bindings
    import sys
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.spec_decode.dflash import DFlashProposer
    sys.modules[__name__].DFlash2Qwen36ForCausalLM = _build_classes()
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, f"{__name__}:{ARCH}")
    original_set = DFlashProposer.set_inputs_first_pass
    original_sample = DFlashProposer._sample_draft_tokens
    original_load = DFlashProposer.load_model

    def set_inputs(self, target_token_ids, next_token_ids, *args, **kwargs):
        if enabled(self.draft_model_config.hf_config):
            if next_token_ids.shape != (1,):
                raise ValueError("DFlash2 requires batch 1 with one bonus anchor")
            self._dflash2_anchor = next_token_ids
        return original_set(self, target_token_ids, next_token_ids, *args, **kwargs)

    def sample(self, hidden_states, sampling_metadata):
        if not enabled(self.draft_model_config.hf_config):
            return original_sample(self, hidden_states, sampling_metadata)
        if not sampling_metadata.all_greedy or hidden_states.shape[0] != contract.SPECULATIVE_TOKENS:
            raise ValueError("DFlash2 native mount requires greedy decoding over rows 1..7")
        anchor, self._dflash2_anchor = self._dflash2_anchor, None
        return selector_walk(self.model, hidden_states, anchor), None

    def load_model(self, target_model):
        result = original_load(self, target_model)
        if enabled(self.draft_model_config.hf_config) and len(self.model.loaded_tensor_names) != TENSOR_COUNT:
            raise ValueError("Actual loader must load all 51 DFlash2 tensors")
        return result

    DFlashProposer.set_inputs_first_pass = set_inputs
    DFlashProposer._sample_draft_tokens = sample
    DFlashProposer.load_model = load_model
    _INSTALLED = True
    return bindings
