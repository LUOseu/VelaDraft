"""DSpark and Domino on the Qwen3.6 target: stock DFlash backbone + trained in-block heads.

Both checkpoints use the anchor-slot convention (DSpark sample_from_anchor=true,
Domino shift_label=true): block row k (k = 0..6, row 0 = the verifier's bonus
token) drafts token anchor+1+k, so the proposer samples rows 0..6 instead of the
stock rows 1..7, then walks the block sequentially with the real previous token:

DSpark (vlm_dflash/dspark, vanilla Markov head, fixed block, S7):
  logits_k = lm_head(h_k) + markov_w2(markov_w1(token_{k-1}))      token_{-1} = anchor
  The confidence head is trained but unused by fixed-length S7 drafting; its
  tensors are loaded and preserved.

Domino (vlm_dflash/dflash/draft_model.py, projector_type=domino, pure_draft_prefix_len=1):
  g_k      = GRU(embed(token_{k-1}), g_{k-1})                       token_{-1} = anchor, g = 0
  logits_k = lm_head(h_k) + [k >= 1] * embed_proj(h_k :: g_k)

Head logits are recast to the hidden dtype (the LogitsProcessor's FP32 output wraps
a BF16 GEMM, so this is exact) and biases are added in BF16, as in training.
Extra heads are replicated on every TP rank; the target head/embedding are shared.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import contract
import spatial_adapter as spatial

HIDDEN = contract.DRAFT_FIELDS["hidden_size"]
VOCAB = contract.DRAFT_FIELDS["vocab_size"]
METHODS = {
    # name -> (architecture, marker, trained tensor count, extra tensor prefixes)
    "dspark": ("DFlashDSparkQwen36ForCausalLM", "qwen36_dspark_v1", 40, ("markov_head.", "confidence_head.")),
    "domino": ("DFlashDominoQwen36ForCausalLM", "qwen36_domino_v1", 40, ("prefix_gru.", "embed_proj.")),
}
_INSTALLED = False


def method_of(config) -> str | None:
    cfg = contract.as_dict(config)
    for name, (arch, marker, _, _) in METHODS.items():
        if cfg.get("architectures") == [arch] and (cfg.get("dflash_config") or {}).get(marker) is True:
            return name
    return None


def validate_config(config, method: str) -> dict:
    cfg = contract.validate_draft_config(config)
    d = cfg["dflash_config"]
    if method == "dspark":
        expected = {"markov_rank": 256, "markov_head_type": "vanilla", "enable_confidence_head": True,
                    "confidence_head_with_markov": True, "sample_from_anchor": True}
    else:
        expected = {"projector_type": "domino", "emb_dim": 256, "gru_hidden_dim": 1024,
                    "pure_draft_prefix_len": 1, "shift_label": True}
    for key, value in expected.items():
        if d.get(key) != value:
            raise ValueError(f"{method} requires dflash_config.{key}={value!r}, got {d.get(key)!r}")
    if method_of(cfg) != method:
        raise ValueError(f"Missing {method} architecture and marker")
    return cfg


class MarkovHead(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.markov_w1 = nn.Embedding(VOCAB, 256, dtype=dtype)
        self.markov_w2 = nn.Linear(256, VOCAB, bias=False, dtype=dtype)


class ConfidenceHead(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = nn.Linear(HIDDEN + 256, 1, dtype=dtype)


def dspark_walk(model, hidden: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """VanillaMarkov.sample_block_tokens, greedy, rows 0..6."""
    base = model.compute_logits(hidden).to(hidden.dtype)
    head = model.markov_head
    previous = anchor.long()
    selected = []
    for k in range(hidden.shape[0]):
        step = base[k:k + 1] + head.markov_w2(head.markov_w1(previous))
        previous = step.argmax(dim=-1)
        selected.append(previous)
    return torch.cat(selected)


def domino_gru_step(embedding, state, weight_ih, weight_hh):
    """Bias-free GRU cell of Qwen3DFlashDraftModel._domino_gru_step."""
    if state is None:
        state = embedding.new_zeros((embedding.shape[0], weight_hh.shape[1]))
    input_reset, input_update, input_new = F.linear(embedding, weight_ih).chunk(3, dim=-1)
    hidden_reset, hidden_update, hidden_new = F.linear(state, weight_hh).chunk(3, dim=-1)
    reset_gate = torch.sigmoid(input_reset + hidden_reset)
    update_gate = torch.sigmoid(input_update + hidden_update)
    new_gate = torch.tanh(input_new + reset_gate * hidden_new)
    return new_gate + update_gate * (state - new_gate)


def domino_walk(model, hidden: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Qwen3DFlashDraftModel._sample_draft_tokens (domino, shift_label, prefix 1), greedy, rows 0..6."""
    base = model.compute_logits(hidden).to(hidden.dtype)
    gru, proj = model.prefix_gru, model.embed_proj
    previous = anchor.long()
    state = None
    selected = []
    for k in range(hidden.shape[0]):
        state = domino_gru_step(model.embed_input_ids(previous).to(hidden.dtype), state,
                                gru.weight_ih_l0, gru.weight_hh_l0)
        step = base[k:k + 1]
        if k >= 1:
            step = step + proj(torch.cat((hidden[k:k + 1], state), dim=-1))
        previous = step.argmax(dim=-1)
        selected.append(previous)
    return torch.cat(selected)


WALKS = {"dspark": dspark_walk, "domino": domino_walk}


def _build_classes():
    from vllm.model_executor.models import qwen3_dflash as official

    def make(method):
        arch, _, count, prefixes = METHODS[method]

        class HeadModel(official.DFlashQwen3ForCausalLM):
            def __init__(self, *, vllm_config, prefix=""):
                config = vllm_config.speculative_config.draft_model_config.hf_config
                validate_config(config, method)
                super().__init__(vllm_config=vllm_config, prefix=prefix)
                if self.config.target_layer_count != contract.TARGET_NUM_LAYERS:
                    raise ValueError("Qwen3.6 heads require the 40-layer target")
                dtype = vllm_config.model_config.dtype
                if method == "dspark":
                    self.markov_head = MarkovHead(dtype)
                    self.confidence_head = ConfidenceHead(dtype)
                else:
                    self.prefix_gru = nn.GRU(input_size=HIDDEN, hidden_size=1024, num_layers=1, batch_first=True,
                                             bias=False, dtype=dtype)
                    self.embed_proj = nn.Sequential(nn.Linear(HIDDEN + 1024, 256, bias=False, dtype=dtype), nn.SiLU(),
                                                    nn.Linear(256, VOCAB, bias=False, dtype=dtype))
                self.loaded_tensor_names = ()

            def load_weights(self, weights):
                state = {}
                for name, value in weights:
                    if name in state:
                        raise ValueError(f"Duplicate {method} tensor: {name}")
                    state[name] = value
                if len(state) != count or any("embed_tokens" in k or "lm_head" in k for k in state):
                    raise ValueError(f"{method} export must hold exactly {count} trained tensors")
                extras = {k: state.pop(k) for k in list(state) if k.startswith(prefixes)}
                super().load_weights(state.items())
                params = dict(self.named_parameters())
                for name, value in extras.items():
                    param = params[name]
                    if param.shape != value.shape or param.dtype != value.dtype:
                        raise ValueError(f"{method} head tensor layout differs: {name}")
                    with torch.no_grad():
                        param.copy_(value)
                self.loaded_tensor_names = tuple(sorted(state) + sorted(extras))

        HeadModel.__name__ = HeadModel.__qualname__ = arch
        return HeadModel

    return {method: make(method) for method in METHODS}


def install():
    global _INSTALLED
    bindings = spatial.install()
    if _INSTALLED:
        return bindings
    import sys
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.spec_decode.dflash import DFlashProposer
    module = sys.modules[__name__]
    for method, cls in _build_classes().items():
        arch = METHODS[method][0]
        setattr(module, arch, cls)
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, f"{__name__}:{arch}")
    original_set = DFlashProposer.set_inputs_first_pass
    original_sample = DFlashProposer._sample_draft_tokens
    original_load = DFlashProposer.load_model

    def set_inputs(self, target_token_ids, next_token_ids, *args, **kwargs):
        method = method_of(self.draft_model_config.hf_config)
        result = original_set(self, target_token_ids, next_token_ids, *args, **kwargs)
        if method is None:
            return result
        if next_token_ids.shape != (1,):
            raise ValueError(f"{method} requires batch 1 with one bonus anchor")
        self._head_anchor = next_token_ids
        num_queries, _, cad = result
        # Anchor-slot convention: rows 0..6 of the single 8-row block.
        rows = torch.arange(contract.SPECULATIVE_TOKENS, dtype=torch.int32, device=next_token_ids.device)
        return num_queries, rows, cad

    def sample(self, hidden_states, sampling_metadata):
        method = method_of(self.draft_model_config.hf_config)
        if method is None:
            return original_sample(self, hidden_states, sampling_metadata)
        if not sampling_metadata.all_greedy:
            raise ValueError(f"{method} native mount requires greedy decoding")
        if hidden_states.shape[0] != contract.SPECULATIVE_TOKENS:
            raise ValueError(f"{method} requires seven anchor-slot rows")
        anchor, self._head_anchor = self._head_anchor, None
        return WALKS[method](self.model, hidden_states, anchor), None

    def load_model(self, target_model):
        result = original_load(self, target_model)
        method = method_of(self.draft_model_config.hf_config)
        if method is not None and len(self.model.loaded_tensor_names) != METHODS[method][2]:
            raise ValueError(f"Actual loader must load every {method} tensor")
        return result

    DFlashProposer.set_inputs_first_pass = set_inputs
    DFlashProposer._sample_draft_tokens = sample
    DFlashProposer.load_model = load_model
    _INSTALLED = True
    return bindings
