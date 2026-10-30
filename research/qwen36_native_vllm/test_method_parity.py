"""CPU parity: native adapter method math vs the training implementation, real trained weights.

The training-side draft models are rebuilt from the finished checkpoints with the
training code, and the target's real LM head and embedding are used. Inputs are
synthetic (seeded) hidden states, so this checks the ported math, not acceptance.

  Q36_RUNS=<task>/runs Q36_TARGET=<Qwen3.6 dir> \
  PYTHONPATH=<training VelaDraft>:research/qwen36_native_vllm python -m unittest test_method_parity
"""
import json
import os
from pathlib import Path
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn

import contract

RUNS = Path(os.environ.get("Q36_RUNS", "."))
TARGET = Path(os.environ.get("Q36_TARGET", "."))
MASK = contract.MASK_TOKEN_ID


def checkpoint(method: str) -> Path:
    train = RUNS / method / "train"
    final = train / "ckpt_final.pt"
    if final.exists():
        return final
    steps = sorted(train.glob("ckpt_step*.pt"))
    if not steps:
        raise unittest.SkipTest(f"no {method} checkpoint")
    return steps[-1]


_TARGET_CACHE = {}


def target_weights():
    if not _TARGET_CACHE:
        from safetensors import safe_open
        index = json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
        wanted = {"lm_head.weight": "lm_head", "model.language_model.embed_tokens.weight": "embed"}
        for name, key in wanted.items():
            with safe_open(str(TARGET / index[name]), framework="pt") as f:
                _TARGET_CACHE[key] = f.get_tensor(name)
    return _TARGET_CACHE["lm_head"], _TARGET_CACHE["embed"]


class Target(nn.Module):
    """Training-side target view: lm_head + input embeddings."""

    def __init__(self):
        super().__init__()
        head, embed = target_weights()
        self.lm_head = nn.Linear(head.shape[1], head.shape[0], bias=False, dtype=head.dtype)
        self.lm_head.weight.data = head
        self.embed = nn.Embedding(embed.shape[0], embed.shape[1], dtype=embed.dtype)
        self.embed.weight.data = embed

    def get_input_embeddings(self):
        return self.embed


class Stub:
    """Adapter-side view of the vLLM draft model (official head returns FP32 of a BF16 GEMM)."""

    def __init__(self, draft, target):
        self.target = target
        for name in ("markov_head", "prefix_gru", "embed_proj", "candidate_selector"):
            if hasattr(draft, name):
                setattr(self, name, getattr(draft, name))
        self.model = draft

    def compute_logits(self, hidden):
        return self.target.lm_head(hidden).float()

    def embed_input_ids(self, ids):
        return self.target.embed(ids)


def load(method):
    module = {"dflash": "vlm_dflash.dflash.factory", "domino": "vlm_dflash.dflash.factory",
              "jetspec": "vlm_dflash.dflash.factory", "dspark": "vlm_dflash.dspark.factory",
              "dflash2": "vlm_dflash.dflash2.factory", "dfly": "vlm_dflash.dfly.factory",
              "dflare": "vlm_dflash.dflare.factory"}[method]
    import importlib
    draft, payload = importlib.import_module(module).load_checkpoint(str(checkpoint(method)))
    if payload["algorithm"] != method:
        raise AssertionError(f"{checkpoint(method)} is {payload['algorithm']}")
    return draft.to(torch.bfloat16).eval()


def block_inputs(draft, seed):
    g = torch.Generator().manual_seed(seed)
    hidden = draft.norm(torch.randn((1, 8, 2048), generator=g).to(torch.bfloat16))
    anchor = torch.randint(0, 248000, (1,), generator=g)
    block = torch.full((1, 8), MASK, dtype=torch.long)
    block[0, 0] = anchor
    return hidden, block, anchor


class MethodParity(unittest.TestCase):
    seeds = range(6)

    @torch.inference_mode()
    def test_dspark_markov_walk(self):
        import head_adapters
        draft, target = load("dspark"), Target()
        for seed in self.seeds:
            hidden, block, anchor = block_inputs(draft, seed)
            expected = draft._sample_draft_tokens(target, hidden, block)[0, :7]
            actual = head_adapters.dspark_walk(Stub(draft, target), hidden[0, :7], anchor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_domino_gru_walk(self):
        import head_adapters
        draft, target = load("domino"), Target()
        for seed in self.seeds:
            hidden, block, anchor = block_inputs(draft, seed)
            expected = draft._sample_draft_tokens(target, hidden, block)[0]
            actual = head_adapters.domino_walk(Stub(draft, target), hidden[0, :7], anchor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_dflash2_selector_walk_and_convolutions(self):
        import dflash2_adapter
        draft, target = load("dflash2"), Target()
        for seed in self.seeds:
            hidden, block, anchor = block_inputs(draft, seed)
            expected = draft._sample_draft_tokens(target, hidden, block)[0]
            actual = dflash2_adapter.selector_walk(Stub(draft, target), hidden[0, 1:], anchor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        g = torch.Generator().manual_seed(11)
        x = torch.randn((8, 2048), generator=g).to(torch.bfloat16)
        for layer in draft.layers:
            for name in ("attention_conv", "mlp_conv"):
                trained = getattr(layer, name)
                ported = dflash2_adapter.GroupedDynamicCausalConv(2048, block_size=8, kernel_size=2, group_size=16,
                                                                  dtype=torch.bfloat16)
                ported.load_state_dict(trained.state_dict())
                (a, ka), (b, kb) = trained.prepare(x), ported.prepare(x)
                torch.testing.assert_close(b, a, rtol=0, atol=0)
                torch.testing.assert_close(ported.finish(x, kb), trained.finish(x, ka), rtol=0, atol=0)

    @torch.inference_mode()
    def test_dfly_context_and_correction_walk(self):
        import dfly_adapter
        draft, target = load("dfly"), Target()
        for seed in self.seeds:
            hidden, block, anchor = block_inputs(draft, seed)
            expected = draft._sample_draft_tokens(target, hidden, block)[0]
            actual = dfly_adapter.sequential_greedy(Stub(draft, target), hidden[0, 1:], anchor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        raw = torch.randn((13, 3, 2048), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        expected = draft._build_layer_contexts(raw[None])
        flat = raw.reshape(13, 6144)
        actual = dfly_adapter.build_layer_contexts(flat, draft.fc(flat), draft.layer_fusion_weights, draft.hidden_norm)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e[0], rtol=0, atol=0)

    @torch.inference_mode()
    def test_dflare_context(self):
        import dflare_adapter
        draft = load("dflare")
        raw = torch.randn((13, 3, 2048), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
        expected = draft._build_layer_contexts(raw[None])
        actual = dflare_adapter.build_layer_contexts(raw.reshape(13, 6144), draft.layer_fusion_weights,
                                                     draft.hidden_norm)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
