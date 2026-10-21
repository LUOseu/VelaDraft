"""Controlled proposals through the real decoder, target, and KV implementation.

The oracle only replaces proposal selection. Both verifier modes and both models'
forwards execute normally; this measures decoder correctness, not draft quality.
"""
from __future__ import annotations

import types

import pytest
import torch
from transformers.cache_utils import DynamicCache

from tests.test_strict_parallel import _tiny_draft, _tiny_target
from vlm_dflash.dflash import draft_model


@torch.no_grad()
def _run(monkeypatch, accepted, sequential, limit=17, eos=False, eos_offset=2):
    target, draft = _tiny_target(), _tiny_draft().eval()
    prompt = torch.tensor([[3, 7, 11, 17, 23, 29]])
    oracle = target.generate(prompt, max_new_tokens=40, do_sample=False,
                             eos_token_id=None, pad_token_id=0)
    captures = []

    class TracedCache(DynamicCache):
        def crop(self, maximum_length):
            super().crop(maximum_length)
            # The first constructed cache belongs to the target.
            if self is captures[0]:
                snapshots.append((self.get_seq_length(), [
                    (layer.keys.clone(), layer.values.clone()) for layer in self.layers
                ]))

        def __init__(self):
            super().__init__()
            captures.append(self)

    snapshots, starts = [], []
    monkeypatch.setattr(draft_model, "DynamicCache", TracedCache)

    def record_start(module, args, kwargs):
        starts.append(int(kwargs["position_ids"][0, -draft.block_size]))

    hook = draft.register_forward_pre_hook(record_start, with_kwargs=True)

    def proposal(self, target, draft_hidden, block_output_ids):
        start = starts[-1]
        tokens = oracle[:, start + 1:start + self.block_size].clone()
        if accepted < self.block_size - 1:
            tokens[:, accepted] = (tokens[:, accepted] + 1) % 254
        return tokens

    draft._sample_draft_tokens = types.MethodType(proposal, draft)
    stop = [int(oracle[0, prompt.shape[1] + eos_offset])] if eos else None
    result, stats = draft.spec_generate(
        target=target, input_ids=prompt, max_new_tokens=limit,
        stop_token_ids=stop, temperature=0, return_stats=True,
        sequential_target_verification=sequential,
    )
    hook.remove()
    expected = oracle[:, :prompt.shape[1] + limit]
    if stop:
        first = (expected[0, prompt.shape[1]:] == stop[0]).nonzero()[0].item()
        expected = expected[:, :prompt.shape[1] + first + 1]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # Every committed target prefix must equal an independent full-prefix
    # forward, including after rejected speculative KV rows were cropped.
    assert snapshots
    for length, layers in snapshots:
        reference = target(oracle[:, :length], use_cache=True).past_key_values
        assert len(layers) == len(reference.layers)
        for (keys, values), reference_layer in zip(layers, reference.layers):
            torch.testing.assert_close(keys, reference_layer.keys, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(values, reference_layer.values, rtol=1e-4, atol=1e-6)
    assert [event["generated_position"] for event in stats["acceptance_events"]] == [
        start - prompt.shape[1] + 1 for start in starts
    ]
    return stats, starts, prompt.shape[1]


@pytest.mark.parametrize("sequential", [True, False], ids=["sequential", "parallel"])
@pytest.mark.parametrize("accepted", [0, 2, 7], ids=["reject_all", "partial", "full"])
def test_controlled_acceptance_and_kv_commit(monkeypatch, accepted, sequential):
    stats, starts, prompt_len = _run(monkeypatch, accepted, sequential)
    assert all(b - a == accepted + 1 for a, b in zip(starts, starts[1:]))
    events = stats["acceptance_events"]
    assert events[0]["accepted_tokens"] == accepted
    assert stats["accepted_tokens"] == sum(e["accepted_tokens"] for e in events)
    assert stats["verify_steps"] == len(events)
    assert stats["accepted_tokens"] == sum(min(accepted, max(0, 17 - (s - prompt_len) - 1)) for s in starts)


@pytest.mark.parametrize("sequential", [True, False], ids=["sequential", "parallel"])
@pytest.mark.parametrize("limit", [1, 3, 8, 9])
def test_full_acceptance_tail_clips_output_and_counters(monkeypatch, sequential, limit):
    stats, _, _ = _run(monkeypatch, 7, sequential, limit=limit)
    assert stats["accepted_tokens"] == limit - (limit + 7) // 8
    assert stats["draft_tokens"] == stats["accepted_tokens"]


@pytest.mark.parametrize("sequential", [True, False], ids=["sequential", "parallel"])
@pytest.mark.parametrize("accepted", [0, 2, 7])
def test_eos_truncates_accepted_or_corrected_output(monkeypatch, sequential, accepted):
    _run(monkeypatch, accepted, sequential, eos=True)


@pytest.mark.parametrize("sequential", [True, False], ids=["sequential", "parallel"])
@pytest.mark.parametrize("eos_offset", [0, 1, 2, 7])
def test_eos_bounds_acceptance_metrics_to_emitted_suffix(monkeypatch, sequential, eos_offset):
    stats, _, _ = _run(monkeypatch, 7, sequential, eos=True, eos_offset=eos_offset)
    # One target anchor followed by only the draft tokens through EOS.
    assert stats["accepted_tokens"] == eos_offset
    assert stats["draft_tokens"] == eos_offset
    assert stats["acceptance_events"][0]["accepted_tokens"] == eos_offset
