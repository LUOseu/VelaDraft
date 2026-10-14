"""Divergence-margin probe v2: frozen-prefix replay + causal gate (CPU fakes)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_divergence_margin", ROOT / "scripts" / "probe_divergence_margin.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


class FakeProcessor:
    def __init__(self, prompt_ids=(1, 2, 3), nondeterministic=False):
        self.prompt_ids = [int(token) for token in prompt_ids]
        self.nondeterministic = nondeterministic
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<fake-rendered-prompt>"

    def __call__(self, text=None, images=None, return_tensors=None):
        self.calls += 1
        scale = float(self.calls) if self.nondeterministic else 1.0
        return {
            "input_ids": torch.tensor([self.prompt_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(self.prompt_ids), dtype=torch.long),
            "pixel_values": torch.full((1, 4), scale),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class FakeTarget(torch.nn.Module):
    """Logits: argmax at position p = fed token p+1, except an optional override
    and a chosen top1/top2 pair at the final position."""

    def __init__(self, prefix, *, final_top1, final_top2, margin, wrong_at=None, wrong_token=99, vocab=128):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.prefix = [int(token) for token in prefix]
        self.final_top1 = int(final_top1)
        self.final_top2 = int(final_top2)
        self.margin = float(margin)
        self.wrong_at = wrong_at
        self.wrong_token = int(wrong_token)
        self.vocab = int(vocab)
        self.calls = 0
        self.seen_input_ids = None

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, **kwargs):
        self.calls += 1
        self.seen_input_ids = input_ids.detach().clone()
        seq = input_ids.shape[1]
        prompt_len = seq - len(self.prefix)
        logits = torch.zeros((1, seq, self.vocab))
        for position in range(seq - 1):
            logits[0, position, int(input_ids[0, position + 1])] = 5.0
        if self.wrong_at is not None:
            position = prompt_len - 1 + self.wrong_at
            logits[0, position, :] = 0.0
            logits[0, position, self.wrong_token] = 5.0
        logits[0, seq - 1, :] = 0.0
        logits[0, seq - 1, self.final_top1] = 5.0
        logits[0, seq - 1, self.final_top2] = 5.0 - self.margin
        return SimpleNamespace(logits=logits)


@pytest.fixture()
def media_dir(tmp_path):
    Image.new("RGB", (4, 4), (200, 30, 30)).save(tmp_path / "img.png")
    return tmp_path


def _rows():
    return [
        {
            "id": "row-0",
            "conversations": [{"from": "human", "value": "<image>What is shown?"}],
            "images": ["img.png"],
        }
    ]


def _bridge(target_tokens, draft_tokens, *, prompt_len=3, per_position=None):
    return {
        "format": "vlm-dflash-vllm-bridge-check-v1",
        "target_only": {
            "records": [
                {
                    "request_id": "req-0",
                    "prompt_len": prompt_len,
                    "token_ids": list(target_tokens),
                    "finish_reason": "length",
                }
            ]
        },
        "dflash": {
            "records": [
                {
                    "request_id": "req-0",
                    "prompt_len": prompt_len,
                    "token_ids": list(draft_tokens),
                    "finish_reason": "length",
                }
            ],
            "spec_decode": {
                "per_position_acceptance": list(per_position or [0.3, 0.1]),
                "acceptance_length": 1.4,
            },
        },
    }


def test_probe_feeds_frozen_prefix_and_reports_margin(media_dir):
    target_tokens = [10, 11, 12, 13, 14]
    draft_tokens = [10, 11, 99, 13, 14]
    model = FakeTarget([10, 11], final_top1=12, final_top2=99, margin=0.125)
    result = probe.run_probe(
        model,
        FakeProcessor(),
        _rows(),
        media_dir,
        _bridge(target_tokens, draft_tokens),
        max_cases=4,
        top_k=5,
    )

    assert model.seen_input_ids is not None, "probe must run a forward"
    assert model.seen_input_ids[0].tolist() == [1, 2, 3, 10, 11], "saved prefix must enter the forward verbatim"
    case = result["cases"][0]
    assert case["divergence"] == 2
    assert case["probe_position"] == 4
    assert case["prefix_consistent"] is True
    assert case["causal_evidence"] is True
    assert case["exclusion_reason"] is None
    assert case["margin_top1_top2"] == pytest.approx(0.125)
    assert case["target_top1"] == 12
    assert case["vllm_dflash_token"] == 99
    assert case["vllm_token_rank"] == 2
    assert case["logit_gap_top1_minus_vllm"] == pytest.approx(0.125)
    assert [candidate["token_id"] for candidate in case["candidates"][:2]] == [12, 99]
    assert result["summary"]["causal_cases"] == [0]
    assert result["summary"]["excluded_cases"] == []
    assert result["format"] == "vlm-dflash-divergence-margin-v2"


def test_prefix_off_trajectory_is_excluded_from_causal_stats(media_dir):
    target_tokens = [10, 11, 12, 13, 14]
    draft_tokens = [10, 11, 99, 13, 14]
    model = FakeTarget([10, 11], final_top1=12, final_top2=99, margin=0.125, wrong_at=0)
    result = probe.run_probe(
        model,
        FakeProcessor(),
        _rows(),
        media_dir,
        _bridge(target_tokens, draft_tokens),
        max_cases=4,
        top_k=5,
    )

    case = result["cases"][0]
    assert case["prefix_consistent"] is False
    assert case["first_prefix_mismatch"] == 0
    assert case["causal_evidence"] is False
    assert case["exclusion_reason"] == "prefix_not_on_model_trajectory"
    assert case["margin_top1_top2"] == pytest.approx(0.125), "observation still recorded"
    assert result["summary"]["causal_cases"] == []
    assert result["summary"]["excluded_cases"] == [
        {"index": 0, "reason": "prefix_not_on_model_trajectory"}
    ]


def test_prompt_len_mismatch_is_rejected_before_forward(media_dir):
    target_tokens = [10, 11, 12, 13, 14]
    draft_tokens = [10, 11, 99, 13, 14]
    model = FakeTarget([10, 11], final_top1=12, final_top2=99, margin=0.125)
    result = probe.run_probe(
        model,
        FakeProcessor(),
        _rows(),
        media_dir,
        _bridge(target_tokens, draft_tokens, prompt_len=99),
        max_cases=4,
        top_k=5,
    )

    case = result["cases"][0]
    assert case["causal_evidence"] is False
    assert case["exclusion_reason"] == "prompt_len_mismatch"
    assert case["candidates"] is None
    assert model.calls == 0, "mismatched inputs must not be forwarded"
    assert result["summary"]["causal_cases"] == []


def test_nondeterministic_preprocessing_is_rejected(media_dir):
    target_tokens = [10, 11, 12, 13, 14]
    draft_tokens = [10, 11, 99, 13, 14]
    model = FakeTarget([10, 11], final_top1=12, final_top2=99, margin=0.125)
    result = probe.run_probe(
        model,
        FakeProcessor(nondeterministic=True),
        _rows(),
        media_dir,
        _bridge(target_tokens, draft_tokens),
        max_cases=4,
        top_k=5,
    )

    case = result["cases"][0]
    assert case["input_preprocessing_deterministic"] is False
    assert case["exclusion_reason"] == "input_preprocessing_not_deterministic"
    assert case["causal_evidence"] is False
    assert model.calls == 0


def test_unobservable_evidence_is_marked_missing(media_dir):
    target_tokens = [10, 11, 12, 13, 14]
    draft_tokens = [10, 11, 99, 13, 14]
    model = FakeTarget([10, 11], final_top1=12, final_top2=99, margin=0.125)
    result = probe.run_probe(
        model,
        FakeProcessor(),
        _rows(),
        media_dir,
        _bridge(target_tokens, draft_tokens),
        max_cases=4,
        top_k=5,
    )

    case = result["cases"][0]
    assert case["evidence"]["same_engine_verify_logits"].startswith("missing")
    assert case["evidence"]["kv_cache_state"].startswith("missing")
    assert case["evidence"]["vllm_processor_inputs"].startswith("missing")
    assert "fed verbatim" in case["evidence"]["frozen_prefix_tokens"]
    assert result["summary"]["attribution"]["root_cause"] == "unresolved"
    assert result["summary"]["attribution"]["level"] == "hf-assisted-diagnostic"
    assert result["same_engine_diagnostics"]["status"] == "missing"
    assert result["summary"]["run_level_per_position_acceptance"] == [0.3, 0.1]


def test_select_cases_skips_exact_and_handles_shorter_draft():
    target_records = [
        {"request_id": "a", "prompt_len": 3, "token_ids": [1, 2, 3]},
        {"request_id": "b", "prompt_len": 3, "token_ids": [4, 5, 6]},
        {"request_id": "c", "prompt_len": 3, "token_ids": [7, 8, 9]},
    ]
    draft_records = [
        {"request_id": "a", "prompt_len": 3, "token_ids": [1, 2, 3]},
        {"request_id": "b", "prompt_len": 3, "token_ids": [4, 99, 6]},
        {"request_id": "c", "prompt_len": 3, "token_ids": [7, 8]},
    ]
    cases = probe.select_cases(target_records, draft_records, max_cases=10)
    assert [(case["index"], case["divergence"]) for case in cases] == [(1, 1), (2, 2)]
    assert cases[0]["prefix"] == [4]
    assert cases[0]["dflash_token"] == 99
    assert cases[1]["dflash_token"] is None, "shorter draft: no emitted token at divergence"
    assert probe.select_cases(target_records, draft_records, max_cases=1) == [cases[0]]


def test_bridge_without_dflash_phase_raises(media_dir):
    bridge = _bridge([1, 2, 3], [1, 2, 3])
    bridge.pop("dflash")
    with pytest.raises(ValueError, match="no dflash phase"):
        probe.run_probe(
            FakeTarget([1], final_top1=1, final_top2=2, margin=0.1),
            FakeProcessor(),
            _rows(),
            media_dir,
            bridge,
        )
