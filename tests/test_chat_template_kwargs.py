"""target.chat_template_kwargs: one render protocol for regen, capture/online and eval."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vlm_dflash.adapters import get_adapter
from vlm_dflash.contract import (
    ContractError,
    contract_chat_template_kwargs,
    contract_defaults_from_config,
    validate_yaml_against_contract,
)
from vlm_dflash.provenance import identity_diff, identity_from_contract, source_identity
from vlm_dflash.target.prompts import CHAT_TEMPLATE_KWARGS_ATTR, render_conversation, render_prompt


class _RecordingProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "rendered"


def test_renders_forward_the_processor_kwargs():
    processor = _RecordingProcessor()
    render_prompt(processor, "<image>What is shown?")
    assert processor.calls[-1] == {"tokenize": False, "add_generation_prompt": True}

    setattr(processor, CHAT_TEMPLATE_KWARGS_ATTR, {"enable_thinking": False})
    render_prompt(processor, "What is shown?")
    render_conversation(processor, [{"role": "user", "content": "hi"}], add_generation_prompt=False)
    assert processor.calls[-2] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert processor.calls[-1] == {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}


def test_finalize_processor_applies_the_contract_kwargs():
    adapter = get_adapter("qwen3_vl")
    processor = _RecordingProcessor()
    adapter.finalize_processor(processor, {"prompt": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert getattr(processor, CHAT_TEMPLATE_KWARGS_ATTR) == {"enable_thinking": False}
    assert processor.tokenizer.padding_side == "left"
    adapter.finalize_processor(processor, {})  # contracts written before the field
    assert getattr(processor, CHAT_TEMPLATE_KWARGS_ATTR) == {}


def test_contract_defaults_and_accessor():
    cfg = {"target": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert contract_defaults_from_config(cfg)["chat_template_kwargs"] == {"enable_thinking": False}
    assert contract_chat_template_kwargs({"prompt": {"chat_template_kwargs": {"x": 1}}}) == {"x": 1}
    assert contract_chat_template_kwargs({}) == {}


def _contract(template_kwargs: dict | None) -> dict:
    contract = {
        "target": {"model_path": "/m", "model_fingerprint": "f", "processor_fingerprint": "p"},
        "mask": {"token_id": 9},
        "layers": {"block_size": 8, "num_draft_layers": 3, "num_target_layers": 12, "draft_layer_ids": [1, 5, 9]},
        "rope": {"sections": [16, 24, 24], "interleaved": False, "mode": "mrope"},
        "image": {"min_pixels": 1, "max_pixels": 2},
    }
    if template_kwargs is not None:
        contract["prompt"] = {"chat_template_kwargs": template_kwargs}
    return contract


def _cfg(template_kwargs: dict | None = None) -> dict:
    cfg = {"drafter": {"mask_token_id": 9, "block_size": 8, "num_hidden_layers": 3}, "target": {}}
    if template_kwargs is not None:
        cfg["target"]["chat_template_kwargs"] = template_kwargs
    return cfg


def test_validation_rejects_a_drifted_render_protocol():
    validate_yaml_against_contract(_cfg({"enable_thinking": False}), _contract({"enable_thinking": False}))
    validate_yaml_against_contract(_cfg(), _contract(None))
    with pytest.raises(ContractError, match="chat_template_kwargs"):
        validate_yaml_against_contract(_cfg(), _contract({"enable_thinking": False}))
    with pytest.raises(ContractError, match="chat_template_kwargs"):
        validate_yaml_against_contract(_cfg({"enable_thinking": True}), _contract({"enable_thinking": False}))


def test_identity_records_the_kwargs_only_when_set():
    plain = identity_from_contract(_contract(None))
    assert "chat_template_kwargs" not in plain
    assert identity_from_contract(_contract({})) == plain  # empty = template defaults
    thinking_off = identity_from_contract(_contract({"enable_thinking": False}))
    assert thinking_off["chat_template_kwargs"] == {"enable_thinking": False}
    # Optional source key: absent stays absent, set vs unset is a mismatch.
    assert "chat_template_kwargs" not in source_identity(plain)
    assert source_identity(thinking_off)["chat_template_kwargs"] == {"enable_thinking": False}
    assert identity_diff(plain, thinking_off) == [
        "chat_template_kwargs: expected=None actual={'enable_thinking': False}"
    ]


def test_contract_kwargs_must_be_a_mapping_without_renderer_owned_names():
    from vlm_dflash.contract import _checked_template_kwargs

    assert _checked_template_kwargs(None) == {}
    assert _checked_template_kwargs({"enable_thinking": False}) == {"enable_thinking": False}
    with pytest.raises(ContractError, match="must be a mapping"):
        _checked_template_kwargs("enable_thinking")
    with pytest.raises(ContractError, match=r"cannot set \['add_generation_prompt', 'tokenize'\]"):
        _checked_template_kwargs({"tokenize": True, "add_generation_prompt": True, "enable_thinking": False})
