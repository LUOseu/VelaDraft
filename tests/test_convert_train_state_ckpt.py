import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.convert_train_state_ckpt import convert_train_state


def _template():
    return {
        "format": "vlm-dflash-ckpt-v1",
        "step": 0,
        "model": {"w": torch.zeros(2)},
        "draft_config": {"hidden_size": 8},
        "algorithm": "dflash",
        "identity": {"target": "qwen2.5-vl-3b"},
        "feature_mode": "offline",
        "training_plan_digest": "abc",
    }


def _state(step=600):
    return {
        "format": "vlm-dflash-train-state-v1",
        "step": step,
        "model": {"w": torch.ones(2) * step},
        "optimizer": {"state": {}},
        "torch_rng_state": torch.tensor([1], dtype=torch.uint8),
    }


def test_conversion_grafts_weights_and_step():
    out = convert_train_state(_template(), _state(600), state_name="ckpt_600.pt")
    assert out["format"] == "vlm-dflash-ckpt-v1"
    assert out["step"] == 600
    assert torch.equal(out["model"]["w"], torch.ones(2) * 600)
    assert out["draft_config"] == {"hidden_size": 8}
    assert out["algorithm"] == "dflash"
    assert out["feature_mode"] == "offline"
    assert out["training_plan_digest"] == "abc"
    assert out["converted_from"]["train_state"] == "ckpt_600.pt"
    assert "optimizer" not in out
    assert "torch_rng_state" not in out


def test_conversion_refuses_wrong_formats():
    with pytest.raises(ValueError):
        convert_train_state({"format": "nope"}, _state())
    with pytest.raises(ValueError):
        convert_train_state(_template(), {"format": "vlm-dflash-train-state-v1"})
