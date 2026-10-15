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


def _state(step=600, **provenance_overrides):
    state = {
        "format": "vlm-dflash-train-state-v1",
        "step": step,
        "model": {"w": torch.ones(2) * step},
        "optimizer": {"state": {}},
        "torch_rng_state": torch.tensor([1], dtype=torch.uint8),
        # train-states merge the same checkpoint_common header as init drafts;
        # the converter refuses pairings whose provenance disagrees.
        "algorithm": "dflash",
        "identity": {"target": "qwen2.5-vl-3b"},
        "draft_config": {"hidden_size": 8},
        "feature_mode": "offline",
        "training_plan_digest": "abc",
    }
    state.update(provenance_overrides)
    return state


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


def test_conversion_refuses_plan_digest_mismatch():
    with pytest.raises(ValueError, match="training_plan_digest"):
        convert_train_state(_template(), _state(training_plan_digest="other-plan"))


def test_conversion_refuses_data_identity_mismatch():
    with pytest.raises(ValueError, match="identity"):
        convert_train_state(_template(), _state(identity={"target": "other-store"}))


def test_conversion_refuses_feature_mode_mismatch():
    with pytest.raises(ValueError, match="feature_mode"):
        convert_train_state(_template(), _state(feature_mode="online"))


def test_conversion_refuses_missing_state_provenance():
    state = _state()
    del state["training_plan_digest"]
    with pytest.raises(ValueError, match="missing from train-state"):
        convert_train_state(_template(), state)


def test_conversion_refuses_one_sided_optional_provenance():
    state = _state(capture_layer_ids=[1, 17, 33])
    with pytest.raises(ValueError, match="present only in train-state"):
        convert_train_state(_template(), state)
