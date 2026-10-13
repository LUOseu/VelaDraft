"""Feature-contract tests: capture-layer layout, subset selection, identity gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.contract import (
    ContractError,
    contract_defaults_from_config,
    contract_capture_layer_ids,
    validate_capture_layer_ids,
)
from vlm_dflash.dflash.features import capture_layer_layout, select_capture_layers
from vlm_dflash.provenance import IdentityMismatchError, assert_identity, identity_from_contract


def test_contract_capture_layer_ids_legacy_fallback():
    legacy = {"layers": {"draft_layer_ids": [1, 17, 33]}}
    assert contract_capture_layer_ids(legacy) == [1, 17, 33]
    explicit = {"layers": {"draft_layer_ids": [1, 17, 33], "capture_layer_ids": [1, 5, 9, 13]}}
    assert contract_capture_layer_ids(explicit) == [1, 5, 9, 13]


def test_validate_capture_layer_ids():
    assert validate_capture_layer_ids([1, 5, 9], 36, "test") == [1, 5, 9]
    with pytest.raises(ContractError):
        validate_capture_layer_ids([], 36, "test")
    with pytest.raises(ContractError):
        validate_capture_layer_ids([5, 1], 36, "test")
    with pytest.raises(ContractError):
        validate_capture_layer_ids([1, 1], 36, "test")
    with pytest.raises(ContractError):
        validate_capture_layer_ids([1, 36], 36, "test")


def test_contract_defaults_extract_algorithm_context_layers():
    cfg = {
        "drafter": {"capture_layer_ids": [1, 9, 17, 25, 33]},
        "algorithm": {"name": "dflare", "dflare": {"context_layer_ids": [1, 17, 33]}},
    }
    defaults = contract_defaults_from_config(cfg)
    assert defaults["algorithm_context_layer_ids"] == [1, 17, 33]
    # DFlash configs (plain string algorithm) keep the routing as the consumer.
    assert contract_defaults_from_config({"algorithm": "dflash"})["algorithm_context_layer_ids"] is None


def test_capture_layer_layout_legacy_and_explicit():
    legacy = {"draft_layer_ids": [1, 17, 33]}
    assert capture_layer_layout(legacy, contract_hidden_size=2048) == ([1, 17, 33], 2048)
    explicit = {"capture_layer_ids": [1, 5, 9], "hidden_size": 2048}
    assert capture_layer_layout(explicit) == ([1, 5, 9], 2048)
    with pytest.raises(ValueError):
        capture_layer_layout({})


def _stacked(hidden_size: int, layers: list[int]):
    # One distinguishable constant per layer so ordering mistakes are visible.
    return torch.cat(
        [torch.full((1, 4, hidden_size), float(layer + 1)) for layer in layers], dim=-1
    )


def test_select_capture_layers_identity_and_subset_order():
    hidden = _stacked(8, [1, 5, 9, 13])
    assert torch.equal(select_capture_layers(hidden, [1, 5, 9, 13], [1, 5, 9, 13]), hidden)
    # Reordered subset: channels must follow the requested order, not the store's.
    reordered = select_capture_layers(hidden, [1, 5, 9, 13], [9, 1])
    assert torch.equal(reordered[:, :, :8], torch.full((1, 4, 8), 10.0))
    assert torch.equal(reordered[:, :, 8:], torch.full((1, 4, 8), 2.0))


def test_select_capture_layers_missing_layer_fails_with_hint():
    hidden = _stacked(8, [1, 5])
    with pytest.raises(ValueError, match="Recapture"):
        select_capture_layers(hidden, [1, 5], [1, 9])
    with pytest.raises(ValueError, match="duplicates"):
        select_capture_layers(hidden, [1, 1], [1])
    with pytest.raises(ValueError, match="not divisible"):
        select_capture_layers(torch.zeros(1, 4, 17), [1, 5], [1])


def test_identity_gate_includes_capture_layers():
    contract = {
        "target": {"model_path": "/m", "model_fingerprint": "f", "processor_fingerprint": "p"},
        "mask": {"token_id": 7},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"draft_layer_ids": [1, 17, 33]},
        "image": {"min_pixels": 1, "max_pixels": 2},
    }
    identity = identity_from_contract(contract)
    # default capture == routing: a legacy record (no field) stays comparable
    legacy = {key: value for key, value in identity.items() if key != "capture_layer_ids"}
    assert_identity(identity, legacy, "test")
    # a superset capture list is a different feature contract
    superset = dict(identity)
    superset["capture_layer_ids"] = [1, 5, 17, 33]
    with pytest.raises(IdentityMismatchError):
        assert_identity(identity, superset, "test")

