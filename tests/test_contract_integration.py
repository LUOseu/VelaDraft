"""Integration contract tests against the real target model directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.contract import ContractError, resolve_contract

MODEL = os.environ.get("VLM_DFLASH_TEST_MODEL", "/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct")

pytestmark = pytest.mark.skipif(
    not Path(MODEL).exists(), reason="real target model not available (set VLM_DFLASH_TEST_MODEL)"
)

DEFAULTS = {
    "mask_token_id": 151669,
    "spatial_rope_sections": [16, 24, 24],
    "draft_num_hidden_layers": 3,
    "block_size": 8,
    "image_min_pixels": 200704,
    "image_max_pixels": 1003520,
}


def test_contract_resolves_qwen25vl3b():
    contract = resolve_contract(MODEL, DEFAULTS)
    assert contract["mask"]["token_id"] == 151669
    assert contract["mask"]["token_str"] is None
    assert contract["rope"]["sections"] == [16, 24, 24]
    assert contract["layers"]["draft_layer_ids"] == [1, 17, 33]
    assert contract["tokens"]["image_token_id"] == 151655
    assert contract["image"]["min_tokens"] == 256


QWEN3_MODEL = os.environ.get("VLM_DFLASH_TEST_MODEL_QWEN3VL", "/home/lyh/vlm-dflash/assets/models/Qwen3-VL-8B-Instruct")

QWEN3_DEFAULTS = {
    "mask_token_id": 151669,
    "spatial_rope_sections": [24, 20, 20],
    "spatial_rope_interleaved": True,
    "draft_num_hidden_layers": 3,
    "block_size": 8,
}


@pytest.mark.skipif(not Path(QWEN3_MODEL).exists(), reason="Qwen3-VL target model not available")
def test_contract_resolves_qwen3vl8b():
    contract = resolve_contract(QWEN3_MODEL, QWEN3_DEFAULTS)
    assert contract["target"]["model_type"] == "qwen3_vl"
    assert contract["mask"]["token_id"] == 151669
    assert contract["rope"]["sections"] == [24, 20, 20]
    assert contract["rope"]["interleaved"] is True
    assert contract["layers"]["draft_layer_ids"] == [1, 17, 33]
    assert contract["tokens"]["image_token_id"] == 151655
    # Qwen3-VL uses a size.{shortest_edge,longest_edge} processor layout with 16px patches
    assert contract["image"]["patch_size"] == 16
    assert "shortest_edge" in contract["image"]["processor_bounds_source"]
    assert contract["image"]["min_pixels"] == 65536


@pytest.mark.skipif(not Path(QWEN3_MODEL).exists(), reason="Qwen3-VL target model not available")
def test_qwen3_contract_rejects_chunked_rope_default():
    defaults = dict(QWEN3_DEFAULTS, spatial_rope_interleaved=False)
    with pytest.raises(ContractError):
        resolve_contract(QWEN3_MODEL, defaults)


def test_contract_rejects_wrong_sections():
    defaults = dict(DEFAULTS, spatial_rope_sections=[8, 32, 24])
    with pytest.raises(ContractError):
        resolve_contract(MODEL, defaults)


def test_contract_rejects_tokenizer_token_as_mask():
    defaults = dict(DEFAULTS, mask_token_id=151645)  # <|im_end|>
    with pytest.raises(ContractError):
        resolve_contract(MODEL, defaults)
