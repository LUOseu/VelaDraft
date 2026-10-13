"""WP6: SmolVLM2 adapter - the first non-Qwen (1-D position, tile-encoded) family.

Everything here runs on CPU against a synthetic SmolVLM2-shaped model dir that
transformers loads for real (SmolVLMConfig + SmolVLMProcessor). The tests pin
the family contract: 1-D rope resolves with ``sections=None``, image tokens
count as tiles x ``image_seq_len``, the mask token is a reserved special row,
and the batch capture/eval plumbing consumes ``pixel_values`` +
``pixel_attention_mask`` instead of the Qwen grid tensors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.adapters import adapter_for_model, get_adapter
from vlm_dflash.adapters.smolvlm import SmolVLMAdapter
from vlm_dflash.contract import ContractError, resolve_contract

IMAGE_SEQ_LEN = 81
RESERVED_ID = 12
IMAGE_TOKEN_ID = 5


@pytest.fixture(scope="module")
def fake_target(tmp_path_factory):
    """A synthetic SmolVLM2-shaped checkpoint directory (no weights needed)."""
    from tokenizers import Tokenizer, models
    from transformers import (
        PreTrainedTokenizerFast,
        SmolVLMConfig,
        SmolVLMImageProcessor,
        SmolVLMProcessor,
        SmolVLMVideoProcessor,
    )

    directory = tmp_path_factory.mktemp("smolvlm-target")
    base_vocab = {"<unk>": 0, "hello": 1, "world": 2, "a": 3, "photo": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab=base_vocab, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<image>",
                "<fake_token_around_image>",
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                *(f"<|reserved_special_token_{index}|>" for index in range(4)),
            ]
        }
    )
    assert tokenizer.convert_tokens_to_ids("<image>") == IMAGE_TOKEN_ID
    assert tokenizer.convert_tokens_to_ids("<|reserved_special_token_2|>") == RESERVED_ID

    text_config = dict(
        model_type="llama",
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config = SmolVLMConfig(
        text_config=text_config,
        vision_config=dict(
            model_type="smolvlm_vision",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=512,
            patch_size=16,
        ),
    )
    config.image_token_id = IMAGE_TOKEN_ID
    config.save_pretrained(directory)

    processor = SmolVLMProcessor(
        image_processor=SmolVLMImageProcessor(
            size={"longest_edge": 1536}, max_image_size={"longest_edge": 384}, do_image_splitting=True
        ),
        tokenizer=tokenizer,
        video_processor=SmolVLMVideoProcessor(),
        image_seq_len=IMAGE_SEQ_LEN,
    )
    processor.save_pretrained(directory)
    # The published SmolVLM2 repos carry a legacy preprocessor_config.json next
    # to processor_config.json; the adapter resolves pixel bounds from it.
    processor_config = json.loads((directory / "processor_config.json").read_text(encoding="utf-8"))
    (directory / "preprocessor_config.json").write_text(
        json.dumps(processor_config["image_processor"], indent=2), encoding="utf-8"
    )
    return directory


def test_fake_target_resolves_1d_tiles_reserved_mask(fake_target):
    contract = resolve_contract(
        fake_target, {"mask_token_id": RESERVED_ID, "draft_num_hidden_layers": 2, "block_size": 4}
    )
    assert contract["rope"]["mode"] == "1d"
    assert contract["rope"]["sections"] is None
    assert contract["rope"]["interleaved"] is False
    assert contract["image"]["token_count_mode"] == "tiles"
    assert contract["image"]["sequence_length_per_tile"] == IMAGE_SEQ_LEN
    assert contract["image"]["merge_size"] is None
    assert contract["mask"]["token_id"] == RESERVED_ID
    assert "reserved special-token row" in contract["mask"]["policy"]
    assert contract["tokens"]["image_token_id"] == IMAGE_TOKEN_ID


def test_contract_rejects_yaml_mrope_defaults_for_1d_family(fake_target):
    with pytest.raises(ContractError, match="without mrope_section"):
        resolve_contract(
            fake_target,
            {"mask_token_id": RESERVED_ID, "spatial_rope_sections": [8, 8, 8]},
        )
    with pytest.raises(ContractError, match="without MRoPE"):
        resolve_contract(
            fake_target,
            {"mask_token_id": RESERVED_ID, "spatial_rope_interleaved": True},
        )


def test_contract_rejects_real_tokenizer_token_as_mask(fake_target):
    with pytest.raises(ContractError, match="reserved_special_token"):
        resolve_contract(fake_target, {"mask_token_id": IMAGE_TOKEN_ID})


def test_contract_rejects_mask_outside_embedding_range(fake_target):
    with pytest.raises(ContractError, match="outside embedding range"):
        resolve_contract(fake_target, {"mask_token_id": 999})


def test_expected_image_token_counts_are_tiles_times_fixed_length():
    adapter = SmolVLMAdapter()
    pixel_values = torch.zeros(1, 17, 3, 384, 384)
    assert adapter.expected_image_token_counts(
        {"pixel_values": pixel_values}, merge_size=None
    ) == [17 * IMAGE_SEQ_LEN]
    single = torch.zeros(1, 1, 3, 384, 384)
    assert adapter.expected_image_token_counts({"pixel_values": single}, merge_size=None) == [IMAGE_SEQ_LEN]
    assert adapter.expected_image_token_counts({}, merge_size=None) is None


def test_capture_forward_inputs_concats_and_rejects_mismatches():
    adapter = SmolVLMAdapter()
    rows = [
        {
            "inputs": {
                "pixel_values": torch.zeros(1, 5, 3, 384, 384),
                "pixel_attention_mask": torch.ones(1, 5, 384, 384, dtype=torch.long),
            }
        },
        {
            "inputs": {
                "pixel_values": torch.zeros(1, 5, 3, 384, 384),
                "pixel_attention_mask": torch.ones(1, 5, 384, 384, dtype=torch.long),
            }
        },
    ]
    kwargs = adapter.capture_forward_inputs(rows, device=torch.device("cpu"))
    assert kwargs["pixel_values"].shape == (2, 5, 3, 384, 384)
    assert kwargs["pixel_attention_mask"].shape == (2, 5, 384, 384)

    text_only = [{"inputs": {}}, {"inputs": {}}]
    assert adapter.capture_forward_inputs(text_only, device=torch.device("cpu")) == {}

    mixed = [rows[0], {"inputs": {}}]
    with pytest.raises(ValueError, match="mixes rows"):
        adapter.capture_forward_inputs(mixed, device=torch.device("cpu"))

    skewed = [rows[0], {"inputs": {**rows[0]["inputs"], "pixel_values": torch.zeros(1, 3, 3, 384, 384)}}]
    with pytest.raises(ValueError, match="disagree on pixel_values shapes"):
        adapter.capture_forward_inputs(skewed, device=torch.device("cpu"))


def test_multimodal_positions_are_three_identical_1d_axes():
    adapter = SmolVLMAdapter()
    positions = adapter.compute_multimodal_positions(
        object(), input_ids=torch.zeros(1, 7, dtype=torch.long), attention_mask=None, image_grid_thw=None
    )
    assert positions.shape == (3, 7)
    assert torch.equal(positions[0], torch.arange(7))
    assert torch.equal(positions[0], positions[1]) and torch.equal(positions[1], positions[2])


def test_forward_kwargs_from_inputs_uses_tile_tensors():
    adapter = SmolVLMAdapter()
    inputs = {
        "pixel_values": torch.zeros(2, 5, 3, 384, 384),
        "pixel_attention_mask": torch.ones(2, 5, 384, 384, dtype=torch.long),
        "attention_mask": torch.ones(2, 40, dtype=torch.long),
        "mm_token_type_ids": None,
    }
    kwargs = adapter.forward_kwargs_from_inputs(inputs)
    assert set(kwargs) == {"pixel_values", "pixel_attention_mask", "attention_mask"}


def test_draft_config_carries_no_spatial_rope_sections(fake_target):
    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.dflash.factory import build_draft_config

    contract = resolve_contract(
        fake_target, {"mask_token_id": RESERVED_ID, "draft_num_hidden_layers": 2, "block_size": 4}
    )
    cfg = {
        "drafter": {
            "num_hidden_layers": 2,
            "block_size": 4,
            "mask_token_id": RESERVED_ID,
            "spatial_rope_enabled": False,
        }
    }
    draft_config = build_draft_config(contract, cfg)
    assert draft_config.dflash_config["spatial_rope_sections"] is None
    assert draft_config.dflash_config["spatial_rope_enabled"] is False
    draft = Qwen3DFlashDraftModel(draft_config)
    assert draft.spatial_rope_sections == ()
    assert draft.spatial_rope_enabled is False


def test_adapter_registry_and_model_dispatch():
    adapter = get_adapter("smolvlm")
    assert isinstance(adapter, SmolVLMAdapter)
    assert adapter.info.position_mode == "1d"
    assert adapter.info.image_token_mode == "tiles"
    assert "pixel_attention_mask" in adapter.info.forward_input_keys
    assert not any("mm_token_type_ids" in key for key in adapter.info.forward_input_keys)

    class _FakeInner:
        compute_3d_position_ids = None

    class _FakeModel:
        model = _FakeInner()
        config = type("C", (), {"model_type": "smolvlm"})()

    assert isinstance(adapter_for_model(_FakeModel()), SmolVLMAdapter)
