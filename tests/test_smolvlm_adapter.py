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


@pytest.mark.parametrize("patches_per_side", [4, 27])
def test_npu_vision_positions_preserve_bf16_bucket_boundaries(monkeypatch, patches_per_side):
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionEmbeddings
    from vlm_dflash.adapters.smolvlm import _npu_vision_embeddings_forward

    size = patches_per_side * 7
    config = SmolVLMVisionConfig(hidden_size=16, image_size=size, patch_size=7)
    embeddings = SmolVLMVisionEmbeddings(config).to(torch.bfloat16)
    pixels = torch.randn(2, 3, size, size, dtype=torch.bfloat16)
    mask = torch.ones(2, patches_per_side, patches_per_side, dtype=torch.bool)
    mask[1, -1, :] = False
    mask[1, :, -2:] = False
    reference = embeddings(pixels, mask)
    original = torch.bucketize
    calls = []

    def bucketize(values, boundaries, **kwargs):
        assert values.dtype == torch.float32
        calls.append(values.dtype)
        return original(values, boundaries, **kwargs)

    monkeypatch.setattr(torch, "bucketize", bucketize)
    actual = _npu_vision_embeddings_forward(embeddings, pixels, mask)
    assert len(calls) == 2
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_npu_vision_adaptation_preserves_accelerate_hooks(monkeypatch):
    from accelerate.hooks import ModelHook, add_hook_to_module
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionEmbeddings
    from vlm_dflash.adapters.base import VLMAdapter

    embeddings = SmolVLMVisionEmbeddings(SmolVLMVisionConfig(hidden_size=16, image_size=28, patch_size=7))
    calls = []

    class Hook(ModelHook):
        def pre_forward(self, module, *args, **kwargs):
            calls.append("hook")
            return args, kwargs

    add_hook_to_module(embeddings, Hook())
    model = torch.nn.Module()
    model.add_module("embeddings", embeddings)
    monkeypatch.setattr(VLMAdapter, "load_model", lambda *a, **k: model)
    SmolVLMAdapter().load_model("unused", backend_hint="npu")
    embeddings(torch.randn(1, 3, 28, 28), torch.ones(1, 4, 4, dtype=torch.bool))
    assert calls == ["hook"]
from vlm_dflash.contract import ContractError, ensure_contract, resolve_contract

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


def test_demo_alignment_lets_a_smolvlm_export_pass_the_contract(fake_target, tmp_path):
    """R-22 regression: the demo must adopt the drafter's non-Qwen drafting geometry.

    A fresh demo config carries Qwen family defaults (mask token 151669,
    spatial rope enabled); ``ensure_contract`` rejects those against a SmolVLM
    target. After ``align_config_with_drafter`` adopts the export's recorded
    values (SmolVLM mask row, spatial rope off), the same call must clear the
    gate.
    """
    from vlm_dflash.demo import align_config_with_drafter, build_demo_config

    trained = resolve_contract(
        fake_target, {"mask_token_id": RESERVED_ID, "draft_num_hidden_layers": 2, "block_size": 4}
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "config.json").write_text(
        json.dumps(
            {
                "format": "vlm-dflash-export-v1",
                "method": "dflash",
                "identity": {
                    "model_fingerprint": trained["target"]["model_fingerprint"],
                    "image_min_pixels": trained["image"]["min_pixels"],
                    "image_max_pixels": trained["image"]["max_pixels"],
                },
                "drafter": {
                    "mask_token_id": RESERVED_ID,
                    "block_size": 4,
                    "num_hidden_layers": 2,
                    "spatial_rope": {"enabled": False, "mode": "replace", "sections": None},
                },
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "model.safetensors").write_bytes(b"")

    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()
    cfg = build_demo_config(
        target=str(fake_target),
        algorithm="dflash",
        run_dir=run_dir,
        pool_path=tmp_path / "pool.jsonl",
        images_dir=tmp_path / "images",
        num_train=2,
        num_val=2,
        num_prompts=2,
        max_new_tokens=16,
        steps=0,
        source_desc="unit",
    )
    with pytest.raises(ContractError, match="outside embedding range"):
        ensure_contract(cfg, run_dir)

    align_config_with_drafter(cfg, "dflash", export_dir)
    contract = ensure_contract(cfg, run_dir)
    assert contract["mask"]["token_id"] == RESERVED_ID
    assert contract["rope"]["mode"] == "1d"
    assert contract["layers"]["num_draft_layers"] == 2
    assert contract["layers"]["block_size"] == 4


def test_expected_image_token_counts_are_tiles_times_fixed_length():
    adapter = SmolVLMAdapter()
    pixel_values = torch.zeros(1, 17, 3, 384, 384)
    assert adapter.expected_image_token_counts(
        {"pixel_values": pixel_values}, merge_size=None
    ) == [17 * IMAGE_SEQ_LEN]
    single = torch.zeros(1, 1, 3, 384, 384)
    assert adapter.expected_image_token_counts({"pixel_values": single}, merge_size=None) == [IMAGE_SEQ_LEN]
    assert adapter.expected_image_token_counts({}, merge_size=None) is None


@pytest.mark.parametrize("image_size,patch_size,scale_factor,expected", [(384, 14, 3, 81), (512, 16, 4, 64)])
def test_tile_length_comes_from_the_loaded_model(image_size, patch_size, scale_factor, expected):
    from types import SimpleNamespace
    config = SimpleNamespace(
        vision_config=SimpleNamespace(image_size=image_size, patch_size=patch_size),
        scale_factor=scale_factor,
    )
    adapter = SmolVLMAdapter()
    assert adapter.expected_image_token_counts(
        {"pixel_values": torch.empty(1, 17, 3, 1, 1)}, merge_size=None, model_config=config
    ) == [17 * expected]


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


def test_capture_batch_clamps_out_of_range_config_pad_id():
    """SmolVLM2's config pad id (128002) is outside its 49280-row embedding table.

    Batched capture pads input_ids to the group max; the filler must stay
    inside the embedding range or the target forward raises on lookup.
    """
    from types import SimpleNamespace

    from vlm_dflash.target.sample_capture import forward_and_capture_batch

    class _FakeSmolVLM:
        config = SimpleNamespace(model_type="smolvlm", pad_token_id=128002)

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(50, 4))

        def parameters(self):
            return iter([torch.zeros(1)])

        def __call__(self, **kwargs):
            input_ids = kwargs["input_ids"]
            assert int(input_ids.min()) >= 0 and int(input_ids.max()) < 50
            batch, length = input_ids.shape
            hidden = [torch.zeros(batch, length, 8, dtype=torch.bfloat16) for _ in range(3)]
            return SimpleNamespace(hidden_states=hidden)

    prepared = [
        {"inputs": {}, "full_ids": torch.tensor([1, 3, 3, 4, 5, 6]), "allow_multi_span": False},
        {"inputs": {}, "full_ids": torch.tensor([1, 3, 3, 4]), "allow_multi_span": False},
    ]
    records = forward_and_capture_batch(
        _FakeSmolVLM(), prepared, [1], image_token_id=3, dtype=torch.float32, merge_size=None
    )
    assert records[0]["input_ids"].tolist() == [1, 3, 3, 4, 5, 6]
    assert records[1]["input_ids"].tolist() == [1, 3, 3, 4]
    assert records[0]["image_span"] == (1, 2)
    assert records[0]["position3"].shape == (3, 6)


def test_batch_processor_images_nests_per_text():
    adapter = SmolVLMAdapter()
    images = ["a", "b", "c"]
    assert adapter.batch_processor_images(images) == [["a"], ["b"], ["c"]]
    # Qwen-VL keeps the flat list (its processor accepts it).
    assert get_adapter("qwen2_5_vl").batch_processor_images(images) == images


def _bridge_tokenizer():
    """Tiny tokenizer with the SmolVLM structure tokens + a newline token."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab={"<unk>": 0, "\n": 1, "hi": 2}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<fake_token_around_image>", "<global-img>", "<row_1_col_1>"]}
    )
    return tokenizer


def _bridge_ids(tokenizer):
    return {
        "img": tokenizer.convert_tokens_to_ids("<image>"),
        "fake": tokenizer.convert_tokens_to_ids("<fake_token_around_image>"),
        "global": tokenizer.convert_tokens_to_ids("<global-img>"),
        "row": tokenizer.convert_tokens_to_ids("<row_1_col_1>"),
        "nl": tokenizer.convert_tokens_to_ids("\n"),
        "text": tokenizer.convert_tokens_to_ids("hi"),
    }


def test_image_token_spans_bridge_tile_structure_and_whitespace():
    import torch

    tokenizer = _bridge_tokenizer()
    ids = _bridge_ids(tokenizer)
    image_block = [
        ids["fake"], ids["row"], ids["img"], ids["img"], ids["img"],
        ids["fake"],
        ids["nl"], ids["fake"], ids["global"], ids["img"], ids["img"], ids["img"],
    ]
    adapter = SmolVLMAdapter()
    one_image = torch.tensor([9, 9] + image_block + [ids["text"]])
    spans = adapter.image_token_spans(one_image, ids["img"], tokenizer=tokenizer)
    assert spans == [(4, 13)]
    # Without the tokenizer the adapter falls back to strict image runs (fail loud).
    strict = adapter.image_token_spans(one_image, ids["img"], tokenizer=None)
    assert strict == [(4, 6), (11, 13)]
    # A text token between two expansions keeps the images apart.
    two_images = torch.tensor([ids["text"]] + image_block + [ids["text"]] + image_block)
    spans = adapter.image_token_spans(two_images, ids["img"], tokenizer=tokenizer)
    assert len(spans) == 2


def test_capture_counts_only_image_tokens_inside_bridged_span():
    import torch
    from types import SimpleNamespace

    from vlm_dflash.target.sample_capture import forward_and_capture_batch

    tokenizer = _bridge_tokenizer()
    ids = _bridge_ids(tokenizer)
    row_ids = [ids["fake"], ids["row"], ids["img"], ids["img"], ids["img"], ids["nl"], ids["img"], ids["img"]]

    class _FakeSmolVLM:
        config = SimpleNamespace(model_type="smolvlm", pad_token_id=0)

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(50, 4))

        def parameters(self):
            return iter([torch.zeros(1)])

        def __call__(self, **kwargs):
            batch, length = kwargs["input_ids"].shape
            hidden = [torch.zeros(batch, length, 8, dtype=torch.bfloat16) for _ in range(3)]
            return SimpleNamespace(hidden_states=hidden)

    prepared = [{"inputs": {}, "full_ids": torch.tensor(row_ids), "allow_multi_span": False}]
    records = forward_and_capture_batch(
        _FakeSmolVLM(), prepared, [1], image_token_id=ids["img"], dtype=torch.float32,
        merge_size=None, tokenizer=tokenizer,
    )
    assert records[0]["image_span"] == (2, 7)
    assert records[0]["n_image_tokens"] == 5
