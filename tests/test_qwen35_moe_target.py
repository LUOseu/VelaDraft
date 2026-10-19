"""Qwen3.5/3.6-MoE target support: partial rotary, interleaved MRoPE bounds, dense draft width."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from vlm_dflash.adapters import get_adapter, supported_model_types
from vlm_dflash.contract import ContractError
from vlm_dflash.rope_layout import mrope_section_error, partial_rotary_dim
from vlm_dflash.dflash.draft_model import (
    Qwen3DFlashDraftModel,
    _draft_rotary_embedding,
    _merge_multiaxis_rotary_embeddings,
    apply_rotary_pos_emb,
    rotate_half,
)
from vlm_dflash.dflash.factory import build_draft_config

QWEN36_ROPE = {
    "rope_type": "default",
    "rope_theta": 10000000.0,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
    "partial_rotary_factor": 0.25,
}
QWEN36_ROPE_FULL_HEAD = {key: value for key, value in QWEN36_ROPE.items() if key != "partial_rotary_factor"}


@pytest.mark.parametrize(
    ("sections", "rotary_dim", "interleaved", "expected"),
    [
        ((11, 11, 10), 64, True, None),  # Qwen3.5/3.6: fits although 3 * 11 > 32
        ((24, 20, 20), 128, True, None),  # Qwen3-VL
        ((16, 24, 24), 128, False, None),  # Qwen2.5-VL chunked
        ((11, 11, 10), 256, True, "rotary dim 256"),
        ((16, 30, 18), 128, True, "axis 1"),
        ((10, 10, 12), 64, True, "axis 2"),
        ((10, 10, 12), 64, False, None),  # chunked has no interleave bound
    ],
)
def test_mrope_section_error(sections, rotary_dim, interleaved, expected):
    error = mrope_section_error(sections, rotary_dim, interleaved)
    assert error is None if expected is None else expected in error


@pytest.mark.parametrize(("head_dim", "factor", "expected"), [(256, 0.25, 64), (128, 1.0, 128), (16, 0.5, 8)])
def test_partial_rotary_dim(head_dim, factor, expected):
    assert partial_rotary_dim(head_dim, factor) == expected


@pytest.mark.parametrize("factor", [0.3, 0.0, 1.5, 0.1])
def test_partial_rotary_dim_rejects_bad_factor(factor):
    with pytest.raises(ValueError, match="even rotary dim"):
        partial_rotary_dim(10, factor)


def test_adapter_is_registered():
    adapter = get_adapter("qwen3_5_moe")
    assert "qwen3_5_moe" in supported_model_types()
    assert adapter.info.model_class == "Qwen3_5MoeForConditionalGeneration"


def test_adapter_forces_the_tracked_torch_reference_for_npu(monkeypatch):
    from vlm_dflash.adapters.base import VLMAdapter

    modeling = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    )
    sentinels = {name: object() for name in names}
    for name, sentinel in sentinels.items():
        monkeypatch.setattr(modeling, name, sentinel)
    monkeypatch.setattr(modeling, "is_fast_path_available", True)
    observed = {}

    def fake_load_model(self, *args, **kwargs):
        observed["kwargs"] = kwargs
        observed["functions"] = {name: getattr(modeling, name) for name in names}
        observed["fast_path"] = modeling.is_fast_path_available
        return SimpleNamespace()

    monkeypatch.setattr(VLMAdapter, "load_model", fake_load_model)
    model = get_adapter("qwen3_5_moe").load_model(
        "/models/qwen3.5",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        backend_hint="npu",
    )
    assert model._vlm_dflash_gated_delta_backend == "torch_reference"
    assert observed["fast_path"] is False
    assert all(value is None for value in observed["functions"].values())
    assert observed["kwargs"]["backend_hint"] == "npu"
    for name, sentinel in sentinels.items():
        assert getattr(modeling, name) is sentinel
    assert modeling.is_fast_path_available is True


def _text_config(rope: dict, partial: float | None = None) -> SimpleNamespace:
    config = SimpleNamespace(rope_parameters=dict(rope))
    if partial is not None:
        config.partial_rotary_factor = partial
    return config


def test_resolve_rope_partial_rotary():
    adapter = get_adapter("qwen3_5_moe")
    text = _text_config(QWEN36_ROPE)
    rope = adapter.resolve_rope(text, text, 256, {})
    assert rope["sections"] == [11, 11, 10]
    assert rope["interleaved"] is True
    assert rope["partial_rotary_factor"] == 0.25


def test_resolve_rope_reads_text_config_partial_factor():
    adapter = get_adapter("qwen3_5_moe")
    text = _text_config(QWEN36_ROPE_FULL_HEAD, partial=0.25)
    assert adapter.resolve_rope(text, text, 256, {})["partial_rotary_factor"] == 0.25


def test_resolve_rope_rejects_sections_against_full_head():
    adapter = get_adapter("qwen3_5_moe")
    text = _text_config(QWEN36_ROPE_FULL_HEAD)
    with pytest.raises(ContractError, match="rotary dim 256"):
        adapter.resolve_rope(text, text, 256, {})


def test_resolve_rope_rejects_interleaved_overflow():
    adapter = get_adapter("qwen3_5_moe")
    text = _text_config({**QWEN36_ROPE, "mrope_section": [10, 10, 12]})
    with pytest.raises(ContractError, match="axis 2"):
        adapter.resolve_rope(text, text, 256, {})


def _draft_qwen3_config(head_dim: int, partial: float | None, **extra) -> Qwen3Config:
    rope = {"rope_type": "default", "rope_theta": 10000000.0}
    if partial is not None:
        rope["partial_rotary_factor"] = partial
    return Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=512,
        rope_parameters=rope,
        **extra,
    )


def test_draft_rotary_partial_width():
    rotary, rotary_dim = _draft_rotary_embedding(_draft_qwen3_config(256, 0.25))
    assert rotary_dim == 64
    assert tuple(rotary.inv_freq.shape) == (32,)


def test_draft_rotary_full_width_is_unchanged():
    config = _draft_qwen3_config(16, None)
    rotary, rotary_dim = _draft_rotary_embedding(config)
    assert rotary_dim == 16
    assert rotary.config is config


def test_resolve_rope_rejects_bad_partial_factor():
    adapter = get_adapter("qwen3_5_moe")
    text = _text_config({**QWEN36_ROPE, "partial_rotary_factor": 0.3})
    with pytest.raises(ContractError, match="even rotary dim"):
        adapter.resolve_rope(text, text, 10, {})


def test_partial_rotary_matches_hf_qwen3_5_moe():
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    hf_config = Qwen3_5MoeTextConfig(
        hidden_size=512, num_attention_heads=2, num_key_value_heads=1, head_dim=256,
        rope_parameters=dict(QWEN36_ROPE),
    )
    hf_rotary = modeling.Qwen3_5MoeTextRotaryEmbedding(hf_config)
    positions = torch.stack(
        [torch.arange(12).repeat(2, 1), (torch.arange(12) // 4).repeat(2, 1), (torch.arange(12) % 4).repeat(2, 1)]
    )
    hidden = torch.zeros(2, 12, 8)
    hf_cos, hf_sin = hf_rotary(hidden, positions)

    rotary, _ = _draft_rotary_embedding(_draft_qwen3_config(256, 0.25))
    cos, sin = _merge_multiaxis_rotary_embeddings(rotary, hidden, positions, (11, 11, 10), interleaved=True)
    assert torch.allclose(cos, hf_cos, atol=1e-6)
    assert torch.allclose(sin, hf_sin, atol=1e-6)

    torch.manual_seed(0)
    q, k = torch.randn(2, 2, 12, 256), torch.randn(2, 1, 12, 256)
    hf_q, hf_k = modeling.apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    our_q, our_k = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(our_q, hf_q, atol=1e-5)
    assert torch.allclose(our_k, hf_k, atol=1e-5)
    # The non-rotary channels pass through untouched.
    assert torch.equal(our_q[..., 64:], q[..., 64:])
    assert torch.equal(our_k[..., 64:], k[..., 64:])


def test_full_rotary_apply_is_the_standard_formula():
    torch.manual_seed(0)
    q, k = torch.randn(1, 2, 3, 16), torch.randn(1, 1, 5, 16)
    cos, sin = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    our_q, our_k = apply_rotary_pos_emb(q, k, cos, sin)
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
    assert torch.allclose(our_q, q * cos_u[..., -3:, :] + rotate_half(q) * sin_u[..., -3:, :])
    assert torch.allclose(our_k, k * cos_u + rotate_half(k) * sin_u)


def _spatial_draft_config(sections: list[int], partial: float = 0.5) -> Qwen3Config:
    config = _draft_qwen3_config(
        16,
        partial,
        block_size=4,
        num_target_layers=2,
        dflash_config={
            "mask_token_id": 255,
            "target_layer_ids": [0, 1],
            "spatial_rope_enabled": True,
            "spatial_rope_mode": "replace",
            "spatial_rope_sections": sections,
            "spatial_rope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def test_partial_rotary_draft_forward_keeps_shapes():
    torch.manual_seed(0)
    draft = Qwen3DFlashDraftModel(_spatial_draft_config([2, 1, 1]))
    assert draft.rotary_dim == 8
    context, block = 5, 4
    positions = torch.arange(context + block).unsqueeze(0)
    multimodal = positions.unsqueeze(0).expand(3, 1, -1).clone()
    out = draft(
        position_ids=positions,
        noise_embedding=torch.randn(1, block, 64),
        target_hidden=torch.randn(1, context, 2 * 64),
        multimodal_position_ids=multimodal,
    )
    assert out.shape == (1, block, 64)
    assert torch.isfinite(out).all()


def test_partial_rotary_draft_rejects_sections_against_rotary_dim():
    with pytest.raises(ValueError, match="rotary dim 8"):
        Qwen3DFlashDraftModel(_spatial_draft_config([4, 2, 2]))


def test_partial_rotary_draft_rejects_interleaved_overflow():
    with pytest.raises(ValueError, match="axis 1"):
        Qwen3DFlashDraftModel(_spatial_draft_config([1, 2, 1]))


def _moe_contract(tmp_path, *, intermediate_in_config: bool) -> dict:
    configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    config = configuration.Qwen3_5MoeConfig(text_config={"rope_parameters": dict(QWEN36_ROPE)})
    if intermediate_in_config:
        config.text_config.intermediate_size = 4096
    config.save_pretrained(tmp_path)
    return {
        "target": {"model_path": str(tmp_path)},
        "layers": {"num_target_layers": 40, "draft_layer_ids": [1, 19, 37]},
        "text": {"hidden_size": 2048},
        "rope": {"sections": [11, 11, 10], "interleaved": True, "partial_rotary_factor": 0.25},
    }


def _drafter_cfg(**extra) -> dict:
    return {"drafter": {"num_hidden_layers": 3, "block_size": 8, "mask_token_id": 248077, **extra}}


def test_build_draft_config_moe_target_uses_drafter_width(tmp_path):
    config = build_draft_config(_moe_contract(tmp_path, intermediate_in_config=False), _drafter_cfg(intermediate_size=6144))
    assert config.intermediate_size == 6144
    assert config.layer_types == ["full_attention"] * 3
    assert config.rope_parameters == {
        "rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
    }
    assert config.dflash_config["spatial_rope_sections"] == [11, 11, 10]


def test_build_draft_config_moe_target_requires_drafter_width(tmp_path):
    with pytest.raises(ValueError, match="drafter.intermediate_size"):
        build_draft_config(_moe_contract(tmp_path, intermediate_in_config=False), _drafter_cfg())


def test_build_draft_config_explicit_width_overrides_dense_target(tmp_path):
    contract = _moe_contract(tmp_path, intermediate_in_config=True)
    assert build_draft_config(contract, _drafter_cfg()).intermediate_size == 4096
    assert build_draft_config(contract, _drafter_cfg(intermediate_size=6144)).intermediate_size == 6144


def test_partial_rotary_inv_freq_survives_dtype_cast():
    draft = Qwen3DFlashDraftModel(_spatial_draft_config([2, 1, 1])).to(torch.bfloat16)
    assert draft.rotary_emb.inv_freq.dtype == torch.float32
    assert tuple(draft.rotary_emb.inv_freq.shape) == (4,)


def _validation_contract(intermediate_size="absent") -> dict:
    contract = {
        "mask": {"token_id": 248077},
        "layers": {"block_size": 8, "num_draft_layers": 3, "num_target_layers": 40},
        "rope": {"sections": [11, 11, 10], "interleaved": True, "mode": "mrope"},
        "image": {"min_pixels": 65536, "max_pixels": 1048576},
        "text": {"hidden_size": 2048},
    }
    if intermediate_size != "absent":
        contract["text"]["intermediate_size"] = intermediate_size
    return contract


def test_contract_rejects_moe_target_without_drafter_width_before_any_stage():
    from vlm_dflash.contract import validate_yaml_against_contract

    with pytest.raises(ContractError, match="drafter.intermediate_size"):
        validate_yaml_against_contract(_drafter_cfg(), _validation_contract(None))
    validate_yaml_against_contract(_drafter_cfg(intermediate_size=6144), _validation_contract(None))
    validate_yaml_against_contract(_drafter_cfg(), _validation_contract(4096))
    validate_yaml_against_contract(_drafter_cfg(), _validation_contract())  # contracts predating the field


def test_spec_decode_eval_refuses_uncroppable_cache():
    from vlm_dflash.adapters import require_cache_crop_rollback

    with pytest.raises(RuntimeError, match="DynamicCache.crop"):
        require_cache_crop_rollback(get_adapter("qwen3_5_moe"), "eval")
    require_cache_crop_rollback(get_adapter("qwen3_vl"), "eval")


def test_eval_and_benchmark_refuse_qwen35_before_any_work(tmp_path):
    from vlm_dflash.benchmark.harness import run_benchmark
    from vlm_dflash.eval.run import evaluate

    contract = {"target": {"model_type": "qwen3_5_moe"}}
    with pytest.raises(RuntimeError, match="eval: target family 'qwen3_5_moe'"):
        evaluate({}, tmp_path, contract)
    with pytest.raises(RuntimeError, match="benchmark: target family 'qwen3_5_moe'"):
        run_benchmark({}, tmp_path, contract, [], out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_spec_generate_refuses_uncroppable_target():
    class _Qwen35LikeTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5_moe")
            self.model = SimpleNamespace(compute_3d_position_ids=lambda **kwargs: None)
            self.device = torch.device("cpu")

    draft = Qwen3DFlashDraftModel(_spatial_draft_config([2, 1, 1]))
    with pytest.raises(RuntimeError, match="spec_generate: target family 'qwen3_5_moe'"):
        draft.spec_generate(
            _Qwen35LikeTarget(), torch.zeros(1, 3, dtype=torch.long), max_new_tokens=4, stop_token_ids=None,
            temperature=0.0,
        )


def test_resolve_rope_treats_null_partial_factor_as_absent():
    adapter = get_adapter("qwen3_5_moe")
    text = _text_config({**QWEN36_ROPE, "partial_rotary_factor": None}, partial=0.25)
    assert adapter.resolve_rope(text, text, 256, {})["partial_rotary_factor"] == 0.25


def test_build_draft_config_treats_null_width_as_missing(tmp_path):
    contract = _moe_contract(tmp_path, intermediate_in_config=False)
    configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    config = configuration.Qwen3_5MoeConfig.from_pretrained(tmp_path)
    config.text_config.intermediate_size = None
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="drafter.intermediate_size"):
        build_draft_config(contract, _drafter_cfg())


def test_contract_warns_when_spec_decode_eval_is_unavailable(tmp_path, capsys, monkeypatch):
    from vlm_dflash import contract as contract_module

    configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    configuration.Qwen3_5MoeConfig(text_config={"rope_parameters": dict(QWEN36_ROPE)}).save_pretrained(tmp_path)

    class _Stop(Exception):
        pass

    def stop_after_rope(*args, **kwargs):
        raise _Stop

    # Stop right after the rope/warning step: tokenizer and processor files are not needed.
    monkeypatch.setattr(contract_module, "build_target_layer_ids", stop_after_rope)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: SimpleNamespace(
        convert_tokens_to_ids=lambda token: 1, convert_ids_to_tokens=lambda index: None, vocab_size=248044,
    ))
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(get_adapter("qwen3_5_moe"), "resolve_tokens", lambda *a: {"image_token_id": 248056})
    with pytest.raises(_Stop):
        contract_module.resolve_contract(tmp_path, {"mask_token_id": 248077})
    assert "no cache rollback" in capsys.readouterr().out
