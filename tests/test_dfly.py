"""DFly unit tests: fusion residual, correction identity/chain, overfit, round trip."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.dfly.factory import resolve_dfly_config
from vlm_dflash.dfly.model import HiddenStatesCorrection, Qwen3DFlyDraftModel
from vlm_dflash.dfly.trainer import DFlyTrainerModule

H, V, BS = 64, 256, 8
CONTEXT = [1, 5, 9]


def _dfly_config(**overrides):
    dflash_config = {
        "mask_token_id": V - 1,
        "target_layer_ids": CONTEXT,
        "context_layer_ids": CONTEXT,
        "spatial_rope_enabled": False,
        "separate_context_kv": False,
        "sample_from_anchor": False,
        "fusion_trainable": True,
        "fusion_init": "distance",
        "enable_hidden_correction": True,
        "hidden_correction_intermediate_size": None,
    }
    dflash_config.update(overrides.pop("dflash_config", {}))
    target_hidden_size = overrides.pop("target_hidden_size", H)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=V,
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=BS,
        num_target_layers=12,
        target_hidden_size=target_hidden_size,
        dflash_config=dflash_config,
        **overrides,
    )
    config._attn_implementation = "sdpa"
    return config


def _tiny_module(seed: int = 0, **cfg_overrides):
    torch.manual_seed(seed)
    config = _dfly_config(**cfg_overrides)
    draft = Qwen3DFlyDraftModel(config)
    embed = torch.nn.Embedding(V, H)
    lm_head = torch.nn.Linear(H, V, bias=False)
    lm_head.weight = embed.weight
    # Mirror the factory: the target head/embed are frozen references.
    embed.weight.requires_grad_(False)
    module = DFlyTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=V - 1,
        block_size=BS,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
    )
    return module, embed


def _tiny_batch(batch_size: int = 2, seq_len: int = 48, seed: int = 0):
    torch.manual_seed(seed)
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    hidden = torch.randn(batch_size, seq_len, len(CONTEXT) * H)
    return input_ids, loss_mask, hidden


class _DummyTarget(torch.nn.Module):
    def __init__(self, embed: torch.nn.Embedding, head: torch.nn.Linear):
        super().__init__()
        self._embed = embed
        self.lm_head = head

    def get_input_embeddings(self):
        return self._embed


def test_registry_and_metadata():
    algorithm = get_algorithm("dfly")
    assert algorithm.name == "dfly"
    assert "dfly.py" in algorithm.metadata.references["upstream_reference"]
    assert "2607.25852" in algorithm.metadata.references["paper"]
    assert algorithm.metadata.targets == {"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"}


def test_structure_shared_kv_fc_and_correction():
    module, _ = _tiny_module()
    model = module.draft_model
    assert isinstance(model, Qwen3DFlyDraftModel)
    assert tuple(model.fc.weight.shape) == (H, len(CONTEXT) * H)
    assert tuple(model.layer_fusion_weights.shape) == (2, len(CONTEXT))
    assert model.hidden_correction is not None
    assert (model.hidden_correction.down_proj.weight == 0).all()
    for layer in model.layers:
        assert hasattr(layer.self_attn, "k_proj")
        assert not hasattr(layer.self_attn, "k_proj_target")


def test_correction_zero_init_is_identity():
    correction = HiddenStatesCorrection(hidden_size=H, intermediate_size=H)
    hidden = torch.randn(2, 3, H)
    embeds = torch.randn(2, 3, H)
    torch.testing.assert_close(correction(hidden, embeds), hidden)


def test_correction_matches_upstream_treeflash_formula():
    """Upstream TreeFlash formula (1), independently recomputed:
    ``h' = h + down_proj(silu(gate_proj(norm(h) :: norm(embed(prev)))) * up_proj(...))``
    with both streams RMS-normalized independently."""
    correction = HiddenStatesCorrection(hidden_size=H, intermediate_size=H)
    torch.manual_seed(3)
    with torch.no_grad():
        correction.down_proj.weight.normal_(0.0, 0.05)
    hidden = torch.randn(2, 3, H)
    embeds = torch.randn(2, 3, H)
    out = correction(hidden, embeds)

    def rms_norm(x, weight, eps):
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        return (x32 * torch.rsqrt(variance + eps)).to(x.dtype) * weight

    joined = torch.cat(
        [
            rms_norm(hidden, correction.hidden_norm.weight, correction.hidden_norm.variance_epsilon),
            rms_norm(embeds, correction.embed_norm.weight, correction.embed_norm.variance_epsilon),
        ],
        dim=-1,
    )
    delta = correction.down_proj(
        torch.nn.functional.silu(correction.gate_proj(joined)) * correction.up_proj(joined)
    )
    torch.testing.assert_close(out, hidden + delta)


def test_visual_diagnostic_routes_through_the_correction_hook():
    """The train-stage visual check must run the served (corrected) path:
    one pre-head hook call per pass, backbone == full at zero-init, and a
    trained correction actually moves the reported full logits."""
    from vlm_dflash.dflash.visual_check import draft_block_logits

    module, _ = _tiny_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    hook = module._correct_draft_hidden
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return hook(*args, **kwargs)

    module._correct_draft_hidden = counted
    backbone, full, anchors, keep_mask = draft_block_logits(module, input_ids, loss_mask, hidden, None)
    assert calls["count"] == 1
    assert anchors.shape[1] == keep_mask.shape[1]
    torch.testing.assert_close(backbone, full)  # zero-init: exact identity

    with torch.no_grad():
        module.draft_model.hidden_correction.down_proj.weight.normal_(0.0, 0.05)
    calls["count"] = 0
    backbone, full, _, _ = draft_block_logits(module, input_ids, loss_mask, hidden, None)
    assert calls["count"] == 1
    assert module._last_correction_delta_abs_mean is not None
    assert module._last_correction_delta_abs_mean > 0.0
    assert not torch.allclose(backbone.float(), full.float())


def test_config_constraints_fail_loud():
    with pytest.raises(ValueError, match="target_hidden_size"):
        torch.manual_seed(0)
        Qwen3DFlyDraftModel(_dfly_config(target_hidden_size=H + 8))
    with pytest.raises(ValueError, match="sample_from_anchor"):
        torch.manual_seed(0)
        Qwen3DFlyDraftModel(_dfly_config(dflash_config={"sample_from_anchor": True}))
    with pytest.raises(ValueError, match="separate_context_kv"):
        torch.manual_seed(0)
        Qwen3DFlyDraftModel(_dfly_config(dflash_config={"separate_context_kv": True}))


def test_training_step_and_metrics():
    module, _ = _tiny_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert torch.isfinite(metrics.loss)
    assert metrics.extra_metrics is not None
    assert metrics.extra_metrics["correction_enabled"] == 1.0
    assert "correction_delta_abs_mean" in metrics.extra_metrics
    assert "context_base_abs_mean" in metrics.extra_metrics
    assert "fusion_entropy_mean" in metrics.extra_metrics
    metrics.loss.backward()
    down_grad = module.draft_model.hidden_correction.down_proj.weight.grad
    assert down_grad is not None and down_grad.abs().sum() > 0
    gate_grad = module.draft_model.hidden_correction.gate_proj.weight.grad
    # zero-init residual: the first backward gives the gate/up path zero grad;
    # the module is kicked off through down_proj (checked above), and the next
    # step (below) must light up the gate/up projections.
    assert gate_grad is None or gate_grad.abs().sum() == 0

    optimizer = torch.optim.SGD(
        [p for p in module.parameters() if p.requires_grad], lr=0.2
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    metrics_second = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    metrics_second.loss.backward()
    gate_grad_second = module.draft_model.hidden_correction.gate_proj.weight.grad
    assert gate_grad_second is not None and gate_grad_second.abs().sum() > 0


def test_zero_init_correction_equals_correction_disabled():
    """With down_proj at zero, the module must be numerically equivalent to a
    correction-free DFly backbone (same weights, same anchors)."""
    module, _ = _tiny_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    torch.manual_seed(2024)
    enabled = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    model = module.draft_model
    saved = model.hidden_correction
    model.hidden_correction = None
    try:
        torch.manual_seed(2024)
        disabled = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    finally:
        model.hidden_correction = saved
    torch.testing.assert_close(enabled.loss, disabled.loss)


def test_correction_actually_changes_the_objective():
    module, _ = _tiny_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    torch.manual_seed(5)
    baseline = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    with torch.no_grad():
        module.draft_model.hidden_correction.down_proj.weight.normal_(0.0, 0.05)
    torch.manual_seed(5)
    corrected = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert not torch.allclose(baseline.loss, corrected.loss)


def test_inference_chain_uses_proposed_prefix_tokens():
    module, embed = _tiny_module()
    model = module.draft_model
    with torch.no_grad():
        model.hidden_correction.down_proj.weight.normal_(0.0, 0.05)
    target = _DummyTarget(embed, module.lm_head)
    torch.manual_seed(11)
    draft_hidden = torch.randn(1, BS, H)
    block_ids = torch.full((1, BS), V - 1, dtype=torch.long)
    block_ids[0, 0] = 42  # the anchor token (already in the accepted prefix)

    proposals = model._sample_draft_tokens(target, draft_hidden, block_ids)
    assert proposals.shape == (1, BS - 1)

    # Manual reconstruction of the documented semantics: slot k conditions on
    # the token proposed at slot k-1 (the anchor for the first proposal).
    current_hidden = draft_hidden[:, -BS:, :]
    completed = block_ids.clone()
    for position in range(1, BS):
        previous = completed[:, position - 1 : position]
        corrected = model.hidden_correction(
            current_hidden[:, position : position + 1, :], embed(previous)
        )
        completed[:, position] = module.lm_head(corrected).argmax(dim=-1).squeeze(1)
    torch.testing.assert_close(proposals, completed[:, 1:])


def test_inference_zero_init_matches_uncorrected_parallel_path():
    module, embed = _tiny_module()
    model = module.draft_model
    assert (model.hidden_correction.down_proj.weight == 0).all()
    target = _DummyTarget(embed, module.lm_head)
    torch.manual_seed(3)
    draft_hidden = torch.randn(1, BS, H)
    block_ids = torch.full((1, BS), V - 1, dtype=torch.long)
    block_ids[0, 0] = 17
    proposals = model._sample_draft_tokens(target, draft_hidden, block_ids)
    expected = module.lm_head(draft_hidden[:, -BS:, :][:, 1:, :]).argmax(dim=-1)
    torch.testing.assert_close(proposals, expected)


def test_config_can_disable_correction():
    module, _ = _tiny_module(dflash_config={"enable_hidden_correction": False})
    assert module.draft_model.hidden_correction is None
    assert not any("hidden_correction" in key for key in module.draft_model.state_dict())
    input_ids, loss_mask, hidden = _tiny_batch()
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert torch.isfinite(metrics.loss)
    assert metrics.extra_metrics["correction_enabled"] == 0.0
    assert "correction_delta_abs_mean" not in metrics.extra_metrics


def test_checkpoint_roundtrip_keeps_correction_params(tmp_path):
    module, _ = _tiny_module()
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": "dfly",
        "draft_config": module.draft_model.config.to_dict(),
        "attention_backend": "sdpa",
        "model": module.draft_model.state_dict(),
    }
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)
    from vlm_dflash.dfly.factory import load_checkpoint

    draft, meta = load_checkpoint(path)
    assert meta["algorithm"] == "dfly"
    keys = set(draft.state_dict())
    assert any("hidden_correction" in key for key in keys)
    assert "layer_fusion_weights" in keys
    for name, tensor in module.draft_model.state_dict().items():
        torch.testing.assert_close(draft.state_dict()[name], tensor)


def test_resolve_dfly_config_namespace():
    resolved = resolve_dfly_config({"algorithm": {"dfly": {"enable_hidden_correction": False}}})
    assert resolved["enable_hidden_correction"] is False
    assert resolved["context_layer_ids"] is None


def test_objective_profile_is_machine_readable_and_recipe_keys_fail_loud(monkeypatch):
    """The CE adaptation is recorded in the draft config and checkpoint; an
    upstream two-stage recipe config is rejected instead of silently training CE."""
    import vlm_dflash.dfly.factory as dfly_factory
    from vlm_dflash.dfly.factory import OBJECTIVE_PROFILE

    cfg = {
        "drafter": {"mask_token_id": V - 1, "block_size": BS, "num_hidden_layers": 2},
        "algorithm": {"dfly": {}},
    }
    contract = {"layers": {"draft_layer_ids": [0, 1]}, "text": {"hidden_size": H}}
    monkeypatch.setattr(
        dfly_factory, "build_dflash_draft_config", lambda contract, cfg: _dfly_config()
    )
    draft_config = dfly_factory.build_draft_config(contract, cfg)
    assert draft_config.dflash_config["objective_profile"] == OBJECTIVE_PROFILE
    meta = get_algorithm("dfly").checkpoint_metadata(contract, cfg, draft_config)
    assert meta["objective_profile"] == OBJECTIVE_PROFILE

    for knobs in ({"loss_objective": "e2e_tv"}, {"lk_loss_weight": 0.5}, {"two_stage": True}):
        with pytest.raises(ValueError, match="unsupported upstream-recipe knobs"):
            resolve_dfly_config({"algorithm": {"dfly": knobs}})
    with pytest.raises(ValueError, match="Supported keys"):
        resolve_dfly_config({"algorithm": {"dfly": {"fusion_trainble": True}}})

