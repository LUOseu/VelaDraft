"""DFlare unit tests: fusion init, separate context K/V, wiring, overfit, round trip."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.dflare.factory import resolve_dflare_config
from vlm_dflash.dflare.model import Qwen3DFlareDraftModel, init_fusion_weights_
from vlm_dflash.dflare.trainer import DFlareTrainerModule

H, V, BS = 64, 256, 8
CONTEXT = [1, 5, 9]


def _flare_config(**overrides):
    dflash_config = {
        "mask_token_id": V - 1,
        "target_layer_ids": CONTEXT,
        "context_layer_ids": CONTEXT,
        "spatial_rope_enabled": False,
        "separate_context_kv": True,
        "fusion_trainable": True,
        "fusion_init": "distance",
    }
    dflash_config.update(overrides.pop("dflash_config", {}))
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
        target_hidden_size=H,
        dflash_config=dflash_config,
        **overrides,
    )
    config._attn_implementation = "sdpa"
    return config


def _tiny_module(seed: int = 0, **cfg_overrides):
    torch.manual_seed(seed)
    config = _flare_config(**cfg_overrides)
    draft = Qwen3DFlareDraftModel(config)
    embed = torch.nn.Embedding(V, H)
    lm_head = torch.nn.Linear(H, V, bias=False)
    lm_head.weight = embed.weight
    # Mirror the factory: the target head/embed are frozen references.
    embed.weight.requires_grad_(False)
    module = DFlareTrainerModule(
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


def _tiny_batch(batch_size: int = 2, seq_len: int = 48, seed: int = 0, layers: int = len(CONTEXT)):
    torch.manual_seed(seed)
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    hidden = torch.randn(batch_size, seq_len, layers * H)
    return input_ids, loss_mask, hidden


def test_registry_and_metadata():
    algorithm = get_algorithm("dflare")
    assert algorithm.name == "dflare"
    assert "dflare.py" in algorithm.metadata.references["upstream_reference"]
    assert "2606.02091" in algorithm.metadata.references["paper"]
    assert algorithm.metadata.targets == {"qwen25_vl": "validated", "qwen3_vl": "reduced_validation"}


def test_fusion_distance_prior_init():
    weights = torch.empty(2, 3)
    init_fusion_weights_(weights, mode="distance")
    # draft layer 0 prefers target layer 0; draft layer 1 prefers the deepest one
    torch.testing.assert_close(weights[0], torch.tensor([0.0, -2.0, -4.0]))
    torch.testing.assert_close(weights[1], torch.tensor([-4.0, -2.0, 0.0]))
    uniform = torch.ones(2, 3)
    init_fusion_weights_(uniform, mode="uniform")
    assert torch.equal(uniform, torch.zeros(2, 3))
    with pytest.raises(ValueError):
        init_fusion_weights_(torch.empty(2, 3), mode="random")


def test_structure_separate_context_kv_and_no_fc():
    module, _ = _tiny_module()
    model = module.draft_model
    assert isinstance(model, Qwen3DFlareDraftModel)
    assert model.num_context_layers == len(CONTEXT)
    assert tuple(model.layer_fusion_weights.shape) == (2, len(CONTEXT))
    assert not hasattr(model, "fc")
    for layer in model.layers:
        assert hasattr(layer.self_attn, "k_proj_target")
        assert hasattr(layer.self_attn, "v_proj_target")


def test_shared_kv_ablation_has_no_target_projections():
    module, _ = _tiny_module(dflash_config={"separate_context_kv": False})
    for layer in module.draft_model.layers:
        assert not hasattr(layer.self_attn, "k_proj_target")


def test_fusion_is_wired_and_receives_gradients():
    module, _ = _tiny_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    grads = {name: parameter.grad for name, parameter in module.draft_model.named_parameters()}
    fusion_grad = grads["layer_fusion_weights"]
    assert fusion_grad is not None and fusion_grad.abs().sum() > 0
    assert any(
        name.endswith("k_proj_target.weight") and grad is not None and grad.abs().sum() > 0
        for name, grad in grads.items()
    )
    # Frozen target head/embed never move.
    assert module.lm_head.weight.requires_grad is False

    # Perturbing the fusion logits must change the loss (fusion actually runs).
    with torch.no_grad():
        module.draft_model.layer_fusion_weights.add_(1.5)
    torch.manual_seed(0)
    perturbed = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert not torch.allclose(perturbed.loss, metrics.loss)


def test_fixed_fusion_receives_no_gradient():
    module, _ = _tiny_module(dflash_config={"fusion_trainable": False})
    assert module.draft_model.layer_fusion_weights.requires_grad is False
    input_ids, loss_mask, hidden = _tiny_batch()
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    metrics.loss.backward()
    assert module.draft_model.layer_fusion_weights.grad is None
    metrics_extra = module.fusion_metrics()
    assert metrics_extra["fusion_trainable"] == 0.0


def test_fusion_metrics_prefer_fp32_master_and_report_prior_drift():
    module, _ = _tiny_module()
    model = module.draft_model
    base = dict(module.fusion_metrics())
    # Untouched fusion sits exactly on its prior.
    assert base["fusion_prob_l1_drift"] == pytest.approx(0.0, abs=1e-12)
    # Attach a fake fp32 master that has moved: the bf16 live parameter cannot
    # show sub-ulp moves, so the diagnostics must follow the master.
    # Row-wise rescaling changes the softmax mixture (a constant shift would not).
    master = torch.nn.Parameter(model.layer_fusion_weights.detach().float().clone() * 0.6)
    model.set_master_parameters({"layer_fusion_weights": master})
    moved = dict(module.fusion_metrics())
    assert moved["fusion_weight_abs_mean"] == pytest.approx(float(master.detach().abs().mean()))
    assert moved["fusion_prob_l1_drift"] > 0.0
    assert moved["fusion_entropy_mean"] != pytest.approx(base["fusion_entropy_mean"])
    # Without a master the metrics fall back to the live parameter.
    model._master_fusion_weights = None
    live = dict(module.fusion_metrics())
    assert live["fusion_weight_abs_mean"] == pytest.approx(
        float(model.layer_fusion_weights.detach().float().abs().mean())
    )


def test_training_loop_attaches_fp32_fusion_master(tmp_path):
    from vlm_dflash.dflash.train import run_training_loop

    torch.manual_seed(0)
    module, _ = _tiny_module()
    seq = 48
    items = []
    for _ in range(6):
        ids = torch.arange(seq).remainder(200).add(10)
        mask = torch.ones(seq)
        mask[:8] = 0.0
        hidden = torch.randn(seq, len(CONTEXT) * H) * 0.2
        items.append(
            {
                "input_ids": ids,
                "loss_mask": mask,
                "hidden": hidden,
                "position3": torch.arange(seq).repeat(3, 1),
                "image_span": (0, 0),
                "sample_id": "tiny",
            }
        )
    class _Wrap(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, index):
            return items[index]

    output = run_training_loop(
        module,
        _Wrap(),
        {"train": {"micro_batch_size": 2, "num_epochs": 1, "lr": 1e-3, "log_every_steps": 0, "draft_dtype": "float32"}},
        {"tokens": {"image_token_id": 999999}},
        torch.device("cpu"),
        total_steps=2,
        seed=0,
        shuffle=False,
        log_path=None,
        save_every=1,
    )
    assert module.draft_model._master_fusion_weights is not None
    assert output["history"], "no history rows"
    row = output["history"][-1]
    assert "fusion_prob_l1_drift" in row
    assert row["fusion_prob_l1_drift"] >= 0.0


def test_superset_store_slices_to_context_layers():
    """A store captured with a superset reproduces the subset training exactly."""
    module, _ = _tiny_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    torch.manual_seed(123)
    subset_metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)

    # Build a 5-layer superset store whose [1, 5, 9] slices are exactly `hidden`
    # (the store concatenates per-layer [B, S, H] features, so split `hidden`
    # into its per-layer chunks first).
    layers = [0, 1, 5, 9, 12]
    chunks = list(hidden.split(H, dim=-1))
    torch.manual_seed(7)
    superset = torch.cat(
        [
            chunks[CONTEXT.index(layer)] if layer in CONTEXT else torch.randn_like(chunks[0])
            for layer in layers
        ],
        dim=-1,
    )
    module.configure_feature_layout([0, 1, 5, 9, 12], hidden_size=H)
    torch.manual_seed(123)
    superset_metrics = module(input_ids=input_ids, hidden_states=superset, loss_mask=loss_mask)
    torch.testing.assert_close(subset_metrics.loss, superset_metrics.loss)


def test_missing_consumed_layer_fails_loud():
    module, _ = _tiny_module()
    with pytest.raises(ValueError, match="Recapture"):
        module.configure_feature_layout([1, 5, 13], hidden_size=H)


def test_checkpoint_roundtrip_keeps_dflare_params(tmp_path):
    module, _ = _tiny_module()
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": "dflare",
        "draft_config": module.draft_model.config.to_dict(),
        "attention_backend": "sdpa",
        "model": module.draft_model.state_dict(),
    }
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)
    from vlm_dflash.dflare.factory import load_checkpoint

    draft, meta = load_checkpoint(path)
    assert meta["algorithm"] == "dflare"
    reloaded_keys = set(draft.state_dict())
    original_keys = set(module.draft_model.state_dict())
    assert original_keys <= reloaded_keys
    assert "layer_fusion_weights" in reloaded_keys
    assert any("k_proj_target" in key for key in reloaded_keys)
    for name, tensor in module.draft_model.state_dict().items():
        torch.testing.assert_close(draft.state_dict()[name], tensor)


def test_resolve_dflare_config_namespace():
    cfg = {"algorithm": {"name": "dflare", "dflare": {"fusion_trainable": False}}}
    resolved = resolve_dflare_config(cfg)
    assert resolved["fusion_trainable"] is False
    assert resolved["separate_context_kv"] is True
    default = resolve_dflare_config({"algorithm": "dflare"})
    assert default["context_layer_ids"] is None

