"""Domino unit tests: curriculum, correction head, dual loss, registry plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.domino.factory import resolve_domino_config
from vlm_dflash.domino.trainer import DominoTrainerModule, get_lambda_base
from vlm_dflash.utils import gradient_coverage_report


def _tiny_domino_module(
    seed: int = 0,
    *,
    shift_label: bool = True,
    pure_draft_prefix_len: int = 1,
    enable_correction_head: bool = True,
    block_size: int = 8,
) -> DominoTrainerModule:
    torch.manual_seed(seed)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=block_size,
        num_target_layers=2,
        dflash_config={
            "mask_token_id": 255,
            "target_layer_ids": [0, 1],
            "spatial_rope_enabled": False,
            "projector_type": "domino",
            "emb_dim": 16,
            "gru_hidden_dim": 32,
            "pure_draft_prefix_len": pure_draft_prefix_len,
            "shift_label": shift_label,
        },
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    # A near-zero correction would make the dual-loss assertions vacuous:
    # SiLU(0) = 0, so the default last-layer init keeps the delta tiny. Give the
    # projection head a decisive scale.
    with torch.no_grad():
        for parameter in draft.embed_proj.parameters():
            parameter.normal_(std=0.05)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    return DominoTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
        sample_from_anchor=shift_label,
        enable_correction_head=enable_correction_head,
    )


def _tiny_batch(batch_size: int = 2, seq_len: int = 48, seed: int = 0):
    torch.manual_seed(seed)
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    embed = torch.nn.Embedding(256, 64)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(256, 64) * 0.05)
        base = embed(input_ids)
        hidden = torch.cat([base + 0.02 * torch.randn_like(base), torch.tanh(base)], dim=-1)
    return input_ids, loss_mask, hidden


# --------------------------------------------------------------------------- #
# curriculum
# --------------------------------------------------------------------------- #

def test_get_lambda_base_linear_decay_and_clamp():
    assert get_lambda_base(0, 100, 1.0, 1.0) == pytest.approx(1.0)
    assert get_lambda_base(50, 100, 1.0, 1.0) == pytest.approx(0.5)
    assert get_lambda_base(100, 100, 1.0, 1.0) == pytest.approx(0.0)
    assert get_lambda_base(200, 100, 1.0, 1.0) == pytest.approx(0.0)
    # decay_ratio shortens the window: done at 50% of steps here.
    assert get_lambda_base(25, 100, 1.0, 0.5) == pytest.approx(0.5)
    assert get_lambda_base(50, 100, 1.0, 0.5) == pytest.approx(0.0)
    # A start below 1.0 scales the whole ramp.
    assert get_lambda_base(0, 100, 0.4, 1.0) == pytest.approx(0.4)
    assert get_lambda_base(50, 100, 0.4, 1.0) == pytest.approx(0.2)


def test_set_training_progress_drives_the_module_weight():
    module = _tiny_domino_module()
    module.set_training_progress(0, 792)
    assert module.current_lambda_base == pytest.approx(1.0)
    module.set_training_progress(396, 792)
    assert module.current_lambda_base == pytest.approx(0.5)
    module.set_training_progress(792, 792)
    assert module.current_lambda_base == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# dual-logit loss
# --------------------------------------------------------------------------- #

def test_domino_curriculum_endpoints_match_single_head_losses():
    """λ=1 must equal the disabled (pure-base) objective; λ=0 the corrected one."""
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_domino_module()
    disabled = _tiny_domino_module(enable_correction_head=False)
    disabled.load_state_dict(module.state_dict())

    disabled_metrics = disabled(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert float(disabled_metrics.extra_metrics["correction_enabled"]) == 0.0
    assert "final_loss" in disabled_metrics.extra_metrics

    module.set_training_progress(0, 100)  # lambda_base = 1.0
    metrics_lambda1 = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert float(metrics_lambda1.extra_metrics["lambda_base"]) == pytest.approx(1.0)
    # Anchors are sampled randomly per forward, so only same-forward
    # comparisons are meaningful: at lambda=1 the mixed loss must equal the
    # base head's loss, and the disabled module must reduce to base CE too.
    assert float(metrics_lambda1.loss.detach()) == pytest.approx(
        float(metrics_lambda1.extra_metrics["base_loss"]), rel=1e-6
    )
    assert float(disabled_metrics.loss.detach()) == pytest.approx(
        float(disabled_metrics.extra_metrics["base_loss"]), rel=1e-6
    )

    module.set_training_progress(100, 100)  # lambda_base = 0.0
    metrics_lambda0 = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert float(metrics_lambda0.extra_metrics["lambda_base"]) == pytest.approx(0.0)
    assert float(metrics_lambda0.loss.detach()) == pytest.approx(
        float(metrics_lambda0.extra_metrics["final_loss"]), rel=1e-6
    )
    # The correction head is decidable at init (scaled projection): the two
    # heads' losses must differ, otherwise the test is vacuous.
    assert abs(
        float(metrics_lambda0.extra_metrics["final_loss"]) - float(metrics_lambda0.extra_metrics["base_loss"])
    ) > 1e-4
    assert "base_accuracy" in metrics_lambda0.extra_metrics
    assert "base_accept_len" in metrics_lambda0.extra_metrics
    assert "correction_abs_mean" in metrics_lambda0.extra_metrics
    assert float(metrics_lambda0.extra_metrics["correction_abs_mean"]) > 0


def test_domino_correction_touches_only_the_suffix():
    """With pure_draft_prefix_len=1 the leading block position stays base-only."""
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_domino_module()
    logits = torch.zeros(2, 16 * 8, 256)
    output_hidden = torch.ones(2, 16 * 8, 64)
    label_indices = torch.zeros(2, 16, 8, dtype=torch.long)
    target_ids = torch.zeros(2, 16, 8, dtype=torch.long)
    corrected, delta = module._domino_correction(
        block_logits=logits.view(2, 16, 8, -1),
        hidden4d=output_hidden.view(2, 16, 8, -1),
        input_ids=input_ids,
        label_indices=label_indices,
        target_ids=target_ids,
        bsz=2,
        n=16,
        bs=8,
    )
    assert module.correction_suffix_start == 1
    assert torch.allclose(corrected[:, :, 0, :], logits.view(2, 16, 8, -1)[:, :, 0, :])
    assert not torch.allclose(corrected[:, :, 1:, :], logits.view(2, 16, 8, -1)[:, :, 1:, :])
    assert delta.shape == (2, 16, 7, 256)


# --------------------------------------------------------------------------- #
# training / gradients
# --------------------------------------------------------------------------- #

def _run_steps(module, input_ids, loss_mask, hidden, steps: int = 60, lr: float = 2e-3):
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        module.set_training_progress(step, steps)
        metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
        optimizer.zero_grad()
        metrics.loss.backward()
        optimizer.step()
        assert torch.isfinite(metrics.loss)
        losses.append(float(metrics.loss.detach()))
    return metrics, losses


def test_domino_overfits_and_correction_head_receives_gradient():
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_domino_module()
    metrics, losses = _run_steps(module, input_ids, loss_mask, hidden, steps=80)
    assert losses[-1] < 0.8 * losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    gru_grad = module.draft_model.prefix_gru.weight_ih_l0.grad
    proj_grad = module.draft_model.embed_proj[0].weight.grad
    assert gru_grad is not None and float(gru_grad.abs().sum()) > 0
    assert proj_grad is not None and float(proj_grad.abs().sum()) > 0
    assert metrics.extra_metrics is not None
    for key in ("lambda_base", "final_loss", "base_loss", "base_accuracy", "base_accept_len"):
        assert key in metrics.extra_metrics
    # Slots: shifted labels supervise every block position (sample_from_anchor).
    assert module.first_supervised_slot == 0
    assert metrics.pos_count.shape[0] == module.block_size


def test_domino_disabled_correction_is_exact_dflash_path():
    """The ablation switch must not build/consume GRU states at all."""
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_domino_module(enable_correction_head=False)
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    metrics.loss.backward()
    report = gradient_coverage_report(module)
    groups = {group["group"]: group for group in report["groups"]}
    assert groups["draft_model.prefix_gru"]["params_with_nonzero_grad"] == 0
    assert groups["draft_model.embed_proj"]["params_with_nonzero_grad"] == 0
    assert groups["draft_model.prefix_gru"]["trainable_params"] > 0
    assert "correction_abs_mean" not in (metrics.extra_metrics or {})


def test_domino_gradient_coverage_reports_the_correction_head():
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_domino_module()
    # Mid-curriculum (lambda < 1): the head participates in the loss. At
    # lambda = 1 (the very first step of a real run) it intentionally does not.
    module.set_training_progress(50, 100)
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    metrics.loss.backward()
    report = gradient_coverage_report(module)
    groups = {group["group"]: group for group in report["groups"]}
    assert groups["draft_model.prefix_gru"]["params_with_nonzero_grad"] > 0
    assert groups["draft_model.embed_proj"]["params_with_nonzero_grad"] > 0
    assert "draft_model.layers" in groups


# --------------------------------------------------------------------------- #
# config plumbing / gates
# --------------------------------------------------------------------------- #

def test_domino_algorithm_registered_with_namespaced_config():
    algorithm = get_algorithm("domino")
    assert algorithm.name == "domino"
    assert algorithm.metadata.maturity["training"] in {"stable", "validated", "experimental", "architecture-only"}
    assert get_algorithm({"name": "domino"}).name == "domino"
    cfg = {"drafter": {}, "algorithm": {"name": "domino", "domino": {"emb_dim": 64, "lambda_base_start": 0.5}}}
    algorithm.validate_config(cfg)
    resolved = resolve_domino_config(cfg)
    assert resolved["emb_dim"] == 64
    assert resolved["lambda_base_start"] == 0.5
    assert resolved["shift_label"] is True  # maintainer VL recipe default
    assert resolved["pure_draft_prefix_len"] == 1
    with pytest.raises(ValueError, match="emb_dim"):
        algorithm.validate_config({"drafter": {}, "algorithm": {"name": "domino", "domino": {"emb_dim": 0}}})
    with pytest.raises(ValueError, match="lambda_base_decay_ratio"):
        algorithm.validate_config(
            {"drafter": {}, "algorithm": {"name": "domino", "domino": {"lambda_base_decay_ratio": 0.0}}}
        )
    with pytest.raises(ValueError, match="pure_draft_prefix_len"):
        algorithm.validate_config(
            {"drafter": {"block_size": 8}, "algorithm": {"name": "domino", "domino": {"pure_draft_prefix_len": 8}}}
        )


def test_domino_checkpoint_metadata_and_inference_module():
    algorithm = get_algorithm("domino")
    module = _tiny_domino_module()
    draft_config = module.draft_model.config
    contract = {"layers": {"draft_layer_ids": [0, 1]}}
    cfg = {"drafter": {"mask_token_id": 255, "block_size": 8}, "algorithm": {"name": "domino"}}
    meta = algorithm.checkpoint_metadata(contract, cfg, draft_config)
    assert meta["domino"]["emb_dim"] == 16
    assert meta["domino"]["shift_label"] is True
    assert meta["domino"]["pure_draft_prefix_len"] == 1
    decoder = algorithm.build_inference_module(draft_config.to_dict(), "sdpa")
    assert isinstance(decoder, Qwen3DFlashDraftModel)
    assert decoder.projector_type == "domino"
    assert decoder.shift_label is True


def test_domino_trainer_validation_gates():
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_domino_module()
    # Mismatched slot conventions must fail loud (the shared machinery would
    # else supervise a shifted window with unshifted labels).
    with pytest.raises(ValueError, match="sample_from_anchor"):
        DominoTrainerModule(
            draft_model=module.draft_model,
            target_lm_head=module.lm_head,
            target_embed_tokens=module.embed_tokens,
            mask_token_id=255,
            block_size=8,
            attention_backend="sdpa",
            num_anchors=16,
            loss_decay_gamma=7.0,
            sample_from_anchor=False,
        )
    with pytest.raises(ValueError, match="lambda_base_decay_ratio"):
        DominoTrainerModule(
            draft_model=module.draft_model,
            target_lm_head=module.lm_head,
            target_embed_tokens=module.embed_tokens,
            mask_token_id=255,
            block_size=8,
            attention_backend="sdpa",
            num_anchors=16,
            loss_decay_gamma=7.0,
            sample_from_anchor=True,
            lambda_base_decay_ratio=1.5,
        )


@torch.no_grad()
def test_domino_spec_generate_runs_in_shifted_mode():
    """Runtime smoke: the ported Domino in-block correction path must produce tokens."""
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(11)
    target_config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    target_config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(target_config)
    target.eval()

    draft = _tiny_domino_module(shift_label=True).draft_model
    draft.eval()
    torch.manual_seed(5)
    input_ids = torch.randint(0, 256, (1, 12))
    ids, stats = draft.spec_generate(
        target=target,
        input_ids=input_ids,
        max_new_tokens=16,
        stop_token_ids=None,
        temperature=0.0,
        return_stats=True,
    )
    assert ids.shape[1] == input_ids.shape[1] + 16
    assert stats["verify_steps"] > 0
