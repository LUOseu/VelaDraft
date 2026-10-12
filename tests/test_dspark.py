"""DSpark unit tests: heads, trainer overfit, gradient coverage, calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.dspark.factory import resolve_dspark_config
from vlm_dflash.dspark.heads import (
    ConfidenceHead,
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov,
    build_markov_head,
)
from vlm_dflash.dspark.model import Qwen3DSparkDraftModel
from vlm_dflash.dspark.trainer import DSparkTrainerModule
from vlm_dflash.utils import gradient_coverage_report


def _tiny_dspark_module(
    seed: int = 0, *, sample_from_anchor: bool = True, markov_rank: int = 32
) -> DSparkTrainerModule:
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
        block_size=8,
        num_target_layers=2,
        dflash_config={
            "mask_token_id": 255,
            "target_layer_ids": [0, 1],
            "spatial_rope_enabled": False,
            "markov_rank": 32,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "sample_from_anchor": sample_from_anchor,
        },
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DSparkDraftModel(config)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    return DSparkTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
        markov_rank=markov_rank,
        markov_head_type="vanilla",
        sample_from_anchor=sample_from_anchor,
    )


def _tiny_batch(batch_size: int = 2, seq_len: int = 48, seed: int = 0):
    torch.manual_seed(seed)
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    embed = torch.nn.Embedding(256, 64)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(256, 64) * 0.05)
        # Captured target features never carry a graph; build them under
        # no_grad so repeated training steps can reuse the same batch.
        base = embed(input_ids)
        hidden = torch.cat([base + 0.02 * torch.randn_like(base), torch.tanh(base)], dim=-1)
    return input_ids, loss_mask, hidden


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #

def test_vanilla_markov_bias_matches_manual_factorization():
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=32, markov_rank=4)
    token_ids = torch.randint(0, 32, (2, 3, 8))
    hidden = torch.zeros(2, 3, 8, 16)
    bias = head.compute_block_bias(prev_token_ids=token_ids, hidden_states=hidden)
    manual = head.markov_w2(head.markov_w1(token_ids.long()))
    assert bias.shape == (2, 3, 8, 32)
    assert torch.allclose(bias, manual, atol=1e-6)
    base_logits = torch.zeros(2, 3, 8, 32)
    corrected = head.apply_block_logits(base_logits, token_ids=token_ids, hidden_states=hidden)
    assert torch.allclose(corrected, manual, atol=1e-6)


def test_markov_head_w1_small_init_keeps_step0_bias_bounded():
    head = VanillaMarkov(vocab_size=64, markov_rank=16)
    token_ids = torch.randint(0, 64, (1, 1, 8))
    bias = head.compute_block_bias(prev_token_ids=token_ids, hidden_states=None)
    assert float(bias.detach().abs().max()) < 0.1


def test_gated_and_rnn_heads_forward_and_grad_flow():
    torch.manual_seed(0)
    for head in (
        GatedMarkovHead(vocab_size=32, markov_rank=4, hidden_size=16),
        RNNHead(vocab_size=32, markov_rank=4, hidden_size=16),
    ):
        token_ids = torch.randint(0, 32, (2, 3, 8))
        hidden = torch.randn(2, 3, 8, 16)
        bias = head.compute_block_bias(prev_token_ids=token_ids, hidden_states=hidden)
        assert bias.shape == (2, 3, 8, 32)
        loss = bias.float().pow(2).mean()
        loss.backward()
        w1_grad = head.markov_w1.weight.grad
        assert w1_grad is not None and float(w1_grad.abs().sum()) > 0
        assert float(head.markov_w2.weight.grad.abs().sum()) > 0


def test_sample_block_tokens_chains_the_previous_sampled_token():
    """A bias that favors token X after token Y must pick X first, then Y again."""
    head = VanillaMarkov(vocab_size=8, markov_rank=2)
    with torch.no_grad():
        head.markov_w1.weight.zero_()
        head.markov_w2.weight.zero_()
        # prev token 5 -> rank basis e0; rank basis e0 -> +10 logit on token 3.
        head.markov_w1.weight[5, 0] = 1.0
        head.markov_w2.weight[3, 0] = 10.0
    base_logits = torch.zeros(1, 4, 8)  # uniform base: the bias decides
    first_prev = torch.tensor([5])
    hidden = torch.zeros(1, 4, 16)
    tokens, corrected = head.sample_block_tokens(
        base_logits, first_prev_token_ids=first_prev, hidden_states=hidden
    )
    assert tokens.shape == (1, 4)
    assert int(tokens[0, 0]) == 3  # first slot conditioned on prev=5
    # Second slot conditions on the sampled token 3 (no bias) -> argmax 0.
    assert int(tokens[0, 1]) == 0
    assert corrected.shape == (1, 4, 8)


def test_build_markov_head_variants_and_disable():
    assert build_markov_head(vocab_size=32, markov_rank=0) is None
    assert isinstance(build_markov_head(vocab_size=32, markov_rank=4), VanillaMarkov)
    assert isinstance(
        build_markov_head(vocab_size=32, markov_rank=4, markov_head_type="gated", hidden_size=16),
        GatedMarkovHead,
    )
    assert isinstance(
        build_markov_head(vocab_size=32, markov_rank=4, markov_head_type="rnn", hidden_size=16),
        RNNHead,
    )
    with pytest.raises(ValueError, match="markov_head_type"):
        build_markov_head(vocab_size=32, markov_rank=4, markov_head_type="transformer")


def test_confidence_head_shapes():
    head = ConfidenceHead(12)
    out = head(torch.zeros(2, 3, 12))
    assert out.shape == (2, 3)


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #

def _run_steps(module, input_ids, loss_mask, hidden, steps: int = 40, lr: float = 2e-3):
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
        optimizer.zero_grad()
        metrics.loss.backward()
        optimizer.step()
        assert torch.isfinite(metrics.loss)
        losses.append(float(metrics.loss.detach()))
    return metrics, losses


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_dspark_trainer_overfits_and_heads_receive_gradient(sample_from_anchor):
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_dspark_module(sample_from_anchor=sample_from_anchor)
    metrics, losses = _run_steps(module, input_ids, loss_mask, hidden, steps=60)
    assert losses[-1] < 0.8 * losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    # GOAL §12: both DSpark-specific heads must actually receive gradients.
    markov_grad = module.markov_head.markov_w2.weight.grad
    conf_grad = module.confidence_head.proj.weight.grad
    assert markov_grad is not None and float(markov_grad.abs().sum()) > 0
    assert conf_grad is not None and float(conf_grad.abs().sum()) > 0
    # Confidence diagnostics were reported and are not constant.
    assert metrics.extra_metrics is not None
    assert "confidence_bce" in metrics.extra_metrics
    assert "confidence_pred_mean" in metrics.extra_metrics
    assert "markov_bias_abs_mean" in metrics.extra_metrics
    # Slot convention: sample_from_anchor supervises all block slots.
    expected_slots = 8 if sample_from_anchor else 7
    assert module.first_supervised_slot == (0 if sample_from_anchor else 1)
    assert metrics.pos_count.shape[0] == expected_slots


def test_dspark_markov_bias_moves_logits_and_loss():
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_dspark_module()
    with torch.no_grad():
        # Force a large Markov bias so the corrected logits differ measurably.
        module.markov_head.markov_w2.weight.normal_(std=0.5)
    metrics_full = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    baseline = _tiny_dspark_module()
    baseline.load_state_dict(module.state_dict())
    with torch.no_grad():
        baseline.markov_head.markov_w2.weight.zero_()
    metrics_zero = baseline(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert not torch.allclose(metrics_full.loss, metrics_zero.loss)
    assert float(metrics_full.extra_metrics["markov_bias_abs_mean"]) > 0
    assert float(metrics_zero.extra_metrics["markov_bias_abs_mean"]) == 0.0


def test_gradient_coverage_report_flags_unused_head():
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_dspark_module()
    # One backward with the confidence term disabled: the confidence head must
    # show zero *nonzero* coverage in the audit (a 0-weighted term still
    # backprops zero-valued grads, so grad_norm is the decisive signal).
    module.confidence_head_alpha = 0.0
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    metrics.loss.backward()
    report = gradient_coverage_report(module)
    groups = {group["group"]: group for group in report["groups"]}
    assert report["trainable_params"] > 0
    assert groups["draft_model.markov_head"]["params_with_nonzero_grad"] > 0
    assert groups["draft_model.markov_head"]["grad_norm"] > 0
    assert groups["draft_model.confidence_head"]["params_with_nonzero_grad"] == 0
    assert groups["draft_model.confidence_head"]["grad_norm"] == 0.0
    assert groups["draft_model.confidence_head"]["trainable_params"] > 0
    # The backbone is reported under its own group, not merged with the heads.
    assert "draft_model.layers" in groups


def test_trainer_heads_are_owned_by_the_draft_model():
    """Regression: a duplicated trainer-side head is trained on nothing.

    The training loop casts ``module.draft_model`` to bf16 and builds the
    optimizer / checkpoint from ``draft_model.named_parameters()`` and
    ``state_dict()``. A head owned by the trainer module instead would keep
    fp32 weights (dtype crash against bf16 features) and stay at its random
    init while never being saved. The trainer must expose the model's
    instances.
    """
    module = _tiny_dspark_module()
    assert module.markov_head is module.draft_model.markov_head
    assert module.confidence_head is module.draft_model.confidence_head
    parameter_names = {name for name, _ in module.draft_model.named_parameters()}
    assert any(name.startswith("markov_head.") for name in parameter_names)
    assert any(name.startswith("confidence_head.") for name in parameter_names)
    state_keys = set(module.draft_model.state_dict())
    assert any(key.startswith("markov_head.") for key in state_keys)
    assert any(key.startswith("confidence_head.") for key in state_keys)


def test_dtype_cast_of_the_draft_model_reaches_the_heads():
    """Mirror the training-loop cast that crashed with fp32 heads + bf16 features."""
    input_ids, loss_mask, hidden = _tiny_batch()
    module = _tiny_dspark_module()
    module.draft_model.to(torch.bfloat16)
    # In a real run the frozen target embed/lm_head are built from the captured
    # (bf16) feature file, so they are bf16 alongside the draft.
    module.lm_head.to(torch.bfloat16)
    module.embed_tokens.to(torch.bfloat16)
    assert module.markov_head.markov_w2.weight.dtype == torch.bfloat16
    assert module.confidence_head.proj.weight.dtype == torch.bfloat16
    metrics = module(input_ids=input_ids, hidden_states=hidden.to(torch.bfloat16), loss_mask=loss_mask)
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    assert module.markov_head.markov_w2.weight.grad is not None
    assert module.confidence_head.proj.weight.grad is not None


def test_trainer_rejects_head_knobs_that_disagree_with_the_model():
    with pytest.raises(ValueError, match="markov_rank"):
        _tiny_dspark_module(markov_rank=8)


def test_trainer_requires_a_dspark_draft_model():
    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

    module = _tiny_dspark_module()
    plain = Qwen3DFlashDraftModel(module.draft_model.config)
    with pytest.raises(ValueError, match="heads live on the draft model"):
        DSparkTrainerModule(
            draft_model=plain,
            target_lm_head=module.lm_head,
            target_embed_tokens=module.embed_tokens,
            mask_token_id=255,
            block_size=8,
            attention_backend="sdpa",
            num_anchors=16,
        )


# --------------------------------------------------------------------------- #
# inference model
# --------------------------------------------------------------------------- #

def test_dspark_sampling_uses_markov_chain_from_anchor():
    module = _tiny_dspark_module(sample_from_anchor=False)
    draft = module.draft_model
    target = torch.nn.Module()
    with torch.no_grad():
        draft.markov_head.markov_w1.weight.zero_()
        draft.markov_head.markov_w2.weight.zero_()
        draft.markov_head.markov_w1.weight[7, 0] = 1.0
        draft.markov_head.markov_w2.weight[3, 0] = 10.0
        target.lm_head = module.lm_head
        target.get_input_embeddings = module.embed_tokens
    block_output_ids = torch.full((1, 8), draft.mask_token_id, dtype=torch.long)
    block_output_ids[:, 0] = 7  # anchor token
    draft_hidden = torch.zeros(1, 8, draft.config.hidden_size)
    tokens = draft._sample_draft_tokens(target, draft_hidden, block_output_ids)
    assert tokens.shape == (1, 7)
    assert int(tokens[0, 0]) == 3  # first proposal conditioned on the anchor token


def test_dspark_sample_from_anchor_returns_block_size_proposals():
    module = _tiny_dspark_module(sample_from_anchor=True)
    draft = module.draft_model
    target = torch.nn.Module()
    target.lm_head = module.lm_head
    target.get_input_embeddings = module.embed_tokens
    block_output_ids = torch.full((1, 8), draft.mask_token_id, dtype=torch.long)
    block_output_ids[:, 0] = 7
    draft_hidden = torch.zeros(1, 8, draft.config.hidden_size)
    tokens = draft._sample_draft_tokens(target, draft_hidden, block_output_ids)
    assert tokens.shape == (1, 8)


@torch.no_grad()
def test_dspark_spec_generate_runs_in_anchor_mode():
    """Regression: the round window is `block_size + 1` in anchor mode.

    ``spec_generate`` used to hand the full round window (9) to
    ``_sample_draft_tokens``, whose contract is the draft's own block (8); every
    DSpark eval round raised ``Expected a draft block of length 8, got 9``.
    """
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

    draft = _tiny_dspark_module(sample_from_anchor=True).draft_model
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


# --------------------------------------------------------------------------- #
# calibration report
# --------------------------------------------------------------------------- #

def test_calibration_report_reports_nonconstant_confidence():
    input_ids, loss_mask, hidden = _tiny_batch(batch_size=4, seq_len=48)
    module = _tiny_dspark_module()
    metrics, _ = _run_steps(module, input_ids, loss_mask, hidden, steps=5)
    batch = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden": hidden,
        "image_mask": torch.zeros_like(loss_mask, dtype=torch.bool),
        "position3": torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long),
    }
    report = module.calibration_report([batch], torch.device("cpu"))
    assert report is not None
    assert report["supervised_slots"] == 8
    assert report["confidence_pred_std"] > 0
    assert report["confidence_auroc"] is None or 0.0 <= report["confidence_auroc"] <= 1.0
    assert len(report["per_slot_pred_mean"]) == 8


def test_calibration_report_handles_varying_anchor_counts():
    """Anchor counts vary per batch ([B, N_i, slots]); the sink must flatten them.

    The canonical 2K run crashed here: ``torch.cat`` over stored tensors with
    different N (e.g. 48 vs 41) raised a size mismatch, aborting the run after
    training had already finished.
    """
    module = _tiny_dspark_module()
    input_ids_a, loss_mask_a, hidden_a = _tiny_batch(batch_size=2, seq_len=48)
    # A shorter sequence samples fewer anchors, so N differs between batches.
    input_ids_b, loss_mask_b, hidden_b = _tiny_batch(batch_size=2, seq_len=24)

    def batch(input_ids, loss_mask, hidden):
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hidden": hidden,
            "image_mask": torch.zeros_like(loss_mask, dtype=torch.bool),
            "position3": torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long),
        }

    report = module.calibration_report(
        [batch(input_ids_a, loss_mask_a, hidden_a), batch(input_ids_b, loss_mask_b, hidden_b)],
        torch.device("cpu"),
    )
    assert report is not None
    assert report["supervised_slots"] == 8
    assert len(report["per_slot_pred_mean"]) == 8
    assert len(report["per_slot_accept_rate"]) == 8
    assert all(0.0 <= value <= 1.0 for value in report["per_slot_pred_mean"])


# --------------------------------------------------------------------------- #
# registry / config plumbing
# --------------------------------------------------------------------------- #

def test_dspark_algorithm_registered_with_namespaced_config():
    algorithm = get_algorithm("dspark")
    assert algorithm.name == "dspark"
    assert algorithm.metadata.maturity["training"] in {"stable", "validated", "experimental", "architecture-only"}
    # Namespaced config form resolves through the same path as the plain string.
    assert get_algorithm({"name": "dspark"}).name == "dspark"
    cfg = {
        "drafter": {},
        "algorithm": {"name": "dspark", "dspark": {"markov_rank": 64, "markov_head_type": "gated"}},
    }
    algorithm.validate_config(cfg)
    resolved = resolve_dspark_config(cfg)
    assert resolved["markov_rank"] == 64
    assert resolved["markov_head_type"] == "gated"
    assert resolved["sample_from_anchor"] is True  # upstream default
    with pytest.raises(ValueError, match="markov_head_type"):
        algorithm.validate_config({"drafter": {}, "algorithm": {"name": "dspark", "dspark": {"markov_head_type": "x"}}})
    with pytest.raises(ValueError, match="markov_rank"):
        algorithm.validate_config(
            {"drafter": {}, "algorithm": {"name": "dspark", "dspark": {"markov_rank": 0}}}
        )


def test_dspark_checkpoint_metadata_and_inference_module():
    algorithm = get_algorithm("dspark")
    module = _tiny_dspark_module()
    draft_config = module.draft_model.config
    contract = {"layers": {"draft_layer_ids": [0, 1]}}
    cfg = {"drafter": {"mask_token_id": 255, "block_size": 8}, "algorithm": {"name": "dspark"}}
    meta = algorithm.checkpoint_metadata(contract, cfg, draft_config)
    assert meta["dspark"]["markov_rank"] == 32
    assert meta["dspark"]["sample_from_anchor"] is True
    decoder = algorithm.build_inference_module(draft_config.to_dict(), "sdpa")
    assert isinstance(decoder, Qwen3DSparkDraftModel)
    assert decoder.sample_from_anchor is True
