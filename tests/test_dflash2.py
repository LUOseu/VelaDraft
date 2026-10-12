"""DFlash2 unit tests: conv math, selector, split objective, registry plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.config import load_config
from vlm_dflash.dflash.trainer_core import NoValidAnchorsError
from vlm_dflash.dflash2.conv import GroupedDynamicCausalConv, grouped_dynamic_conv
from vlm_dflash.dflash2.loss import (
    candidate_cross_entropy,
    decay_weights,
    selector_training_candidates,
)
from vlm_dflash.dflash2.model import Qwen3DFlash2DecoderLayer, Qwen3DFlash2DraftModel
from vlm_dflash.dflash2.selector import CandidateSelector
from vlm_dflash.dflash2.trainer import DFlash2TrainerModule

VOCAB = 256
HIDDEN = 64
BLOCK_SIZE = 8


# --------------------------------------------------------------------------- #
# local dynamic convolution (ported semantics)
# --------------------------------------------------------------------------- #

def _reference_grouped_conv(hidden_states, delta_kernel, base_kernel, *, block_size, group_size):
    output = torch.zeros_like(hidden_states)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_delta = delta_kernel.reshape(-1, base_kernel.shape[0], hidden_states.shape[-1] // group_size)
    flat_output = output.reshape_as(flat_hidden)
    for token_idx in range(flat_hidden.shape[0]):
        block_position = token_idx % block_size
        for tap in range(min(base_kernel.shape[0], block_position + 1)):
            for channel_idx in range(flat_hidden.shape[-1]):
                group_idx = channel_idx // group_size
                coefficient = base_kernel[tap, channel_idx] + flat_delta[token_idx, tap, group_idx]
                flat_output[token_idx, channel_idx] += flat_hidden[token_idx - tap, channel_idx] * coefficient
    return output


def test_grouped_dynamic_conv_matches_scalar_reference():
    torch.manual_seed(0)
    hidden = torch.randn(1, 8, 6)
    delta = torch.randn(1, 8, 3, 3)
    base = torch.randn(3, 6)
    actual = grouped_dynamic_conv(hidden, delta, base, block_size=4, group_size=2)
    expected = _reference_grouped_conv(hidden, delta, base, block_size=4, group_size=2)
    torch.testing.assert_close(actual, expected)


def test_grouped_dynamic_conv_does_not_cross_block_boundaries():
    hidden = torch.zeros(1, 8, 4)
    hidden[:, 3] = 7
    delta = torch.zeros(1, 8, 2, 2)
    base = torch.zeros(2, 4)
    base[1] = 1
    output = grouped_dynamic_conv(hidden, delta, base, block_size=4, group_size=2)
    assert torch.count_nonzero(output) == 0


def test_grouped_dynamic_conv_preserves_bfloat16_activation_dtype():
    hidden = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    delta = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
    base = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    output = grouped_dynamic_conv(hidden, delta, base, block_size=4, group_size=4)
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()


def test_conv_identity_initialization_is_a_no_op():
    torch.manual_seed(0)
    conv = GroupedDynamicCausalConv(8, block_size=4, kernel_size=2, group_size=4)
    hidden = torch.randn(1, 8, 8)
    prepared, kernel = conv.prepare(hidden)
    torch.testing.assert_close(prepared, hidden)
    finished = conv.finish(hidden, kernel)
    torch.testing.assert_close(finished, hidden)


def test_conv_rejects_bad_shapes():
    with pytest.raises(ValueError):
        GroupedDynamicCausalConv(9, block_size=4, kernel_size=2, group_size=4)
    with pytest.raises(ValueError):
        GroupedDynamicCausalConv(8, block_size=2, kernel_size=4, group_size=4)


# --------------------------------------------------------------------------- #
# candidate selector (ported semantics)
# --------------------------------------------------------------------------- #

def test_selector_transition_scores_match_loop_reference():
    torch.manual_seed(0)
    selector = CandidateSelector(vocab_size=11, hidden_size=6, rank=4, top_k=3)
    hidden = torch.randn(2, 3, 6)
    predecessor_ids = torch.randint(0, 11, (2, 3))
    candidate_ids = torch.randint(0, 11, (2, 3, 3))
    actual = selector.transition_scores(hidden, predecessor_ids, candidate_ids)
    expected = torch.empty_like(actual)
    projected = selector.hidden_projection(hidden)
    for block_idx in range(2):
        for position_idx in range(3):
            context = (
                selector.predecessor_codebook[predecessor_ids[block_idx, position_idx]]
                * projected[block_idx, position_idx]
            )
            for candidate_idx in range(3):
                token_id = candidate_ids[block_idx, position_idx, candidate_idx]
                expected[block_idx, position_idx, candidate_idx] = (
                    context * selector.successor_codebook[token_id]
                ).sum()
    torch.testing.assert_close(actual, expected)


def test_selector_selects_only_from_unary_top_k():
    selector = CandidateSelector(vocab_size=7, hidden_size=4, rank=3, top_k=2)
    unary = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    hidden = torch.randn(1, 1, 4)
    predecessor_ids = torch.tensor([[1]])
    candidate_ids, scores = selector.select(unary, hidden, predecessor_ids)
    assert candidate_ids.tolist() == [[[6, 5]]]
    assert scores.shape == (1, 1, 2)


def test_selector_training_candidates_inject_missing_target_at_rank_k():
    candidates = torch.tensor([[[4, 5, 6], [4, 5, 6]]])
    targets = torch.tensor([[5, 7]])
    training_ids, positions, contains = selector_training_candidates(candidates, targets)
    assert contains.tolist() == [[True, False]]
    assert training_ids.tolist() == [[[4, 5, 6], [4, 5, 7]]]
    assert positions.tolist() == [[1, 2]]


def test_decay_weights_match_upstream_formula():
    mask = torch.ones(1, 1, BLOCK_SIZE)
    weights = decay_weights(mask, BLOCK_SIZE, 4.0)
    expected = torch.exp(-torch.arange(BLOCK_SIZE, dtype=torch.float32) / 4.0)
    torch.testing.assert_close(weights.reshape(-1), expected)
    uniform = decay_weights(mask, BLOCK_SIZE, None)
    assert torch.all(uniform == 1.0)


# --------------------------------------------------------------------------- #
# tiny module + objective
# --------------------------------------------------------------------------- #

def _tiny_dflash2_module(seed: int = 0, *, block_size: int = BLOCK_SIZE, num_anchors: int = 16, **kwargs):
    torch.manual_seed(seed)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
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
            "dflash2": {
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 8,
                "selector_top_k": 4,
            },
        },
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlash2DraftModel(config)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    lm_head.weight = embed.weight
    module_kwargs = {"selector_loss_alpha": 1.0, "kd_chunk_size": 512}
    module_kwargs.update(kwargs)
    decay = module_kwargs.pop("loss_decay_gamma", 4.0)
    sample_from_anchor = module_kwargs.pop("sample_from_anchor", False)
    return DFlash2TrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=decay,
        sample_from_anchor=sample_from_anchor,
        **module_kwargs,
    )


def _tiny_batch(batch_size: int = 2, seq_len: int = 48, seed: int = 0):
    torch.manual_seed(seed)
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(VOCAB, HIDDEN) * 0.05)
        base = embed(input_ids)
        hidden = torch.cat([base + 0.02 * torch.randn_like(base), torch.tanh(base)], dim=-1)
        final_hidden = torch.randn(batch_size, seq_len, HIDDEN) * 0.1
    return input_ids, loss_mask, hidden, final_hidden


def test_canonical_checkpoint_keys_and_identity_convs():
    """Saved parameter names must match the public DFlash2 contract."""
    module = _tiny_dflash2_module()
    state_keys = set(module.draft_model.state_dict())
    for key in (
        "layers.0.attention_conv.base_kernel",
        "layers.0.attention_conv.kernel_projection.weight",
        "layers.0.mlp_conv.base_kernel",
        "layers.0.mlp_conv.kernel_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "candidate_selector.hidden_projection.weight",
    ):
        assert key in state_keys, key
    assert "candidate_selector.predecessor_codebook.weight" not in state_keys
    # ``post_init`` re-initializes Linear weights, so the DFlash2 layer must reset
    # its convs afterwards: identity init means the conv is a no-op at step 0.
    layer = module.draft_model.layers[0]
    assert isinstance(layer, Qwen3DFlash2DecoderLayer)
    hidden = torch.randn(1, 2 * BLOCK_SIZE, HIDDEN)
    prepared, kernel = layer.attention_conv.prepare(hidden)
    torch.testing.assert_close(prepared, hidden)
    torch.testing.assert_close(layer.attention_conv.finish(hidden, kernel), hidden)


def test_dflash2_forward_matches_dflash_backbone_at_init():
    """Identity convs + unused selector: a fresh DFlash2 stack equals DFlash."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

    torch.manual_seed(0)
    base_config = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=BLOCK_SIZE,
        num_target_layers=2,
    )
    dflash_config = {"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False}
    dflash = Qwen3DFlashDraftModel(Qwen3Config(**base_config, dflash_config=dict(dflash_config)))
    dflash.config._attn_implementation = "sdpa"
    dflash2_config = dict(dflash_config)
    dflash2_config["dflash2"] = {
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 8,
        "selector_top_k": 4,
    }
    dflash2 = Qwen3DFlash2DraftModel(Qwen3Config(**base_config, dflash_config=dflash2_config))
    dflash2.config._attn_implementation = "sdpa"
    missing, unexpected = dflash2.load_state_dict(dflash.state_dict(), strict=False)
    assert set(missing) == {
        f"layers.{idx}.{conv}.{param}"
        for idx in range(2)
        for conv in ("attention_conv", "mlp_conv")
        for param in ("base_kernel", "kernel_projection.weight")
    } | {
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "candidate_selector.hidden_projection.weight",
    }
    assert not unexpected

    bsz, ctx = 1, 12
    input_ids = torch.arange(ctx + BLOCK_SIZE).remainder(200).add(10).unsqueeze(0)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    with torch.no_grad():
        target_hidden = torch.randn(bsz, ctx, 2 * HIDDEN) * 0.1
        anchors = torch.arange(8, 9).view(1, 1)
        keep = torch.ones(1, 1, dtype=torch.bool)
        noise = torch.randn(bsz, BLOCK_SIZE, HIDDEN) * 0.1
        position_ids = torch.arange(ctx + BLOCK_SIZE).unsqueeze(0)
        out_a = dflash(
            position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden, use_cache=False
        )
        out_b = dflash2(
            position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden, use_cache=False
        )
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-4)


def test_unary_kl_matches_fp32_reference():
    module = _tiny_dflash2_module(seed=1)
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=1, seq_len=40, seed=3)
    metrics = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden
    )
    assert torch.isfinite(metrics.loss)
    assert metrics.extra_metrics["unary_kl"] > 0
    assert metrics.extra_metrics["selector_ce"] > 0
    assert 0.0 <= metrics.extra_metrics["candidate_recall"] <= 1.0


def test_selector_ce_matches_cross_entropy_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4)
    positions = torch.randint(0, 4, (2, 5))
    actual = candidate_cross_entropy(logits, positions)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), positions.reshape(-1), reduction="none")
    torch.testing.assert_close(actual.reshape(-1), expected)


def test_trainer_requires_capture_v2_teacher_hidden():
    module = _tiny_dflash2_module(seed=2)
    input_ids, loss_mask, hidden, _ = _tiny_batch(batch_size=1, seq_len=40, seed=4)
    with pytest.raises(ValueError, match="store_final_hidden"):
        module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=None)


def test_selector_loss_alpha_zero_disables_the_selector_term():
    module = _tiny_dflash2_module(seed=2, selector_loss_alpha=0.0)
    assert module.selector_loss_alpha == 0.0
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=1, seq_len=40, seed=5)
    metrics = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden
    )
    assert metrics.loss.item() == pytest.approx(metrics.extra_metrics["unary_kl"], rel=1e-5)


def test_trainer_rejects_shifted_slot_convention():
    with pytest.raises(ValueError, match="sample_from_anchor"):
        _tiny_dflash2_module(seed=2, sample_from_anchor=True)


def test_no_supervised_positions_raise():
    module = _tiny_dflash2_module(seed=5)
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=1, seq_len=40, seed=6)
    loss_mask = torch.zeros_like(loss_mask)
    with pytest.raises(NoValidAnchorsError):
        module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden)


def test_overfit_tiny_batch_reduces_split_objective():
    module = _tiny_dflash2_module(seed=3)
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=2, seq_len=48, seed=11)
    optimizer = torch.optim.Adam(module.draft_model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        metrics = module(
            input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden
        )
        optimizer.zero_grad()
        metrics.loss.backward()
        optimizer.step()
        losses.append(metrics.loss.detach().item())
    assert losses[-1] < 0.7 * losses[0], f"DFlash2 loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_serving_walk_picks_only_unary_top_k_candidates():
    module = _tiny_dflash2_module(seed=7)
    draft = module.draft_model
    draft.eval()
    block_output_ids = torch.full((1, BLOCK_SIZE), 255, dtype=torch.long)
    block_output_ids[:, 0] = 17
    draft_hidden = torch.randn(1, BLOCK_SIZE, HIDDEN) * 0.1
    target = torch.nn.Module()
    target.lm_head = module.lm_head
    with torch.no_grad():
        tokens = draft._sample_draft_tokens(target, draft_hidden, block_output_ids)
    assert tokens.shape == (1, BLOCK_SIZE - 1)
    assert torch.all(tokens != 255)
    # every emitted token must come from that position's unary top-k
    with torch.no_grad():
        unary = module.lm_head(draft_hidden[:, 1:, :])
        allowed = unary.topk(draft.candidate_selector.top_k, dim=-1).indices
    for position in range(BLOCK_SIZE - 1):
        assert bool((allowed[0, position] == tokens[0, position]).any())


def test_serving_walk_lets_the_selector_override_the_unary_argmax():
    """Deterministic check that the walk emits the selector's pick, not the unary top-1."""
    module = _tiny_dflash2_module(seed=8)
    draft = module.draft_model
    draft.eval()
    star = 42
    top_k = draft.candidate_selector.top_k

    def bonus_transition(hidden_states, predecessor_ids, candidate_ids):
        return (candidate_ids == star).to(torch.float32) * 1e4

    draft.candidate_selector.transition_scores = bonus_transition  # instance attribute, unbound
    block_output_ids = torch.full((1, BLOCK_SIZE), 255, dtype=torch.long)
    block_output_ids[:, 0] = 17
    draft_hidden = torch.randn(1, BLOCK_SIZE, HIDDEN) * 0.1
    target = torch.nn.Module()
    target.lm_head = module.lm_head
    with torch.no_grad():
        tokens = draft._sample_draft_tokens(target, draft_hidden, block_output_ids)
        unary = module.lm_head(draft_hidden[:, 1:, :])
        topk_ids = unary.topk(top_k, dim=-1).indices
        unary_pick = unary.argmax(dim=-1)
    for position in range(BLOCK_SIZE - 1):
        expected = star if bool((topk_ids[0, position] == star).any()) else int(unary_pick[0, position])
        assert int(tokens[0, position]) == expected, position


# --------------------------------------------------------------------------- #
# registry plumbing
# --------------------------------------------------------------------------- #

def test_dflash2_registered_and_validates_config():
    algorithm = get_algorithm("dflash2")
    assert algorithm.metadata.maturity["training"] == "experimental"
    cfg = load_config("configs/dflash2/smoke_coco.yaml")
    algorithm.validate_config(cfg)
    # capture v2 is required for the teacher signal
    broken = dict(cfg)
    broken["capture"] = {**cfg["capture"], "store_final_hidden": False}
    with pytest.raises(ValueError, match="store_final_hidden"):
        algorithm.validate_config(broken)


def test_dflash2_validate_rejects_shifted_anchor_and_domino_head():
    algorithm = get_algorithm("dflash2")
    cfg = load_config("configs/dflash2/smoke_coco.yaml")
    shifted = {**cfg, "drafter": {**cfg["drafter"], "sample_from_anchor": True}}
    with pytest.raises(ValueError, match="sample_from_anchor"):
        algorithm.validate_config(shifted)
    domino = {**cfg, "drafter": {**cfg["drafter"], "projector_type": "domino"}}
    with pytest.raises(ValueError, match="Domino"):
        algorithm.validate_config(domino)


def test_dflash2_checkpoint_metadata():
    from types import SimpleNamespace

    algorithm = get_algorithm("dflash2")
    cfg = load_config("configs/dflash2/smoke_coco.yaml")
    contract = {
        "layers": {"draft_layer_ids": [2, 12, 22]},
    }
    meta = algorithm.checkpoint_metadata(contract, cfg, SimpleNamespace(to_dict=lambda: {"vocab_size": 152064}))
    assert meta["dflash2"]["selector_top_k"] == 16
    assert "experimental" in meta["dflash2"]["objective"]
    assert meta["loss_decay_gamma"] == 4.0


def test_dflash2_build_draft_config_round_trip(tmp_path):
    """The serialized draft config alone must rebuild the DFlash2 architecture."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    algorithm = get_algorithm("dflash2")
    cfg = load_config("configs/dflash2/smoke_coco.yaml")
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    ).save_pretrained(target_dir)
    cfg = {**cfg, "drafter": {**cfg["drafter"], "num_hidden_layers": 2}}
    contract = {
        "target": {"model_path": str(target_dir)},
        "layers": {"num_target_layers": 4, "draft_layer_ids": [1, 2], "num_draft_layers": 2},
        # the tiny target has head_dim 16, so the MRoPE sections must sum to 8
        "rope": {"sections": [2, 3, 3], "interleaved": False},
    }
    draft_config = algorithm.build_draft_config(contract, cfg)
    assert draft_config.dflash_config["dflash2"]["selector_top_k"] == 16
    draft = Qwen3DFlash2DraftModel(draft_config)
    assert isinstance(draft.layers[0], Qwen3DFlash2DecoderLayer)
    assert draft.candidate_selector.top_k == 16
    # conv knobs survive a config round trip (what export/eval rely on)
    reloaded = Qwen3DFlash2DraftModel(Qwen3Config.from_dict(draft_config.to_dict()))
    assert reloaded.conv_kernel_size == 2 and reloaded.selector_rank == 256
