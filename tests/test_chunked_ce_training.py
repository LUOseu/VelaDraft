"""Bound vocabulary activation memory without changing the block CE objective."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_trainer_smoke import _tiny_module
from tests.test_recipe_row_compaction import _batch
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule


def _module(chunk, algorithm="dflash", backward="checkpoint"):
    if algorithm == "dfly":
        from tests.test_dfly import _tiny_module as dfly_module
        base, _ = dfly_module(seed=11)
    else:
        base = _tiny_module(seed=11)
    base.lm_head.requires_grad_(False)
    return type(base)(base.draft_model, base.lm_head, base.embed_tokens, 255,
                      block_size=8, num_anchors=16, attention_backend="sdpa",
                      loss_decay_gamma=7.0, ce_chunk_size=chunk, ce_backward=backward)


@pytest.mark.parametrize("chunk", [1, 13, 1024])
@pytest.mark.parametrize("algorithm", ["dflash", "dfly"])
@pytest.mark.parametrize("backward", ["checkpoint", "precompute"])
def test_chunked_ce_preserves_loss_metrics_and_all_gradients(chunk, algorithm, backward):
    if algorithm == "dfly":
        from tests.test_dfly import _tiny_batch
        ids, mask, hidden = _tiny_batch(seed=9)
        mask[1, 20:] = 0
    else:
        ids, mask, hidden, _ = _batch()

    def run(size):
        module = _module(size, algorithm, backward if size else "checkpoint")
        torch.manual_seed(4)
        metrics = module(input_ids=ids, hidden_states=hidden, loss_mask=mask,
                         sample_weights=torch.linspace(0.3, 1.7, ids.shape[0]))
        (metrics.loss / 3).backward()
        return metrics, {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}, module.last_workload

    ref, ref_grad, _ = run(0)
    out, grad, counts = run(chunk)
    for field in ("loss", "accuracy", "valid_tokens", "correct_tokens", "loss_weight", "accept_len",
                  "accept_len_sum", "valid_blocks", "pos_count", "pos_correct"):
        torch.testing.assert_close(getattr(out, field), getattr(ref, field), rtol=1e-5, atol=2e-6)
    assert set(grad) == set(ref_grad)
    for name, value in ref_grad.items():
        torch.testing.assert_close(grad[name], value, rtol=1e-4, atol=1e-6, msg=name)
    assert counts["scored_rows"] == counts["effective_positions"]
    assert counts["student_head_rows"] == (2 if backward == "checkpoint" else 1) * counts["effective_positions"]


def test_ce_switch_rejects_recipe_and_variable_prefix():
    from vlm_dflash.objectives import resolve_recipe
    base = _tiny_module()
    kwargs = dict(draft_model=base.draft_model, target_lm_head=base.lm_head,
                  target_embed_tokens=base.embed_tokens, mask_token_id=255, ce_chunk_size=4)
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dflash")
    with pytest.raises(ValueError, match="ce_chunk_size"):
        DFlashTrainerModule(**kwargs, train_recipe=recipe)
    with pytest.raises(ValueError, match="ce_chunk_size"):
        DFlashTrainerModule(**kwargs, loss_type="variable_prefix")


def test_all_zero_sample_weights_have_zero_connected_loss():
    module = _module(4)
    ids, mask, hidden, _ = _batch()
    out = module(input_ids=ids, hidden_states=hidden, loss_mask=mask, sample_weights=torch.zeros(3))
    out.loss.backward()
    assert out.loss.item() == 0
    assert module.last_workload.get("student_head_rows", 0) == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in module.parameters())


@pytest.mark.parametrize("name", ["dspark", "dflash2", "jetspec", "domino"])
def test_config_rejects_custom_loss_algorithms(name):
    from vlm_dflash.config import validate_runtime_config
    with pytest.raises(ValueError, match="ce_chunk_size"):
        validate_runtime_config({"algorithm": name, "drafter": {"ce_chunk_size": 16}})


def test_benchmark_gradient_gate_rejects_missing_and_nonfinite_gradients():
    from scripts.benchmark_teacher_projection import gradient_drift
    ref = {"a": torch.ones(2), "b": torch.ones(1)}
    assert not gradient_drift(ref, {"a": torch.ones(2)})["complete_and_finite"]
    assert not gradient_drift(ref, {"a": torch.full((2,), float("nan")), "b": torch.ones(1)})["complete_and_finite"]
    assert gradient_drift(ref, ref)["complete_and_finite"]


def test_precomputed_ce_retain_graph_and_scale_are_not_inplace():
    module = _module(7, backward="precompute")
    ids, mask, hidden, _ = _batch()
    out = module(input_ids=ids, hidden_states=hidden, loss_mask=mask)
    p = next(module.draft_model.parameters())
    g1 = torch.autograd.grad(out.loss * 0.37, p, retain_graph=True)[0]
    g2 = torch.autograd.grad(out.loss * 0.74, p)[0]
    torch.testing.assert_close(g2, 2 * g1)


def test_precompute_rejects_trainable_head():
    module = _module(7, backward="precompute")
    module.lm_head.weight.requires_grad_(True)
    ids, mask, hidden, _ = _batch()
    with pytest.raises(ValueError, match="frozen"):
        module(input_ids=ids, hidden_states=hidden, loss_mask=mask)


def test_precompute_handles_noncontiguous_hidden_without_losing_gradient_writes():
    from vlm_dflash.dflash.chunked_ce import precomputed_block_ce
    from vlm_dflash.dflash.decay_loss import DFlashDecayLoss
    torch.manual_seed(16)
    hidden = torch.randn(2, 12, 21, dtype=torch.float64).transpose(1, 2).requires_grad_()
    assert not hidden.is_contiguous()
    head = torch.nn.Linear(12, 37).double().requires_grad_(False)
    targets = torch.randint(0, 37, (2, 21))
    mask = torch.rand(2, 21, dtype=torch.float64)
    fn = DFlashDecayLoss(normalize="mean")
    expected = fn(head(hidden), targets, mask, block_size=8).total_loss
    out, _ = precomputed_block_ce(hidden, head, targets, mask, loss_fn=fn, block_size=8,
                                  chunk_size=7, counter=lambda **kw: None)
    torch.testing.assert_close(out.total_loss, expected)
    a = torch.autograd.grad(out.total_loss * 0.37, hidden)[0]
    b = torch.autograd.grad(expected * 0.37, hidden)[0]
    torch.testing.assert_close(a, b)
