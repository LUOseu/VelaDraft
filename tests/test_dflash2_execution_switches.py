"""DFlash2 execution switches (D1 projection reuse, D2 row compaction + fused KL) keep the objective."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_dflash2 import _tiny_batch, _tiny_dflash2_module

ARMS = {
    "d0": {},
    "d1": {"teacher_projection_reuse": True},
    "compaction_only": {"scoring_row_compaction": True},
    "d2": {"teacher_projection_reuse": True, "scoring_row_compaction": True, "fused_teacher_kl": True},
}


def _padded_batch():
    """Two samples with different supervised lengths, so the shorter one has padding blocks."""
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=2, seq_len=64, seed=7)
    loss_mask[1, 30:] = 0
    return input_ids, loss_mask, hidden, final_hidden


def _run(arm: str, *, scale: float = 0.25, kd_chunk_size: int = 5):
    module = _tiny_dflash2_module(seed=4, num_anchors=12, kd_chunk_size=kd_chunk_size, **ARMS[arm])
    input_ids, loss_mask, hidden, final_hidden = _padded_batch()
    torch.manual_seed(123)
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden)
    (metrics.loss * scale).backward()
    grads = {name: p.grad.detach().clone() for name, p in module.draft_model.named_parameters() if p.grad is not None}
    return metrics, grads, dict(module.last_workload)


@pytest.mark.parametrize("arm", ["d1", "compaction_only", "d2"])
def test_switches_match_reference_loss_metrics_and_gradients(arm):
    ref, ref_grads, _ = _run("d0")
    out, grads, _ = _run(arm)
    torch.testing.assert_close(out.loss, ref.loss, rtol=1e-5, atol=1e-6)
    for key in ("unary_kl", "selector_ce", "candidate_recall", "unary_oracle_al"):
        assert out.extra_metrics[key] == pytest.approx(ref.extra_metrics[key], rel=1e-5, abs=1e-6), key
    for field in ("valid_tokens", "correct_tokens", "accept_len_sum", "valid_blocks", "pos_correct", "pos_count"):
        torch.testing.assert_close(getattr(out, field), getattr(ref, field))
    assert set(grads) == set(ref_grads)
    for name, grad in ref_grads.items():
        torch.testing.assert_close(grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_projection_counters():
    _, _, d0 = _run("d0")
    _, _, d1 = _run("d1")
    _, _, d2 = _run("d2")
    supervised = d0["supervised_rows"]
    assert d0["effective_positions"] == supervised
    # Reference projects every valid teacher row twice (argmax label + KL), D1 once.
    assert d0["teacher_head_rows"] == 2 * supervised
    assert d1["teacher_head_rows"] == supervised == d2["teacher_head_rows"]
    assert d1["teacher_head_calls"] * 2 == d0["teacher_head_calls"]
    # Without compaction the student head sees every allocated row; D2 only supervised rows.
    assert d0["student_head_rows"] == d0["allocated_rows"] == d1["student_head_rows"]
    assert d2["student_head_rows"] == supervised < d0["kept_rows"] <= d0["allocated_rows"]
    assert d0["unique_teacher_positions"] == d2["unique_teacher_positions"] <= supervised


def test_fused_kl_without_grad_matches_and_is_rejected_without_reuse():
    ref = _tiny_dflash2_module(seed=4, num_anchors=12, kd_chunk_size=5)
    d2 = _tiny_dflash2_module(seed=4, num_anchors=12, kd_chunk_size=5, **ARMS["d2"])
    input_ids, loss_mask, hidden, final_hidden = _padded_batch()
    with torch.no_grad():
        torch.manual_seed(5)
        expected = ref(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden)
        torch.manual_seed(5)
        actual = d2(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="teacher_projection_reuse"):
        _tiny_dflash2_module(seed=4, fused_teacher_kl=True)


def test_chunk_size_does_not_change_the_d2_objective():
    a, grads_a, _ = _run("d2", kd_chunk_size=3)
    b, grads_b, _ = _run("d2", kd_chunk_size=0)
    torch.testing.assert_close(a.loss, b.loss, rtol=1e-5, atol=1e-6)
    for name, grad in grads_a.items():
        torch.testing.assert_close(grads_b[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_bf16_d2_tracks_reference():
    """BF16 drafter: D2 stays within bf16 tolerance of the dense reference."""

    def run(arm):
        module = _tiny_dflash2_module(seed=4, num_anchors=12, kd_chunk_size=5, **ARMS[arm])
        module.draft_model.to(torch.bfloat16)
        module.lm_head.to(torch.bfloat16)
        module.embed_tokens.to(torch.bfloat16)
        input_ids, loss_mask, hidden, final_hidden = _padded_batch()
        torch.manual_seed(123)
        metrics = module(
            input_ids=input_ids,
            hidden_states=hidden.to(torch.bfloat16),
            loss_mask=loss_mask,
            teacher_hidden=final_hidden.to(torch.bfloat16),
        )
        (metrics.loss * 0.25).backward()
        return metrics, {n: p.grad.float() for n, p in module.draft_model.named_parameters() if p.grad is not None}

    ref, ref_grads = run("d0")
    out, grads = run("d2")
    torch.testing.assert_close(out.loss.float(), ref.loss.float(), rtol=2e-2, atol=1e-3)
    for name, grad in ref_grads.items():
        scale = grad.abs().max().clamp_min(1e-8)
        assert float((grads[name] - grad).abs().max() / scale) < 5e-2, name
