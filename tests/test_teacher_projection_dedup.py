"""Reuse teacher identity, never anchor-dependent student states."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.objectives import TeacherLogitsStream, lk_term, e2e_tv_term


def _case():
    torch.manual_seed(31)
    head = torch.nn.Linear(12, 37, bias=True).double().requires_grad_(False)
    hidden = torch.randn(2, 24, 12, dtype=torch.float64)
    # Same positions in two samples are distinct identities. Adjacent anchors
    # have overlapping prediction windows within a sample.
    labels = torch.tensor([3, 4, 5, 6]).view(1, 4, 1) + torch.arange(8).view(1, 1, 8)
    return head, hidden, labels.expand(2, -1, -1).clone()


@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("compact", [False, True])
def test_reuse_preserves_rows_and_projects_unique_sample_positions(first, compact):
    head, hidden, labels = _case()
    sample_index = None
    if compact:
        sample_index = torch.tensor([1, 0, 1])
        labels = labels[sample_index, torch.tensor([0, 1, 2])].unsqueeze(1)
    calls, counts = [], {}
    head.register_forward_pre_hook(lambda _m, args: calls.append(len(args[0])))

    def count(**values):
        for key, value in values.items():
            counts[key] = counts.get(key, 0) + value

    baseline = TeacherLogitsStream(head, hidden, labels, first_slot=first, sample_index=sample_index)
    optimized = TeacherLogitsStream(
        head, hidden, labels, first_slot=first, sample_index=sample_index,
        projection_dedup=True, counter=count,
    )
    expected = baseline.full()
    actual = optimized.full()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert calls[-1] < calls[-2]
    assert counts["teacher_head_rows"] == calls[-1]
    assert counts["teacher_requested_rows"] == optimized.num_rows
    assert counts["teacher_reused_rows"] == optimized.num_rows - calls[-1]
    assert not actual.requires_grad


def test_chunk_replay_is_bounded_and_does_not_cache_stale_teacher_values():
    head, hidden, labels = _case()
    stream = TeacherLogitsStream(head, hidden, labels, first_slot=1, projection_dedup=True)
    reference = TeacherLogitsStream(head, hidden, labels, first_slot=1)
    for start, stop in [(0, 13), (13, 30), (30, 56), (-2, 2), (56, 80), (9, 5)]:
        torch.testing.assert_close(stream.logits(start, stop), reference.logits(start, stop))
    old = stream.logits(0, 14).clone()
    hidden.add_(1)
    new = stream.logits(0, 14)
    assert not torch.equal(old, new)
    torch.testing.assert_close(new, reference.logits(0, 14))


@pytest.mark.parametrize("kind", ["kl", "hybrid", "tv", "e2e"])
def test_loss_and_student_gradient_parity_including_checkpoint_replay(kind):
    head, hidden, labels = _case()
    reference = TeacherLogitsStream(head, hidden, labels, first_slot=1)
    optimized = TeacherLogitsStream(head, hidden, labels, first_slot=1, projection_dedup=True)
    student = torch.randn(2, 4, 7, 37, requires_grad=True)
    mask = torch.ones(2, 4, 7)
    mask[1, 0, 3:] = 0

    def run(stream):
        if kind == "e2e":
            loss, _ = e2e_tv_term(student, valid_mask_pb=mask, teacher_chunk_fn=stream.logits)
        else:
            loss = lk_term(student.flatten(0, 2), weights=mask, loss_type=kind,
                           eta=1.0, teacher_chunk_fn=stream.logits)
        gradient = torch.autograd.grad(loss, student)[0]
        return loss.detach(), gradient

    expected, actual = run(reference), run(optimized)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("head", [torch.nn.Linear(12, 37), torch.nn.Sequential(torch.nn.Linear(12, 37))])
def test_reuse_rejects_trainable_or_non_linear_heads(head):
    _, hidden, labels = _case()
    with pytest.raises(ValueError, match="frozen.*Linear"):
        TeacherLogitsStream(head, hidden, labels, first_slot=1, projection_dedup=True)


def test_no_overlap_skips_gather_and_counts_actual_work():
    head, hidden, _ = _case()
    labels = torch.arange(1, 9).view(1, 1, 8)
    counts = {}
    stream = TeacherLogitsStream(head, hidden, labels, first_slot=1, projection_dedup=True,
                                counter=lambda **kw: counts.update(kw))
    actual = stream.full()
    torch.testing.assert_close(actual, head(hidden[0, 1:8]))
    assert counts["teacher_reused_rows"] == 0
    assert counts["teacher_head_rows"] == 7


@pytest.mark.parametrize("algorithm,stage", [("dflash", 0), ("dflash", 1), ("dfly", 0), ("dfly", 1), ("dspark", 0)])
def test_full_trainer_loss_every_gradient_and_counters(algorithm, stage):
    from vlm_dflash.objectives import resolve_recipe

    if algorithm == "dfly":
        from tests.test_dfly import _tiny_module, _tiny_batch
        original, _ = _tiny_module(seed=8)
    elif algorithm == "dspark":
        from tests.test_dspark import _tiny_dspark_module, _tiny_batch
        original = _tiny_dspark_module(seed=8)
    else:
        from tests.test_trainer_smoke import _tiny_module
        from tests.test_dspark import _tiny_batch
        original = _tiny_module(seed=8)
    original.lm_head.requires_grad_(False)
    recipe = resolve_recipe({"algorithm": {"train_recipe":
        "upstream_dspark" if algorithm == "dspark" else "upstream_two_stage"}}, algorithm)
    module = type(original)(
        original.draft_model, original.lm_head, original.embed_tokens, 255,
        block_size=8, attention_backend="sdpa", num_anchors=16,
        train_recipe=recipe, teacher_projection_dedup=True,
    )
    module.set_training_progress(stage, 2)
    ids, mask, hidden = _tiny_batch(seed=9)
    mask[1, 20:] = 0
    teacher = torch.randn(2, 48, 64)

    def run(enabled):
        module.teacher_projection_dedup = enabled
        module.zero_grad(set_to_none=True)
        torch.manual_seed(52)
        out = module(input_ids=ids, hidden_states=hidden, loss_mask=mask, teacher_hidden=teacher)
        (out.loss / 3).backward()
        return out, {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}, dict(module.last_workload)

    ref, ref_grad, ref_counts = run(False)
    out, grad, counts = run(True)
    for field in ("loss", "loss_weight", "valid_tokens", "correct_tokens", "pos_count", "pos_correct", "accept_len_sum"):
        torch.testing.assert_close(getattr(out, field), getattr(ref, field), rtol=1e-6, atol=1e-7)
    assert set(grad) == set(ref_grad)
    for name, value in ref_grad.items():
        torch.testing.assert_close(grad[name], value, rtol=3e-5, atol=2e-7, msg=name)
    assert counts["teacher_head_rows"] < ref_counts["teacher_head_rows"]
    assert counts["teacher_requested_rows"] == ref_counts["teacher_requested_rows"]
    assert module.execution_switches()["teacher_projection_dedup"]


def test_factory_forwards_switch_and_rejects_noop_objectives():
    from tests.test_trainer_smoke import _tiny_module
    from vlm_dflash.dflash.factory import build_trainer_module
    from vlm_dflash.objectives import resolve_recipe

    original = _tiny_module()
    cfg = {"drafter": {"mask_token_id": 255, "block_size": 8, "attention_backend": "sdpa",
                       "teacher_projection_dedup": True}}
    with pytest.raises(ValueError, match="requires a shared training recipe"):
        build_trainer_module(original.draft_model.config, {}, cfg, original.embed_tokens.weight.detach())
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dflash")
    module = build_trainer_module(original.draft_model.config, {}, cfg, original.embed_tokens.weight.detach(),
                                 module_kwargs={"train_recipe": recipe})
    assert module.teacher_projection_dedup


@pytest.mark.parametrize("algorithm", ["dflash", "dfly", "dflash2", "jetspec", "domino", "dspark"])
def test_config_rejects_dedup_when_no_shared_teacher_recipe(algorithm):
    from vlm_dflash.config import validate_runtime_config

    with pytest.raises(ValueError, match="teacher_projection_dedup"):
        validate_runtime_config({"algorithm": algorithm, "drafter": {"teacher_projection_dedup": True}})
