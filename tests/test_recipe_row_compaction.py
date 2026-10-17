"""Recipe-run row compaction + the counts-only scoring path (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.decay_loss import DFlashDecayLoss
from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule
from vlm_dflash.dfly.trainer import DFlyTrainerModule
from vlm_dflash.objectives import TeacherLogitsStream, resolve_recipe


class _CountingHead(torch.nn.Module):
    """Records the row count of every projection routed through it."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.row_counts: list[int] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.row_counts.append(int(hidden.reshape(-1, hidden.size(-1)).shape[0]))
        return self.inner(hidden)


def _tiny_module(
    seed: int = 11,
    *,
    module_class=DFlyTrainerModule,
    recipe: bool = True,
    stage: int = 1,
    recipe_row_compaction=None,
    recipe_compile_tv_kl: bool = False,
):
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
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    resolved = (
        resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dfly") if recipe else None
    )
    module = module_class(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
        train_recipe=resolved,
        recipe_row_compaction=recipe_row_compaction,
        recipe_compile_tv_kl=recipe_compile_tv_kl,
    )
    if resolved is not None:
        module.set_training_progress(stage, 2)
    return module


def _batch():
    torch.manual_seed(12)
    batch_size, seq_len = 3, 48
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    # One long span, two short ones: N is driven by the long sample, so the
    # short samples carry padding blocks (the case compaction exists for).
    loss_mask[1, 16:] = 0
    loss_mask[2, 12:] = 0
    hidden = torch.randn(batch_size, seq_len, 128)
    teacher_hidden = torch.randn(batch_size, seq_len, 64)
    return input_ids, loss_mask, hidden, teacher_hidden


def _run(module: torch.nn.Module, batch) -> dict:
    input_ids, loss_mask, hidden, teacher_hidden = batch
    head = _CountingHead(module.lm_head)
    object.__setattr__(module, "lm_head", head)
    module.zero_grad(set_to_none=True)
    metrics = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=teacher_hidden
    )
    metrics.loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in module.draft_model.named_parameters()
        if param.grad is not None
    }
    extras = {
        key: float(value)
        for key, value in (metrics.extra_metrics or {}).items()
        if isinstance(value, (int, float))
    }
    return {
        "loss": float(metrics.loss.detach()),
        "valid_tokens": float(metrics.valid_tokens),
        "correct_tokens": float(metrics.correct_tokens),
        "loss_weight": float(metrics.loss_weight),
        "accuracy": float(metrics.accuracy),
        "accept_len": float(metrics.accept_len),
        "pos_correct": metrics.pos_correct.clone(),
        "pos_count": metrics.pos_count.clone(),
        "extra": extras,
        "grads": grads,
        "head_rows": list(head.row_counts),
    }


@pytest.mark.parametrize("stage", [0, 1])
def test_row_compaction_is_exact_on_cpu(stage):
    batch = _batch()
    dense = _run(_tiny_module(stage=stage, recipe_row_compaction=False), batch)
    compact = _run(_tiny_module(stage=stage, recipe_row_compaction=True), batch)
    # The compaction must actually engage: fewer rows hit the frozen head.
    assert compact["head_rows"][0] < dense["head_rows"][0]
    assert compact["loss"] == pytest.approx(dense["loss"], rel=1e-9, abs=1e-12)
    assert compact["valid_tokens"] == dense["valid_tokens"]
    assert compact["loss_weight"] == dense["loss_weight"]
    assert compact["accuracy"] == dense["accuracy"]
    assert compact["accept_len"] == dense["accept_len"]
    assert torch.equal(compact["pos_correct"], dense["pos_correct"])
    assert torch.equal(compact["pos_count"], dense["pos_count"])
    assert set(compact["extra"]) == set(dense["extra"])
    for key, value in dense["extra"].items():
        assert compact["extra"][key] == pytest.approx(value, rel=1e-5, abs=1e-8), key
    assert set(compact["grads"]) == set(dense["grads"])
    for name, grad in dense["grads"].items():
        assert torch.equal(compact["grads"][name], grad), name


def test_auto_enable_and_explicit_reject():
    auto = _tiny_module(recipe_row_compaction=None)
    assert auto.recipe_row_compaction is True
    off = _tiny_module(recipe_row_compaction=False)
    assert off.recipe_row_compaction is False
    with pytest.raises(ValueError, match="not supported"):
        _tiny_module(module_class=DFlashTrainerModule, recipe_row_compaction=True)
    base = _tiny_module(module_class=DFlashTrainerModule, recipe_row_compaction=None)
    assert base.recipe_row_compaction is False


def test_no_recipe_keeps_dense_scoring():
    batch = _batch()
    on = _run(_tiny_module(recipe=False, recipe_row_compaction=True), batch)
    off = _run(_tiny_module(recipe=False, recipe_row_compaction=False), batch)
    assert on["head_rows"] == off["head_rows"]
    assert on["loss"] == off["loss"]


def test_accuracy_counts_match_forward():
    loss_fn = DFlashDecayLoss(loss_gamma=7.0, normalize="mean", first_slot=1)
    torch.manual_seed(4)
    logits = torch.randn(6, 7, 256)
    targets = torch.randint(0, 256, (6, 7))
    mask = torch.ones(6, 7)
    mask[0, 3:] = 0
    out = loss_fn(logits, targets, mask, num_tokens=None, block_size=8)
    correct, count = loss_fn.accuracy_counts(logits, targets, mask, block_size=8)
    assert torch.equal(correct, out.draft_correct_per_pos)
    assert torch.equal(count, out.draft_count_per_pos)
    assert loss_fn.total_loss_dtype(logits) == out.total_loss.dtype


def test_teacher_stream_sample_index_matches_dense():
    torch.manual_seed(5)
    batch, n, block_size, hidden_size, vocab = 3, 4, 8, 16, 32
    hidden = torch.randn(batch, 20, hidden_size)
    labels = torch.randint(1, 19, (batch, n, block_size))
    head = torch.nn.Linear(hidden_size, vocab, bias=False)
    dense = TeacherLogitsStream(head, hidden, labels, first_slot=1)
    selection = torch.tensor([[0, 1], [2, 3]])
    compact_labels = labels[selection[:, 0], selection[:, 1]].unsqueeze(1)
    compact = TeacherLogitsStream(
        head, hidden, compact_labels, first_slot=1, sample_index=selection[:, 0]
    )
    slots = block_size - 1
    dense_rows = dense.logits(0, dense.num_rows)
    expected = torch.cat(
        [
            dense_rows[(int(b) * n + int(k)) * slots : (int(b) * n + int(k) + 1) * slots]
            for b, k in selection.tolist()
        ]
    )
    # The compacted call feeds the frozen head a different number of rows, so
    # BLAS picks a different blocking/association order: rows match to fp32
    # rounding (~1 ULP), not bitwise. Ordering, sample mapping and positions
    # are what this test pins; the bitwise-gradient test above covers the math.
    assert torch.allclose(compact.logits(0, compact.num_rows), expected, rtol=1e-5, atol=1e-6)


def test_teacher_stream_sample_index_validation():
    head = torch.nn.Linear(16, 32, bias=False)
    hidden = torch.randn(3, 20, 16)
    labels = torch.randint(1, 19, (2, 1, 8))
    with pytest.raises(ValueError, match="one sample per block"):
        TeacherLogitsStream(head, hidden, labels, first_slot=1, sample_index=torch.tensor([0]))
    with pytest.raises(ValueError, match=r"in \[0, 3\)"):
        TeacherLogitsStream(head, hidden, labels, first_slot=1, sample_index=torch.tensor([0, 3]))

def test_recipe_valid_tokens_match_dense_counts():
    """Regression: a recipe run must report the *count* sums as valid tokens.

    ``accuracy_counts`` returns ``(correct, count)``; the trainer call site once
    unpacked it as ``(count, correct)``, which silently swapped
    ``valid_tokens``/``correct_tokens`` -- and with them the logged accuracy and
    accept metrics -- whenever a recipe was active. The compact-vs-dense test
    could not see it because both sides took the same swapped branch. This pins
    the recipe path against the non-recipe path (an independent reduction over
    the same mask) and against the per-offset tensors it returns.
    """
    batch = _batch()
    on = _run(_tiny_module(recipe=True, recipe_row_compaction=False), batch)
    off = _run(_tiny_module(recipe=False, recipe_row_compaction=False), batch)
    assert on["valid_tokens"] == off["valid_tokens"]
    assert on["correct_tokens"] == off["correct_tokens"]
    assert on["accuracy"] == off["accuracy"]
    assert torch.equal(on["pos_count"], off["pos_count"])
    assert torch.equal(on["pos_correct"], off["pos_correct"])
    # The scalar sums must agree with the per-offset tensors they came from.
    assert on["valid_tokens"] == float(on["pos_count"].sum())
    assert on["correct_tokens"] == float(on["pos_correct"].sum())
    # A random-init model is nowhere near perfect, so the swap is unmissable.
    assert on["valid_tokens"] > on["correct_tokens"]
