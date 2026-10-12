"""JetSpec unit tests: KD math, causal mask, teacher alignment, registry plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.config import load_config
from vlm_dflash.dflash.features import FeatureDataset, collate_features
from vlm_dflash.dflash.mask import build_dflash_mask_mod
from vlm_dflash.dflash.trainer_core import NoValidAnchorsError
from vlm_dflash.jetspec.kd_loss import ForwardKDLoss, forward_kl_per_token
from vlm_dflash.jetspec.trainer import JetSpecTrainerModule
from vlm_dflash.utils import write_json

VOCAB = 256
HIDDEN = 64
BLOCK_SIZE = 8


def _tiny_jetspec_module(seed: int = 0, *, block_size: int = BLOCK_SIZE, num_anchors: int = 16, **kwargs):
    torch.manual_seed(seed)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

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
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    lm_head.weight = embed.weight
    module_kwargs = {"kd_temperature": 1.0, "kd_chunk_size": 2048}
    module_kwargs.update(kwargs)
    decay = module_kwargs.pop("loss_decay_gamma", None)
    return JetSpecTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=decay,
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


# --------------------------------------------------------------------------- #
# KD loss math (ported semantics)
# --------------------------------------------------------------------------- #

def test_forward_kl_per_token_matches_reference():
    torch.manual_seed(0)
    teacher = torch.randn(7, VOCAB, dtype=torch.float64)
    student = torch.randn(7, VOCAB, dtype=torch.float64)
    log_p = torch.log_softmax(teacher, dim=-1)
    log_q = torch.log_softmax(student, dim=-1)
    expected = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    actual = forward_kl_per_token(log_p, log_q)
    assert torch.allclose(actual, expected, atol=1e-12)


def test_kd_loss_temperature_scaling_is_t_squared():
    torch.manual_seed(1)
    teacher = torch.randn(5, VOCAB)
    student = torch.randn(5, VOCAB)
    labels = torch.zeros(5, dtype=torch.long)
    plain = ForwardKDLoss(temperature=1.0)(student, teacher, labels)
    scaled = ForwardKDLoss(temperature=2.0)(student, teacher, labels)
    reference = ForwardKDLoss(temperature=1.0)(student / 2.0, teacher / 2.0, labels) * 4.0
    assert scaled.item() == pytest.approx(reference.item(), rel=1e-6)
    assert not plain.item() == pytest.approx(scaled.item(), rel=1e-3)


def test_kd_loss_ignore_index_and_mean():
    torch.manual_seed(2)
    teacher = torch.randn(6, VOCAB)
    student = torch.randn(6, VOCAB)
    labels = torch.tensor([0, -100, 0, -100, -100, 0])
    loss = ForwardKDLoss()(student, teacher, labels)
    per_token = forward_kl_per_token(
        torch.log_softmax(teacher.float(), dim=-1), torch.log_softmax(student.float(), dim=-1)
    )
    assert loss.item() == pytest.approx(per_token[[0, 2, 5]].mean().item(), rel=1e-6)
    # All-ignored batch -> zero, no NaN.
    zeros = ForwardKDLoss()(student, teacher, torch.full((6,), -100))
    assert zeros.item() == 0.0


def test_kd_loss_chunked_matches_full():
    torch.manual_seed(3)
    teacher = torch.randn(11, VOCAB)
    student = torch.randn(11, VOCAB)
    labels = torch.zeros(11, dtype=torch.long)
    labels[4] = -100
    full = ForwardKDLoss(chunk_size=0)(student, teacher, labels)
    chunked = ForwardKDLoss(chunk_size=3)(student, teacher, labels)
    assert chunked.item() == pytest.approx(full.item(), rel=1e-6)


# --------------------------------------------------------------------------- #
# causal in-block mask
# --------------------------------------------------------------------------- #

def _materialize_mask_mod(anchors, keep, ctx_len, block_size, causal):
    mask_mod = build_dflash_mask_mod(anchors, keep, ctx_len, block_size, anchors.shape[1], causal=causal)
    q_len = anchors.shape[1] * block_size
    q_idx = torch.arange(q_len).view(-1, 1)
    kv_idx = torch.arange(ctx_len + q_len).view(1, -1)
    return mask_mod(torch.tensor(0), torch.tensor(0), q_idx, kv_idx)


def test_causal_mask_forbids_future_in_block_attention():
    anchors = torch.tensor([[4, 20]])
    keep = torch.tensor([[True, True]])
    ctx_len, block_size = 8, 4
    causal = _materialize_mask_mod(anchors, keep, ctx_len, block_size, causal=True)
    bidirectional = _materialize_mask_mod(anchors, keep, ctx_len, block_size, causal=False)
    # Query offset i of block 0 (row i) may only see in-block offsets j <= i.
    for i in range(block_size):
        row = causal[i]
        for j in range(block_size):
            visible = bool(row[ctx_len + j].item())
            assert visible == (j <= i), f"causal row {i} offset {j}"
            # The bidirectional DFlash mask sees the whole block.
            assert bool(bidirectional[i][ctx_len + j].item())
    # Context prefix stays fully visible (strictly before the anchor).
    for i in range(block_size):
        assert bool(causal[i][: anchors[0, 0]].all().item())
        assert not bool(causal[i][anchors[0, 0]])


# --------------------------------------------------------------------------- #
# teacher alignment / fail-loud contracts
# --------------------------------------------------------------------------- #

def test_gather_teacher_hidden_alignment():
    """Offset k (k>=1) predicting anchor+k uses the teacher hidden at anchor+k-1."""
    module = _tiny_jetspec_module()
    bsz, seq_len = 1, 20
    input_ids = torch.randint(0, VOCAB - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    anchors = torch.tensor([[2, 10]])
    keep = torch.tensor([[True, True]])
    label_indices, _, _ = module._build_block_targets(input_ids, loss_mask, anchors, keep, seq_len)
    # Channel 0 of the teacher hidden encodes the absolute position.
    teacher_hidden = torch.zeros(bsz, seq_len, HIDDEN)
    teacher_hidden[0, :, 0] = torch.arange(seq_len, dtype=torch.float32)
    gathered = module._gather_teacher_hidden(teacher_hidden, label_indices, seq_len)
    n = anchors.size(1)
    positions = gathered[..., 0].reshape(bsz, n, BLOCK_SIZE - 1)
    assert torch.equal(positions, label_indices[:, :, :-1].float())
    assert positions.tolist() == [[[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]]


def test_jet_requires_capture_v2_teacher_hidden():
    module = _tiny_jetspec_module()
    input_ids, loss_mask, hidden, _ = _tiny_batch()
    with pytest.raises(ValueError, match="capture-v2"):
        module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)


def test_jet_rejects_shifted_slot_convention_and_decay():
    with pytest.raises(ValueError, match="sample_from_anchor"):
        _tiny_jetspec_module(sample_from_anchor=True)
    with pytest.raises(ValueError, match="loss_decay_gamma"):
        _tiny_jetspec_module(loss_decay_gamma=7.0)


def test_no_supervised_positions_raise():
    module = _tiny_jetspec_module()
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=1, seq_len=24)
    loss_mask = torch.zeros_like(loss_mask)
    loss_mask[0, 8:16] = 1.0  # block from anchor 8 covers 8..15, but...
    loss_mask[0, 8] = 1.0
    loss_mask[0, 9:] = 0.0  # ...no supervised continuation anywhere
    with pytest.raises((NoValidAnchorsError, ValueError)):
        module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden)


# --------------------------------------------------------------------------- #
# trainer forward: causal mask binding + KD consistency + learning
# --------------------------------------------------------------------------- #

def test_trainer_forward_uses_causal_mask(monkeypatch):
    from vlm_dflash.dflash import trainer_core

    seen = {}
    real = trainer_core.create_dflash_sdpa_mask

    def _spy(*args, **kwargs):
        seen["causal"] = kwargs.get("causal")
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer_core, "create_dflash_sdpa_mask", _spy)
    module = _tiny_jetspec_module()
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch()
    metrics = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden
    )
    assert seen["causal"] is True
    assert torch.isfinite(metrics.loss)


def test_chunked_trainer_loss_matches_reference_kd():
    module = _tiny_jetspec_module(kd_chunk_size=3)
    torch.manual_seed(4)
    bsz, n, slots = 2, 5, BLOCK_SIZE - 1
    student = torch.randn(bsz, n * slots, VOCAB)
    teacher_hidden = torch.randn(bsz, n * slots, HIDDEN)
    valid = torch.ones(bsz, n * slots)
    valid[0, 3] = 0.0
    loss, stats = module._forward_kl_loss(student, teacher_hidden, valid)
    # Reference: unbounded projection + the ported logits-in KD loss.
    with torch.no_grad():
        teacher_logits = module.lm_head(teacher_hidden.reshape(-1, HIDDEN))
    labels = torch.where(valid.reshape(-1) > 0, torch.zeros(bsz * n * slots, dtype=torch.long), -100)
    reference = module.kd_loss_fn(student.reshape(-1, VOCAB).detach(), teacher_logits, labels)
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5)
    # The teacher argmax stats match a full-argmax computation on valid positions
    # (invalid positions are zero-filled and never consumed).
    flat_valid = valid.reshape(-1).bool()
    assert torch.equal(stats["teacher_ids"][flat_valid], teacher_logits.argmax(dim=-1)[flat_valid])
    assert torch.equal(stats["teacher_ids"][~flat_valid], torch.zeros_like(stats["teacher_ids"][~flat_valid]))


def test_metric_accuracy_is_draft_vs_teacher_agreement():
    module = _tiny_jetspec_module()
    input_ids, loss_mask, hidden, final_hidden = _tiny_batch(batch_size=2, seq_len=48, seed=7)
    metrics = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=final_hidden
    )
    # A fully random tiny draft rarely agrees: agreement must be a real [0,1]
    # number, not the degenerate all-ones/all-zeros.
    assert 0.0 <= float(metrics.accuracy) <= 1.0
    assert float(metrics.accept_len) >= 1.0
    assert float(metrics.valid_tokens) > 0


def test_overfit_tiny_batch_reduces_kd_loss():
    module = _tiny_jetspec_module(seed=3)
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
    assert losses[-1] < 0.5 * losses[0], f"KD loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


# --------------------------------------------------------------------------- #
# feature-store plumbing
# --------------------------------------------------------------------------- #

def test_collate_carries_final_hidden_when_present(tmp_path):
    shard = {
        "samples": [
            {
                "sample_id": "a",
                "input_ids": torch.arange(10, dtype=torch.int32),
                "loss_mask": torch.ones(10, dtype=torch.bool),
                "hidden": torch.randn(10, 2 * HIDDEN),
                "position3": torch.zeros(3, 10, dtype=torch.int32),
                "image_span": (1, 2),
                "final_hidden": torch.randn(10, HIDDEN),
            },
            {
                "sample_id": "b",
                "input_ids": torch.arange(6, dtype=torch.int32),
                "loss_mask": torch.ones(6, dtype=torch.bool),
                "hidden": torch.randn(6, 2 * HIDDEN),
                "position3": torch.zeros(3, 6, dtype=torch.int32),
                "image_span": (1, 2),
                "final_hidden": torch.randn(6, HIDDEN),
            },
        ]
    }
    from vlm_dflash.utils import write_jsonl

    torch.save(shard, tmp_path / "shard_000.pt")
    write_jsonl(
        tmp_path / "index.jsonl",
        [
            {"sample_id": "a", "shard": "shard_000.pt", "shard_index": 0, "seq_len": 10},
            {"sample_id": "b", "shard": "shard_000.pt", "shard_index": 1, "seq_len": 6},
        ],
    )
    write_json(tmp_path / "manifest.json", {"format": "vlm-dflash-features-v1", "stores_final_hidden": True})
    dataset = FeatureDataset(tmp_path)
    batch = collate_features([dataset[0], dataset[1]], image_token_id=999)
    assert batch["final_hidden"].shape == (2, 10, HIDDEN)
    assert torch.equal(batch["final_hidden"][1, 6:], torch.zeros(4, HIDDEN))
    # A store without final_hidden must not invent the key (fresh dataset: the
    # store memoizes the loaded shard).
    del shard["samples"][0]["final_hidden"]
    del shard["samples"][1]["final_hidden"]
    torch.save(shard, tmp_path / "shard_000.pt")
    dataset = FeatureDataset(tmp_path)
    batch = collate_features([dataset[0], dataset[1]], image_token_id=999)
    assert "final_hidden" not in batch


# --------------------------------------------------------------------------- #
# registry / config plumbing
# --------------------------------------------------------------------------- #

def test_jetspec_registered_and_validates_config():
    algorithm = get_algorithm("jetspec")
    assert algorithm.name == "jetspec"
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "jetspec" / "smoke_coco.yaml")
    algorithm.validate_config(cfg)  # must accept the shipped config
    # store_final_hidden back to false -> fail loud (teacher signal missing).
    cfg_bad = load_config(Path(__file__).resolve().parents[1] / "configs" / "jetspec" / "smoke_coco.yaml")
    cfg_bad["capture"]["store_final_hidden"] = False
    with pytest.raises(ValueError, match="store_final_hidden"):
        algorithm.validate_config(cfg_bad)


def test_jetspec_checkpoint_metadata():
    algorithm = get_algorithm("jetspec")
    module = _tiny_jetspec_module()
    contract = {"layers": {"draft_layer_ids": [0, 1]}}
    cfg = {
        "drafter": {"mask_token_id": 255, "block_size": 8, "loss_decay_gamma": None},
        "algorithm": {"name": "jetspec", "jetspec": {"kd_temperature": 1.5, "kd_chunk_size": 512}},
    }
    meta = algorithm.checkpoint_metadata(contract, cfg, module.draft_model.config)
    assert meta["jetspec"] == {"kd_temperature": 1.5, "kd_chunk_size": 512, "causal_in_block": True}
    assert meta["loss_decay_gamma"] is None
    decoder = algorithm.build_inference_module(module.draft_model.config.to_dict(), "sdpa")
    assert decoder.mask_token_id == 255
