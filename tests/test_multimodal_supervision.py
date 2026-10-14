"""Multimodal supervision representation: the invariants the trainer must keep.

Schema v2 turns a multi-image / multi-turn conversation into one token sequence
plus two supervision tensors: a ``loss_mask`` marking the supervised assistant
tokens and an ``image_mask`` marking image-token spans (derived in
``collate_features`` from the contract's ``image_token_id``, so offline and
online batches share one definition). This file pins the *consumer-side*
contract of that representation:

* anchor sampling and block supervision never touch ``loss_mask == 0`` tokens;
* a seeded forward is deterministic, and a fully unsupervised padding row does
  not change another sample's loss (batch padding must not corrupt it);
* a text-only batch leaves the visual branch an exact no-op (all image-side
  gradients are exactly zero), while image tokens are demonstrably connected to
  the loss and to the visual gradients;
* the zero-gated pooled adapter is an exact no-op with images present (the
  documented warm-start property);
* an image after the first supervised token is rejected -- the cached visual
  context would leak future visual information into the supervised prefix;
* visual-conditioned runs resume exactly, refuse a changed visual branch, and
  export/reload with the visual branch intact (see the companion export test).

Numeric notes: some assertions compare losses *bitwise* and therefore reseed
the torch RNG immediately before every forward (anchor sampling draws
``torch.rand`` per micro-batch). The image-vs-text comparison runs in float64:
at initialization the fusion contributes ~1e-5 to the draft states, which is
one fp32 ulp of the loss and too fragile to assert on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.dflash.train import run_training_loop
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule

# The visual branch holds one fusion module per draft layer; each owns its
# two-query resampler plus the cross-attention projections and a scalar gate.
_VISUAL_PARAMS_PER_LAYER = 12


def _build_module(
    *,
    visual_gate_init: float = 1.0,
    visual_adapter_type: str = "cross_attention",
    visual_num_query_tokens: int = 2,
    seed: int = 11,
) -> DFlashTrainerModule:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(seed)
    dflash_config = {
        "mask_token_id": 255,
        "target_layer_ids": [0, 1],
        "spatial_rope_enabled": False,
    }
    if visual_num_query_tokens > 0:
        dflash_config.update(
            visual_num_query_tokens=visual_num_query_tokens,
            visual_adapter_type=visual_adapter_type,
            visual_adapter_dim=16,
            visual_num_attention_heads=2,
            visual_gate_init=visual_gate_init,
        )
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
        dflash_config=dflash_config,
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
    )


def _tiny_batch(rows: int = 2, seq_len: int = 48, *, seed: int = 12):
    torch.manual_seed(seed)
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(rows, 1)
    loss_mask = torch.ones(rows, seq_len)
    loss_mask[:, :8] = 0  # prefix positions stay unsupervised
    hidden = torch.randn(rows, seq_len, 128)
    return input_ids, loss_mask, hidden


def _image_span(rows: int, seq_len: int = 48) -> torch.Tensor:
    """Image tokens inside the unsupervised prefix of row 0 only."""
    mask = torch.zeros(rows, seq_len, dtype=torch.bool)
    mask[0, 2:6] = True
    return mask


def _forward(module, input_ids, loss_mask, hidden, image_mask, *, seed: int = 123):
    torch.manual_seed(seed)
    return module(
        input_ids=input_ids,
        hidden_states=hidden,
        loss_mask=loss_mask,
        image_mask=image_mask,
    )


def _to_double(module: DFlashTrainerModule) -> DFlashTrainerModule:
    """Cast the trainer to float64.

    ``lm_head`` / ``embed_tokens`` are non-registered references (see
    ``DFlashTrainerModule.__init__``), so ``module.double()`` does not reach
    them and they must be cast explicitly.
    """
    module.double()
    module.lm_head.double()
    module.embed_tokens.double()
    return module


def _visual_parameters(module: DFlashTrainerModule) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in module.draft_model.named_parameters()
        if "visual_fusion" in name
    }


# --------------------------------------------------------------------------- #
# supervision mapping (exact, no sampling flake)
# --------------------------------------------------------------------------- #


def test_supervision_never_lands_on_unsupervised_positions():
    """Kept anchors and every supervised block slot carry ``loss_mask > 0``."""
    module = _build_module()
    input_ids, loss_mask, _ = _tiny_batch(rows=2)
    loss_mask[0, 20:24] = 0.0  # hole inside the supervised region
    loss_mask[1, 8:30] = 0.0  # long unsupervised stretch for row 1
    seq_len = input_ids.shape[1]

    torch.manual_seed(5)
    anchors, keep = module._sample_anchor_positions(seq_len, loss_mask, torch.device("cpu"))
    label_indices, target_ids, block_mask = module._build_block_targets(
        input_ids, loss_mask, anchors, keep, seq_len
    )

    kept_rows, kept_cols = keep.nonzero(as_tuple=True)
    kept_anchors = anchors[kept_rows, kept_cols]
    assert kept_anchors.numel() > 0
    assert torch.all(loss_mask[kept_rows, kept_anchors] > 0)

    gathered = loss_mask.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, label_indices.clamp(max=seq_len - 1))
    supervised = block_mask > 0
    assert torch.equal(supervised, keep.unsqueeze(-1) & (label_indices < seq_len) & (gathered > 0))
    assert torch.all(gathered[supervised] > 0)
    reference_labels = input_ids.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(
        2, label_indices.clamp(max=seq_len - 1)
    )
    assert torch.equal(target_ids[supervised], reference_labels[supervised])


# --------------------------------------------------------------------------- #
# mask / input invariance
# --------------------------------------------------------------------------- #


def test_seeded_forward_is_deterministic():
    module = _build_module()
    input_ids, loss_mask, hidden = _tiny_batch()
    images = _image_span(2)
    first = _forward(module, input_ids, loss_mask, hidden, images).loss.detach()
    second = _forward(module, input_ids, loss_mask, hidden, images).loss.detach()
    assert torch.equal(first, second)


def test_unsupervised_padding_rows_do_not_change_the_loss():
    """A fully unsupervised extra row is inert: same loss and valid-block count."""
    module = _build_module()
    input_ids, loss_mask, hidden = _tiny_batch(rows=2)
    two_rows = _forward(module, input_ids, loss_mask, hidden, _image_span(2))

    padded_ids = torch.cat([input_ids, input_ids[:1]], dim=0)
    padded_mask = torch.cat([loss_mask, torch.zeros(1, loss_mask.shape[1])], dim=0)
    padded_hidden = torch.cat([hidden, hidden[:1]], dim=0)
    padded = _forward(module, padded_ids, padded_mask, padded_hidden, _image_span(3))

    assert torch.equal(two_rows.loss.detach(), padded.loss.detach())
    assert int(two_rows.valid_blocks) == int(padded.valid_blocks)


# --------------------------------------------------------------------------- #
# visual conditioning: loss / gradient connection
# --------------------------------------------------------------------------- #


def test_image_tokens_drive_the_loss_and_the_visual_gradients():
    module = _to_double(_build_module(visual_gate_init=1.0))
    input_ids, loss_mask, hidden = _tiny_batch()
    loss_mask = loss_mask.double()
    hidden = hidden.double()
    images = _image_span(2)

    with torch.no_grad():
        text_only = _forward(module, input_ids, loss_mask, hidden, None).loss.detach()
        with_images = _forward(module, input_ids, loss_mask, hidden, images).loss.detach()
    assert not torch.equal(text_only, with_images)

    module.zero_grad(set_to_none=True)
    _forward(module, input_ids, loss_mask, hidden, images).loss.backward()
    visual = _visual_parameters(module)
    assert len(visual) == 2 * _VISUAL_PARAMS_PER_LAYER, "the visual branch must own its full parameter set"
    for name, parameter in visual.items():
        assert parameter.grad is not None, name
        assert bool((parameter.grad != 0).any()), name


def test_text_only_batch_leaves_the_visual_branch_an_exact_noop():
    """Without image tokens the cached context is zero: zero grads, same loss."""
    module = _to_double(_build_module(visual_gate_init=1.0))
    input_ids, loss_mask, hidden = _tiny_batch()
    loss_mask = loss_mask.double()
    hidden = hidden.double()
    no_images = torch.zeros_like(loss_mask, dtype=torch.bool)

    with torch.no_grad():
        absent = _forward(module, input_ids, loss_mask, hidden, None).loss.detach()
        all_false = _forward(module, input_ids, loss_mask, hidden, no_images).loss.detach()
    assert torch.equal(absent, all_false)

    module.zero_grad(set_to_none=True)
    _forward(module, input_ids, loss_mask, hidden, no_images).loss.backward()
    for name, parameter in _visual_parameters(module).items():
        assert parameter.grad is not None, name
        assert not bool((parameter.grad != 0).any()), name


def test_pooled_adapter_at_zero_gate_ignores_images_exactly():
    """The pooled-MLP adapter's zero gate is the documented warm-start no-op."""
    module = _build_module(visual_gate_init=0.0, visual_adapter_type="pooled_mlp", visual_num_query_tokens=1)
    input_ids, loss_mask, hidden = _tiny_batch()
    images = _image_span(2)
    with torch.no_grad():
        text_only = _forward(module, input_ids, loss_mask, hidden, None).loss.detach()
        with_images = _forward(module, input_ids, loss_mask, hidden, images).loss.detach()
    assert torch.equal(text_only, with_images)


# --------------------------------------------------------------------------- #
# image-visibility validation
# --------------------------------------------------------------------------- #


def test_visual_trainer_rejects_image_mask_that_leaks_future_turns():
    module = _build_module()
    input_ids, loss_mask, hidden = _tiny_batch()

    late = torch.zeros_like(loss_mask, dtype=torch.bool)
    late[0, 30:34] = True  # after the first supervised token (position 8)
    with pytest.raises(ValueError, match="leak future visual information"):
        _forward(module, input_ids, loss_mask, hidden, late)

    overlapping = torch.zeros_like(loss_mask, dtype=torch.bool)
    overlapping[0, 7:9] = True  # straddles the first supervised token
    with pytest.raises(ValueError, match="leak future visual information"):
        _forward(module, input_ids, loss_mask, hidden, overlapping)

    with pytest.raises(ValueError, match="same \\[batch, sequence\\] shape"):
        _forward(module, input_ids, loss_mask, hidden, torch.zeros(1, input_ids.shape[1], dtype=torch.bool))


# --------------------------------------------------------------------------- #
# resume: the visual supervision state round-trips
# --------------------------------------------------------------------------- #

_RESUME_TRAIN = {
    "train": {
        "micro_batch_size": 2,
        "grad_accumulation_steps": 1,
        "num_epochs": 1,
        "lr": 1e-3,
        "log_every_steps": 0,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.1,
        "min_lr_ratio": 0.1,
        "draft_dtype": "float32",
    }
}
# ``image_token_id=1``: the dataset plants image tokens in the unsupervised
# prefix, so ``collate_features`` derives a real image_mask for every batch.
_RESUME_CONTRACT = {"tokens": {"image_token_id": 1}}
_IMAGE_TOKEN_ID = 1


class _VisualDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 6, seq: int = 48, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            ids = torch.arange(seq) % 200 + 10
            ids[2:6] = _IMAGE_TOKEN_ID
            mask = torch.ones(seq)
            mask[:8] = 0.0
            hidden = torch.randn(seq, 2 * 64, generator=generator) * 0.2
            self.items.append(
                {
                    "input_ids": ids,
                    "loss_mask": mask,
                    "hidden": hidden,
                    "position3": torch.arange(seq).repeat(3, 1),
                    "image_span": (0, 0),
                    "sample_id": "tiny-visual",
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _run(module, *, resume_state=None, sink=None):
    return run_training_loop(
        module,
        _VisualDataset(),
        _RESUME_TRAIN,
        _RESUME_CONTRACT,
        torch.device("cpu"),
        total_steps=6,
        seed=11,
        shuffle=True,
        log_path=None,
        resume_state=resume_state,
        on_checkpoint=sink,
        save_every=2,
    )


def test_visual_run_resumes_exactly_and_refuses_a_changed_branch():
    full_losses = [row["loss"] for row in _run(_build_module())["history"]]

    saved: dict = {}
    interrupted = _run(_build_module(), sink=lambda state: saved.setdefault("state", state))
    assert len(interrupted["history"]) == 6
    assert "state" in saved and int(saved["state"]["step"]) == 2
    assert any("visual_fusion" in key for key in saved["state"]["model"]), "visual weights are part of the resume state"

    resumed = _run(_build_module(), resume_state=saved["state"])
    resumed_losses = [row["loss"] for row in resumed["history"]]
    assert resumed["summary"]["steps"] == 6
    assert resumed["summary"]["resumed_from_step"] == 2
    for index in range(2):
        assert resumed_losses[index] == full_losses[index]
    for first, second in zip(resumed_losses[2:], full_losses[2:]):
        assert first == pytest.approx(second, abs=1e-9)

    with pytest.raises(RuntimeError):
        _run(_build_module(visual_num_query_tokens=3), resume_state=saved["state"])


# --------------------------------------------------------------------------- #
# export: the visual branch and its serving contract round-trip
# --------------------------------------------------------------------------- #


def _visual_parameters_from_state(state: dict) -> dict:
    return {key: value for key, value in state.items() if "visual_fusion" in key}


def _write_visual_run(tmp_path):
    """Minimal run directory holding a visual-conditioned DFlash checkpoint."""
    module = _build_module()
    draft = module.draft_model
    run_dir = tmp_path / "run_visual"
    (run_dir / "train").mkdir(parents=True)
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 100, "max_pixels": 200},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"num_draft_layers": 2, "draft_layer_ids": [0, 1]},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
        "head": {"tie_word_embeddings": True},
    }
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"drafter": {"spatial_rope_mode": "replace", "spatial_rope_enabled": False}}),
        encoding="utf-8",
    )
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": "dflash",
        "draft_config": draft.config.to_dict(),
        "attention_backend": "sdpa",
        "mask_token_id": 255,
        "block_size": 8,
        "num_anchors": 16,
        "loss_decay_gamma": 7.0,
        "target_layer_ids": [0, 1],
        "capture_layer_ids": [0, 1],
        "identity": {"model_fingerprint": "f" * 64},
        "step": 3,
        "model": {key: value.clone() for key, value in draft.state_dict().items()},
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")
    return run_dir, draft


def test_visual_checkpoint_exports_and_reloads_with_the_visual_branch(tmp_path):
    from vlm_dflash.export.hf import export_run, load_export

    run_dir, draft = _write_visual_run(tmp_path)
    export_dir = export_run(run_dir)
    model, exported = load_export(export_dir)

    original_state = draft.state_dict()
    loaded_state = model.state_dict()
    assert set(original_state) == set(loaded_state)
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key]), key
    assert len(_visual_parameters_from_state(loaded_state)) == 2 * _VISUAL_PARAMS_PER_LAYER

    visual = exported["visual_conditioning"]
    assert visual == {"num_query_tokens": 2, "adapter_type": "cross_attention", "adapter_dim": 16}
    card = (export_dir / "model_card.md").read_text(encoding="utf-8")
    assert "visual conditioning" in card
    assert "cached visual vector" in card

    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    assert config["visual_conditioning"] == visual
    draft_config = config["drafter"]["draft_config"]["dflash_config"]
    assert draft_config["visual_num_query_tokens"] == 2
    assert draft_config["visual_adapter_type"] == "cross_attention"
