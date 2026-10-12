"""Exact-resume test: interrupted + resumed training reproduces the uninterrupted run.

Runs the real training loop on a tiny CPU draft with float32 (``train.draft_dtype``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule
from vlm_dflash.dflash.train import run_training_loop

_CFG = {
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
_CONTRACT = {"tokens": {"image_token_id": 999999}}


def _tiny_module(seed: int = 0) -> DFlashTrainerModule:
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
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=8,
        loss_decay_gamma=7.0,
    )


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 6, seq: int = 48, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n):
            ids = torch.arange(seq) % 200 + 10
            mask = torch.ones(seq)
            mask[:8] = 0.0
            hidden = torch.randn(seq, 2 * 64, generator=generator) * 0.2
            position3 = torch.arange(seq).repeat(3, 1)
            self.items.append(
                {
                    "input_ids": ids,
                    "loss_mask": mask,
                    "hidden": hidden,
                    "position3": position3,
                    "image_span": (0, 0),
                    "sample_id": "tiny",
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _run(module, *, resume_state=None, sink=None):
    return run_training_loop(
        module,
        _TinyDataset(),
        _CFG,
        _CONTRACT,
        torch.device("cpu"),
        total_steps=6,
        seed=11,
        shuffle=True,
        log_path=None,
        resume_state=resume_state,
        on_checkpoint=sink,
        save_every=2,
    )


def test_resume_reproduces_uninterrupted_run():
    uninterrupted = _run(_tiny_module())
    full_losses = [row["loss"] for row in uninterrupted["history"]]

    saved: dict = {}

    def sink(state: dict) -> None:
        saved.setdefault("state", state)

    interrupted = _run(_tiny_module(), sink=sink)
    assert len([row for row in interrupted["history"]]) == 6
    assert "state" in saved
    assert int(saved["state"]["step"]) == 2

    resumed = _run(_tiny_module(), resume_state=saved["state"])
    resumed_losses = [row["loss"] for row in resumed["history"]]
    assert resumed["summary"]["steps"] == 6
    assert resumed["summary"]["resumed_from_step"] == 2
    assert len(resumed_losses) == 6
    for index in range(2):
        assert resumed_losses[index] == full_losses[index]
    for first, second in zip(resumed_losses[2:], full_losses[2:]):
        assert first == pytest.approx(second, abs=1e-9)


def test_resume_state_carries_optimizer_and_rng():
    saved: dict = {}
    _run(_tiny_module(), sink=lambda state: saved.setdefault("state", state))
    state = saved["state"]
    assert state["format"] == "vlm-dflash-train-state-v1"
    assert state["steps_total_planned"] == 6
    assert "torch_rng_state" in state
    assert state["optimizer"]["state"], "optimizer state must be present for exact resume"


def test_gradient_coverage_is_measured_while_grads_are_populated():
    """GOAL 13 audit must read the live .grad fields, not the cleared ones.

    The fp32-master sync clears every ``.grad`` before the optimizer step, so
    auditing after it reported 0/N trainable tensors with grad — a result that
    reads like "nothing trains" while the run is fine.
    """
    output = _run(_tiny_module())
    coverage = output["gradient_coverage"]
    assert coverage is not None
    assert coverage["params_with_grad"] > 0
    assert coverage["params_with_grad"] == coverage["trainable_tensors"]
    assert coverage["fraction_with_grad"] > 0.9  # element-weighted, not 22/74k
    groups = {group["group"]: group for group in coverage["groups"]}
    assert groups["draft_model.layers"]["params_with_nonzero_grad"] > 0
    assert groups["draft_model.layers"]["trainable_tensors"] > 0
