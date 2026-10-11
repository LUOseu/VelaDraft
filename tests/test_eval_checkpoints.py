"""Regression tests for the eval checkpoint selection / gate wiring.

A previous edit left the checkpoint loop body unreachable, so the eval wrote an
empty ``runs`` mapping and reported ``token_exact: True`` vacuously. These tests
pin the guards that make that impossible to miss: an empty checkpoint list is
rejected up front, and a result that does not cover every requested checkpoint
is never written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.eval.run import assert_all_checkpoints_evaluated, checkpoint_names_from_config


def test_checkpoint_names_rejects_empty_list():
    with pytest.raises(ValueError, match="checkpoints is empty"):
        checkpoint_names_from_config({"checkpoints": []})


def test_checkpoint_names_defaults_and_coerces():
    assert checkpoint_names_from_config({}) == ["ckpt_final.pt", "ckpt_init.pt"]
    assert checkpoint_names_from_config({"checkpoints": ["ckpt_final.pt"]}) == ["ckpt_final.pt"]


def test_partial_eval_is_rejected():
    names = ["ckpt_final.pt", "ckpt_init.pt"]
    with pytest.raises(RuntimeError, match="refusing to write a partial eval result"):
        assert_all_checkpoints_evaluated({"ckpt_final.pt": {"token_exact": True}}, names)
    with pytest.raises(RuntimeError, match="refusing to write a partial eval result"):
        assert_all_checkpoints_evaluated({}, names)
    assert_all_checkpoints_evaluated({name: {"token_exact": True} for name in names}, names)
