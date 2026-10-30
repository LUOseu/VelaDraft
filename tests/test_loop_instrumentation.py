"""Training-loop measurement and control: stop_at_step, forks, timing windows, validation hook."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_resume import _CONTRACT, _tiny_module, _TinyDataset
from vlm_dflash.dflash.train import run_training_loop


def _cfg(**train):
    cfg = {
        "train": {
            "micro_batch_size": 2,
            "grad_accumulation_steps": 2,
            "num_epochs": 1,
            "lr": 1e-3,
            "log_every_steps": 0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "draft_dtype": "float32",
        }
    }
    cfg["train"].update(train)
    return cfg


def _run(cfg, *, resume_state=None, sink=None, log_path=None, hook=None, save_every=2):
    return run_training_loop(
        _tiny_module(),
        _TinyDataset(n=24),
        cfg,
        _CONTRACT,
        torch.device("cpu"),
        total_steps=6,
        seed=11,
        shuffle=True,
        log_path=log_path,
        resume_state=resume_state,
        on_checkpoint=sink,
        save_every=save_every,
        validation_hook=hook,
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_stop_at_step_writes_state_and_measurement_fields(tmp_path):
    saved = []
    log_path = tmp_path / "train_log.jsonl"
    out = _run(_cfg(stop_at_step=3), sink=saved.append, log_path=log_path)
    assert out["summary"]["completed_step"] == 3
    assert out["summary"]["stopped_at_step"] == 3
    assert saved[-1]["step"] == 3 and saved[-1]["consumed_samples"] == 3 * 4
    rows = _rows(log_path)
    assert [row["step"] for row in rows] == [1, 2, 3]
    for row in rows:
        assert row["update_s"] > 0 and row["samples"] == 4
        assert row["w_microbatches"] == 2
        assert row["w_allocated_rows"] >= row["w_kept_rows"] >= row["w_effective_positions"] > 0
        assert row["w_student_head_rows"] == row["w_allocated_rows"]
    # The checkpoint written after update 2 is aux time inside update 3's interval.
    assert rows[2]["aux_s"] > 0 and rows[0]["aux_s"] == rows[1]["aux_s"] == 0
    assert out["summary"]["instrument"]["checkpoint_s"] > 0


def test_stop_at_zero_writes_the_initial_state():
    saved = []
    out = _run(_cfg(stop_at_step=0), sink=saved.append)
    assert out["summary"]["completed_step"] == 0
    assert [state["step"] for state in saved] == [0]


def test_fork_reproduces_the_uninterrupted_continuation():
    states = {}
    full = _run(_cfg(), sink=lambda state: states.setdefault(state["step"], state))
    fork = copy.deepcopy(states[2])
    fork["fork"] = True
    forked = _run(_cfg(), resume_state=fork)
    assert [row["step"] for row in forked["history"]] == [3, 4, 5, 6]
    for mine, ref in zip(forked["history"], full["history"][2:]):
        assert mine["loss"] == pytest.approx(ref["loss"], abs=1e-9)


def test_fork_into_another_micro_batch_keeps_the_data_position(tmp_path):
    states = {}
    _run(_cfg(), sink=lambda state: states.setdefault(state["step"], state))
    fork = copy.deepcopy(states[2])
    fork["fork"] = True
    log_path = tmp_path / "train_log.jsonl"
    out = _run(_cfg(micro_batch_size=4, grad_accumulation_steps=1), resume_state=fork, log_path=log_path)
    rows = _rows(log_path)
    # 8 consumed samples = 2 micro-batches of 4; every later update adds one.
    assert [row["consumed_microbatches"] for row in rows] == [3, 4, 5, 6]
    assert out["summary"]["completed_step"] == 6
    bad = copy.deepcopy(states[2])
    bad["fork"] = True
    bad["consumed_samples"] = 6
    with pytest.raises(ValueError, match="whole number"):
        _run(_cfg(micro_batch_size=4, grad_accumulation_steps=1), resume_state=bad)


def test_timing_windows_and_segments(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    _run(_cfg(timing_windows=[[1, 4]], timing_segment_every=2), log_path=log_path, save_every=0)
    windows = _rows(tmp_path / "timing_windows.jsonl")
    assert len(windows) == 1
    window = windows[0]
    assert (window["start_step"], window["stop_step"], window["updates"], window["samples"]) == (1, 4, 3, 12)
    assert window["wall_s"] > 0 and window["synchronized"]
    assert window["workload"]["microbatches"] == 6
    segments = _rows(tmp_path / "timing_segments.jsonl")
    assert [(s["start_step"], s["stop_step"]) for s in segments] == [(0, 2), (2, 4), (4, 6)]


def test_validation_hook_boundaries():
    calls = []
    _run(_cfg(), hook=lambda step, epoch_end, final: calls.append((step, epoch_end, final)), save_every=0)
    assert calls[0] == (0, False, False)
    assert [call[0] for call in calls if not call[1]] == [0, 1, 2, 3, 4, 5, 6]
    assert calls[-1] == (6, False, True)


def test_feature_replay_reproduces_the_recorded_updates(tmp_path):
    states = {}
    _run(_cfg(), sink=lambda state: states.setdefault(state["step"], state))
    record = copy.deepcopy(states[2])
    record["fork"] = True
    replay_dir = tmp_path / "replay"
    recorded = _run(_cfg(feature_replay={"mode": "record", "dir": str(replay_dir), "microbatches": 8}),
                    resume_state=record)
    assert len(list(replay_dir.glob("mb*.pt"))) == 8
    replay = copy.deepcopy(states[2])
    replay["fork"] = True
    replayed = _run(_cfg(feature_replay={"mode": "replay", "dir": str(replay_dir)}), resume_state=replay)
    for mine, ref in zip(replayed["history"], recorded["history"]):
        assert mine["loss"] == pytest.approx(ref["loss"], abs=1e-9)


def test_resume_trims_timing_records_past_the_checkpoint(tmp_path):
    from vlm_dflash.dflash.train import _reconcile_train_log

    path = tmp_path / "timing_segments.jsonl"
    rows = [{"start_step": 0, "stop_step": 250}, {"start_step": 250, "stop_step": 500},
            {"start_step": 500, "stop_step": 750}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _reconcile_train_log(path, 500, step_key="stop_step")
    assert [json.loads(line)["stop_step"] for line in path.read_text().splitlines()] == [250, 500]
    assert json.loads((tmp_path / "timing_segments.uncommitted.jsonl").read_text())["stop_step"] == 750
