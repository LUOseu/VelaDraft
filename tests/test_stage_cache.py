"""Staged pipeline cache: reuse only when the recorded inputs digest matches.

Regression: `train` used to skip prepare/regen/capture whenever the output
file existed, silently reusing stale artifacts after dataset or config
changes. The digest of each stage's inputs is now recorded and compared, and
anything unexpected refuses before expensive compute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.cli import stage_digest, stage_digest_payload, stage_status, write_stage_manifest


def _contract(model_path: str = "/models/target") -> dict:
    return {
        "target": {"model_path": model_path, "model_fingerprint": "mf", "processor_fingerprint": "pf"},
        "mask": {"token_id": 151669},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"draft_layer_ids": [1, 12, 23]},
        "image": {"min_pixels": 200704, "max_pixels": 1003520},
    }


def _cfg() -> dict:
    return {
        "dataset": {"source": "jsonl", "path": "data/rows.jsonl"},
        "regen": {"max_new_tokens": 160},
        "capture": {"seq_length": 2048},
    }


def _record(tmp_path: Path, stage: str, marker: str, digest: str, upstream: str | None) -> None:
    path = tmp_path / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    write_stage_manifest(tmp_path, stage, stage_digest_payload(_cfg(), _contract(), stage, upstream), digest)


def test_missing_outputs_rerun(tmp_path):
    digest = stage_digest(_cfg(), _contract(), "data", None)
    assert stage_status(tmp_path, "data", "data/data.jsonl", digest, force=False) == "run"


def test_matching_digest_skips(tmp_path):
    digest = stage_digest(_cfg(), _contract(), "data", None)
    _record(tmp_path, "data", "data/data.jsonl", digest, None)
    assert stage_status(tmp_path, "data", "data/data.jsonl", digest, force=False) == "skip"


def test_outputs_without_manifest_refuse(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "data.jsonl").write_text("{}\n", encoding="utf-8")
    digest = stage_digest(_cfg(), _contract(), "data", None)
    with pytest.raises(SystemExit, match="no recorded inputs digest"):
        stage_status(tmp_path, "data", "data/data.jsonl", digest, force=False)


def test_changed_inputs_refuse_before_reuse(tmp_path):
    old_digest = stage_digest(_cfg(), _contract(), "data", None)
    _record(tmp_path, "data", "data/data.jsonl", old_digest, None)
    changed = _cfg()
    changed["dataset"]["slice"] = 128
    new_digest = stage_digest(changed, _contract(), "data", None)
    assert new_digest != old_digest
    with pytest.raises(SystemExit, match="different inputs"):
        stage_status(tmp_path, "data", "data/data.jsonl", new_digest, force=False)
    # --force rebuilds instead of refusing
    assert stage_status(tmp_path, "data", "data/data.jsonl", new_digest, force=True) == "run"


def test_digest_is_portable_across_model_dirs_but_not_contract_changes(tmp_path):
    a = stage_digest(_cfg(), _contract("/models/a"), "regen", "upstream")
    b = stage_digest(_cfg(), _contract("/models/b"), "regen", "upstream")
    assert a == b
    changed = _contract()
    changed["image"]["max_pixels"] = 802816
    assert stage_digest(_cfg(), changed, "regen", "upstream") != a
    assert stage_digest(_cfg(), _contract(), "regen", "other-upstream") != a


def test_manifest_round_trips_payload(tmp_path):
    digest = stage_digest(_cfg(), _contract(), "capture", "regen-digest")
    _record(tmp_path, "capture", "features/index.jsonl", digest, "regen-digest")
    manifest = json.loads((tmp_path / "capture" / "stage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "capture"
    assert manifest["inputs_digest"] == digest
    assert manifest["payload"]["upstream_digest"] == "regen-digest"
    assert "recorded_at" in manifest
