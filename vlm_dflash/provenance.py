"""Provenance records and the teacher/target identity gate.

Any stage that consumes artifacts produced from a specific target model
(regen -> capture -> train -> eval) records its target identity and refuses to
continue when the identity of the upstream artifact does not match. There is no
bypass: switching target models requires a new run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import now_iso, read_json, write_json


class IdentityMismatchError(RuntimeError):
    """Raised when an artifact's target identity differs from the live one."""


def provenance_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "provenance.json"


def load_provenance(run_dir: str | Path) -> dict:
    path = provenance_path(run_dir)
    if path.exists():
        return read_json(path)
    return {"created": now_iso(), "stages": {}}


def record_stage(run_dir: str | Path, stage: str, payload: dict) -> dict:
    prov = load_provenance(run_dir)
    entry = dict(payload)
    entry["recorded_at"] = now_iso()
    prov["stages"][stage] = entry
    prov["updated"] = now_iso()
    write_json(provenance_path(run_dir), prov)
    return prov


def identity_diff(expected: dict, actual: dict) -> list[str]:
    """Return human-readable differences between two identity records."""
    diffs = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            diffs.append(f"{key}: expected={expected.get(key)!r} actual={actual.get(key)!r}")
    return diffs


def assert_identity(expected: dict, actual: dict, context: str) -> None:
    diffs = identity_diff(expected, actual)
    if diffs:
        joined = "\n  ".join(diffs)
        raise IdentityMismatchError(
            f"Target identity mismatch ({context}); refusing to continue.\n  {joined}\n"
            "Use a new run directory when switching target model/revision/processor settings."
        )


def identity_from_contract(contract: dict) -> dict:
    """Canonical identity subset used for all stage gates."""
    return {
        "target_model_path": contract["target"]["model_path"],
        "model_fingerprint": contract["target"]["model_fingerprint"],
        "processor_fingerprint": contract["target"]["processor_fingerprint"],
        "mask_token_id": contract["mask"]["token_id"],
        "spatial_rope_sections": contract["rope"]["sections"],
        "draft_layer_ids": contract["layers"]["draft_layer_ids"],
        "image_min_pixels": contract["image"]["min_pixels"],
        "image_max_pixels": contract["image"]["max_pixels"],
    }
