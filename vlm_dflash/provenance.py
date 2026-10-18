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


# Recorded for audit, but not part of the equality gate: the same model copied
# to another directory (same fingerprints) is the same target. Real model
# changes are caught by model_fingerprint / processor_fingerprint and the
# resolved contract fields below.
AUDIT_ONLY_IDENTITY_KEYS = frozenset({"target_model_path"})

#: Identity keys that define what a feature source *contains*: the frozen
#: target/processor, the vision-token / MRoPE / mask semantics and the ordered
#: list of captured target layers (plus the image bounds that shaped the
#: pixels). Reusing a valid capture is gated on exactly these keys.
SOURCE_IDENTITY_KEYS = frozenset(
    {
        "model_fingerprint",
        "processor_fingerprint",
        "mask_token_id",
        "spatial_rope_sections",
        "capture_layer_ids",
        "image_min_pixels",
        "image_max_pixels",
        "chat_template_kwargs",
    }
)

#: Identity keys that belong to the *consumer* model (which captured layers a
#: draft layer taps; i.e. the draft routing/depth). A new run may consume the
#: same store with a different routing as long as its consumed subset is inside
#: the store's capture list -- that check lives in the contract / feature-layout
#: validation, not in the source gate. Exact resume of the *same* run keeps
#: comparing the full identity.
CONSUMER_IDENTITY_KEYS = frozenset({"draft_layer_ids"})

#: Identity keys recorded only when set (absent = the default behavior), so
#: artifacts written before they existed keep their identity unchanged.
OPTIONAL_IDENTITY_KEYS = frozenset({"chat_template_kwargs"})


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


def identity_diff(
    expected: dict, actual: dict, ignore: frozenset[str] = AUDIT_ONLY_IDENTITY_KEYS
) -> list[str]:
    """Return human-readable differences between two identity records.

    Keys in ``ignore`` (audit-only, e.g. the local model path) never fail the
    gate; identity is decided by content fingerprints and resolved contract
    fields. Note the model fingerprint covers config/generation/index content
    plus weight shard names/sizes -- it identifies a checkpoint layout, not a
    byte-for-byte hash of the weights.
    """
    expected = _with_legacy_capture_layers(expected)
    actual = _with_legacy_capture_layers(actual)
    diffs = []
    for key in sorted(set(expected) | set(actual)):
        if key in ignore:
            continue
        if expected.get(key) != actual.get(key):
            diffs.append(f"{key}: expected={expected.get(key)!r} actual={actual.get(key)!r}")
    return diffs


def _with_legacy_capture_layers(identity: dict) -> dict:
    """Identity records written before ``capture_layer_ids`` existed captured
    exactly the draft routing (the old code's single layout). Defaulting the
    missing key to ``draft_layer_ids`` keeps those artifacts comparable instead
    of failing the gate on a field that never carried independent information.
    """
    if "capture_layer_ids" not in identity and "draft_layer_ids" in identity:
        return {**identity, "capture_layer_ids": list(identity["draft_layer_ids"])}
    return identity


def assert_identity(
    expected: dict, actual: dict, context: str, ignore: frozenset[str] = AUDIT_ONLY_IDENTITY_KEYS
) -> None:
    diffs = identity_diff(expected, actual, ignore=ignore)
    if diffs:
        joined = "\n  ".join(diffs)
        raise IdentityMismatchError(
            f"Target identity mismatch ({context}); refusing to continue.\n  {joined}\n"
            "Use a new run directory when switching target model/revision/processor settings."
        )


def source_identity(identity: dict) -> dict:
    """The feature-source subset of an identity record (legacy-compatible).

    Missing source keys compare as ``None``, so an artifact that records fewer
    fields than the live contract fails the gate instead of silently skipping
    the check. ``_with_legacy_capture_layers`` still recovers the true layout
    for stores written before ``capture_layer_ids`` existed.
    """
    identity = _with_legacy_capture_layers(identity)
    return {
        key: identity.get(key)
        for key in SOURCE_IDENTITY_KEYS
        if key in identity or key not in OPTIONAL_IDENTITY_KEYS
    }


def source_identity_from_contract(contract: dict) -> dict:
    """Feature-source identity subset of the resolved contract."""
    return source_identity(identity_from_contract(contract))


def assert_source_identity(
    expected: dict, actual: dict, context: str, ignore: frozenset[str] = AUDIT_ONLY_IDENTITY_KEYS
) -> None:
    """Gate feature-store reuse on the source identity only.

    The store's content is defined by the captured layers (and the target /
    processor / mask / MRoPE / image bounds that produced them), not by the
    consumer's routing: a run whose draft depth differs may reuse the same
    capture when its consumed layers are a subset of the stored ones (enforced
    by the feature-layout check) -- rejecting that was a false negative.
    Consumer identity (``draft_layer_ids``) and exact-resume semantics stay
    strict in :func:`assert_identity` and the training-plan digest.
    """
    expected_source = source_identity(expected)
    actual_source = source_identity(actual)
    diffs = []
    for key in sorted(SOURCE_IDENTITY_KEYS):
        if key in ignore:
            continue
        if expected_source.get(key) != actual_source.get(key):
            diffs.append(
                f"{key}: expected={expected_source.get(key)!r} actual={actual_source.get(key)!r}"
            )
    if diffs:
        joined = "\n  ".join(diffs)
        raise IdentityMismatchError(
            f"Feature-source identity mismatch ({context}); refusing to reuse the capture.\n  {joined}\n"
            "Recapture the features for this target/processor/layer list, or point the run at the "
            "store that matches its contract."
        )


def identity_from_contract(contract: dict) -> dict:
    """Canonical identity subset used for all stage gates.

    ``target_model_path`` is recorded for audit but excluded from the equality
    gate (see :data:`AUDIT_ONLY_IDENTITY_KEYS`); moving the same checkpoint to
    a new directory must not invalidate a run.
    """
    from .contract import contract_capture_layer_ids, contract_chat_template_kwargs

    identity = {
        "target_model_path": contract["target"]["model_path"],
        "model_fingerprint": contract["target"]["model_fingerprint"],
        "processor_fingerprint": contract["target"]["processor_fingerprint"],
        "mask_token_id": contract["mask"]["token_id"],
        "spatial_rope_sections": contract["rope"]["sections"],
        "draft_layer_ids": contract["layers"]["draft_layer_ids"],
        "capture_layer_ids": contract_capture_layer_ids(contract),
        "image_min_pixels": contract["image"]["min_pixels"],
        "image_max_pixels": contract["image"]["max_pixels"],
    }
    # Only recorded when set, so identities (and stage digests) of runs that
    # render with the template defaults stay unchanged.
    template_kwargs = contract_chat_template_kwargs(contract)
    if template_kwargs:
        identity["chat_template_kwargs"] = template_kwargs
    return identity
