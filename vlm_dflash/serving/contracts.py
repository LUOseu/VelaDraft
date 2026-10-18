"""Lightweight wire contracts for the VelaDraft serving extension.

The platform task deliberately does not edit the repository-wide
``runtime_contracts.py``.  This module is the self-contained serializer and
validator used by ``scripts/run_serving_gate.py`` and by the JSON records that
are exchanged between the offline correctness gate and the HTTP measurement
harness.

Schema version: ``veladraft-platform-v1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA_VERSION = "veladraft-platform-v1"
PROJECT_SCOPE = "platform_extension"
INCLUDE_IN_PAPER = False

CORRECTNESS_STATUSES = {
    "not_checked",
    "verified_greedy_exact",
    "verified_semantic_acceptance",
    "experimental_numeric_divergence",
    "failed",
    "unsupported",
}
CAPABILITY_STATUSES = {"supported", "experimental", "unsupported", "untested"}
MATURITY_LEVELS = {"implemented", "cpu_tested", "gpu_smoke", "validated", "blocked"}

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "run_id",
    "request_id",
    "group_id",
    "target_identity",
    "draft_identity",
    "backend_identity",
    "protocol_hash",
    "correctness_status",
    "measurement_scope",
    "raw_counts",
    "latency",
    "resource_usage",
)
REQUIRED_RAW_COUNTS = (
    "emitted",
    "proposed",
    "accepted",
    "verify_rounds",
    "target_forward_calls",
    "draft_forward_calls",
)
REQUIRED_TARGET_IDENTITY = (
    "model",
    "revision",
    "tokenizer",
    "processor",
    "template",
    "precision",
    "missing_reasons",
)
REQUIRED_DRAFT_IDENTITY = (
    "kind",
    "model",
    "revision",
    "tokenizer",
    "precision",
    "params",
    "missing_reasons",
)
REQUIRED_BACKEND_IDENTITY = (
    "backend",
    "version",
    "code_commit",
    "device_topology",
)
CAPABILITY_IDS = (
    "ar_image",
    "ngram_image",
    "dflash_image",
    "stream",
    "cancel",
    "round_trace",
    "return_token_ids",
    "metrics",
)


class ContractError(ValueError):
    """Raised when a record does not satisfy the platform wire contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(payload: dict[str, Any]) -> str:
    """Hash the identity/generation/timing inputs covered by a protocol.

    Callers pass the fields that actually affect the run.  The canonical JSON
    encoding makes the hash stable across dict insertion order and platforms.
    """

    if not isinstance(payload, dict) or not payload:
        raise ContractError("protocol_hash payload must be a non-empty dict")
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_string(value: Any, path: str, allow_empty: bool = False) -> None:
    _require(isinstance(value, str), f"{path} must be a string")
    if not allow_empty:
        _require(bool(value.strip()), f"{path} must be non-empty")


def _require_mapping(value: Any, path: str) -> None:
    _require(isinstance(value, dict), f"{path} must be an object")


def _validate_identity(identity: Any, path: str, required: Iterable[str]) -> None:
    _require_mapping(identity, path)
    for key in required:
        _require(key in identity, f"{path}.{key} is required")
    reasons = identity.get("missing_reasons")
    _require(isinstance(reasons, list), f"{path}.missing_reasons must be a list")
    _require(all(isinstance(item, str) for item in reasons), f"{path}.missing_reasons must contain strings")


def _validate_raw_counts(record: dict[str, Any]) -> None:
    counts = record["raw_counts"]
    _require_mapping(counts, "raw_counts")
    for key in REQUIRED_RAW_COUNTS:
        _require(key in counts, f"raw_counts.{key} is required")
        value = counts[key]
        if value is not None:
            _require(isinstance(value, int), f"raw_counts.{key} must be an int or null")
            _require(value >= 0, f"raw_counts.{key} must be >= 0")
    unavailable = counts.get("unavailable", {})
    _require_mapping(unavailable, "raw_counts.unavailable")
    missing_reasons = record.get("missing_reasons", {})
    _require_mapping(missing_reasons, "missing_reasons")
    for key in REQUIRED_RAW_COUNTS:
        if counts[key] is None:
            reason = unavailable.get(key) or missing_reasons.get(f"raw_counts.{key}")
            _require(
                isinstance(reason, str) and bool(reason.strip()),
                f"null raw_counts.{key} needs a reason in raw_counts.unavailable or missing_reasons",
            )


def _validate_latency(record: dict[str, Any]) -> None:
    latency = record["latency"]
    _require_mapping(latency, "latency")
    _require(bool(latency), "latency must record at least one timing field")
    saw_seconds = False
    for key, value in latency.items():
        if key.endswith("_per_s") or "tokens_per" in key:
            raise ContractError(f"latency.{key} is a rate, not a latency field")
        if key.endswith("_s"):
            saw_seconds = True
            if value is not None:
                _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"latency.{key} must be numeric or null")
                _require(float(value) >= 0.0, f"latency.{key} must be >= 0")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            raise ContractError(f"latency.{key} is numeric but does not end in '_s'")
    _require(saw_seconds, "latency must contain at least one '*_s' field")


def _validate_resource_usage(record: dict[str, Any]) -> None:
    usage = record["resource_usage"]
    _require_mapping(usage, "resource_usage")
    for key in ("gpu_hours", "gpu_count", "devices", "peak_memory_mib", "wall_s"):
        _require(key in usage, f"resource_usage.{key} is required")
    gpu_hours = usage["gpu_hours"]
    _require(
        gpu_hours is None or (isinstance(gpu_hours, (int, float)) and not isinstance(gpu_hours, bool)),
        "resource_usage.gpu_hours must be numeric or null",
    )
    if gpu_hours is None:
        reason = usage.get("gpu_hours_reason")
        _require(isinstance(reason, str) and bool(reason.strip()), "null resource_usage.gpu_hours needs gpu_hours_reason")
    else:
        _require(float(gpu_hours) >= 0.0, "resource_usage.gpu_hours must be >= 0")
    _require(isinstance(usage["gpu_count"], int) and usage["gpu_count"] >= 1, "resource_usage.gpu_count must be >= 1")
    _require(isinstance(usage["devices"], list), "resource_usage.devices must be a list")
    _require_mapping(usage["peak_memory_mib"], "resource_usage.peak_memory_mib")
    _require(isinstance(usage["wall_s"], (int, float)) and not isinstance(usage["wall_s"], bool), "resource_usage.wall_s must be numeric")
    _require(float(usage["wall_s"]) >= 0.0, "resource_usage.wall_s must be >= 0")
    if "additive" in usage:
        _require(isinstance(usage["additive"], bool), "resource_usage.additive must be boolean")
    if usage.get("gpu_hours_scope") in {"request_exposure", "window_exposure"}:
        _require(usage.get("additive") is False, "request/window exposure must set additive=false")


def validate_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one RunRecord and return it unchanged.

    The validator is intentionally strict about the fields that are easy to
    silently fake (unknown counters, non-second timing fields, missing identity
    reasons) and permissive about extension fields.
    """

    _require_mapping(record, "record")
    for key in REQUIRED_TOP_LEVEL:
        _require(key in record, f"missing top-level field: {key}")
    _require(record["schema_version"] == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION!r}")
    _require(record["project_scope"] == PROJECT_SCOPE, f"project_scope must be {PROJECT_SCOPE!r}")
    _require(record["include_in_paper"] is False, "include_in_paper must be false")
    for key in ("run_id", "request_id", "group_id"):
        _require_string(record[key], key)
    _validate_identity(record["target_identity"], "target_identity", REQUIRED_TARGET_IDENTITY)
    if record["draft_identity"] is not None:
        _validate_identity(record["draft_identity"], "draft_identity", REQUIRED_DRAFT_IDENTITY)
    else:
        reasons = record.get("missing_reasons", {}).get("draft_identity")
        _require(
            isinstance(reasons, str) and bool(reasons.strip()),
            "null draft_identity needs missing_reasons['draft_identity'] (for example, AR baseline)",
        )
    _validate_identity(record["backend_identity"], "backend_identity", REQUIRED_BACKEND_IDENTITY)
    _require_string(record["protocol_hash"], "protocol_hash")
    _require(len(record["protocol_hash"]) == 64, "protocol_hash must be a 64-char sha256 hex digest")
    _require(all(c in "0123456789abcdef" for c in record["protocol_hash"]), "protocol_hash must be lowercase hex")
    _require(record["correctness_status"] in CORRECTNESS_STATUSES, f"unknown correctness_status {record['correctness_status']!r}")
    _require_mapping(record["measurement_scope"], "measurement_scope")
    _require(bool(record["measurement_scope"]), "measurement_scope must not be empty")
    _validate_raw_counts(record)
    _validate_latency(record)
    _validate_resource_usage(record)
    return record


def validate_run_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for record in records:
        validate_run_record(record)
        key = (record["run_id"], record["request_id"], record.get("repeat_index"))
        _require(key not in seen, f"duplicate record key: {key}")
        seen.add(key)
        out.append(record)
    _require(bool(out), "at least one RunRecord is required")
    return out


def validate_capabilities(capabilities: dict[str, Any]) -> dict[str, Any]:
    _require_mapping(capabilities, "capabilities")
    _require(capabilities.get("schema_version") == SCHEMA_VERSION, "capabilities schema_version mismatch")
    _require(capabilities.get("project_scope") == PROJECT_SCOPE, "capabilities project_scope mismatch")
    _require(capabilities.get("include_in_paper") is False, "capabilities include_in_paper must be false")
    for key in ("generated_at", "vllm_pin", "environment", "capabilities"):
        _require(key in capabilities, f"capabilities.{key} is required")
    _require_mapping(capabilities["environment"], "capabilities.environment")
    items = capabilities["capabilities"]
    _require_mapping(items, "capabilities.capabilities")
    for cap_id in CAPABILITY_IDS:
        _require(cap_id in items, f"capability {cap_id!r} is required")
    for cap_id, item in items.items():
        path = f"capabilities.{cap_id}"
        _require_mapping(item, path)
        _require(item.get("status") in CAPABILITY_STATUSES, f"{path}.status must be one of {sorted(CAPABILITY_STATUSES)}")
        _require(item.get("maturity") in MATURITY_LEVELS, f"{path}.maturity must be one of {sorted(MATURITY_LEVELS)}")
        _require_string(item.get("version", ""), f"{path}.version")
        _require(isinstance(item.get("evidence"), list) and item["evidence"], f"{path}.evidence must be a non-empty list")
        _require(isinstance(item.get("limits"), list), f"{path}.limits must be a list")
    return capabilities


# ---------------------------------------------------------------------------
# Session / Proposal / Commit state machine
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """An isolated request plus the committed prefix version it owns."""

    session_id: str
    target_identity: dict[str, Any]
    media_sha256: Optional[str] = None
    prefix_version: int = 0
    committed_token_ids: list[int] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ContractError("session_id must be non-empty")
        if self.prefix_version < 0:
            raise ContractError("prefix_version must be >= 0")


@dataclass
class Proposal:
    """A candidate draft that must not be emitted before verification."""

    request_id: str
    based_on_prefix_version: int
    token_ids: list[int]
    source: str
    position_semantics: str = "target_token_positions"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ContractError("request_id must be non-empty")
        if self.based_on_prefix_version < 0:
            raise ContractError("based_on_prefix_version must be >= 0")
        if not self.token_ids:
            raise ContractError("a Proposal must carry at least one candidate token")
        if any(not isinstance(token, int) or token < 0 for token in self.token_ids):
            raise ContractError("proposal token ids must be non-negative ints")


@dataclass
class Commit:
    """Only verified tokens and visible rollback information."""

    session_id: str
    prefix_version: int
    token_ids: list[int]
    accepted: int
    rejected: int = 0
    reason: Optional[str] = None
    eos: bool = False

    def __post_init__(self) -> None:
        if self.prefix_version < 1:
            raise ContractError("commit prefix_version must be >= 1")
        if self.accepted < 0 or self.rejected < 0:
            raise ContractError("accepted/rejected must be >= 0")
        if any(not isinstance(token, int) or token < 0 for token in self.token_ids):
            raise ContractError("commit token ids must be non-negative ints")


def commit_proposal(session: SessionState, proposal: Proposal, commit: Commit) -> SessionState:
    """Apply a verified commit, rejecting stale or cross-session events.

    The engine owns verification; this helper only enforces the observable
    session/pre-fix-version boundary.  A stale proposal (for example one that
    arrives after a cancellation) cannot advance the session.
    """

    if session.closed:
        raise ContractError("cannot commit to a closed session")
    if commit.session_id != session.session_id:
        raise ContractError("cross-session commit")
    if proposal.request_id != commit.session_id:
        raise ContractError("commit session_id does not match proposal request_id")
    if proposal.based_on_prefix_version != session.prefix_version:
        raise ContractError(
            f"stale proposal: based_on_prefix_version={proposal.based_on_prefix_version} "
            f"but current prefix_version={session.prefix_version}"
        )
    if commit.prefix_version != session.prefix_version + 1:
        raise ContractError("commit must advance prefix_version by exactly one")
    if commit.accepted > len(proposal.token_ids):
        raise ContractError("accepted cannot exceed proposed token count")
    if commit.token_ids[: commit.accepted] != proposal.token_ids[: commit.accepted]:
        raise ContractError("committed prefix must match the verified proposal prefix")
    if commit.rejected not in (0, len(proposal.token_ids) - commit.accepted):
        # Engines may report a bonus-token rollback; be explicit rather than
        # silently accepting an unexplained mismatch.
        raise ContractError("rejected must be zero or the unaccepted proposal suffix length")
    session.committed_token_ids.extend(commit.token_ids)
    session.prefix_version = commit.prefix_version
    if commit.eos:
        session.closed = True
    return session


def close_session(session: SessionState) -> SessionState:
    session.closed = True
    return session
