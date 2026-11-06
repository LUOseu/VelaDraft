"""Private JSONL workload contract (manifest + rows + labels + local media).

First-round scope is text + local images. Video, VLA actions, tool calls and
multi-turn state advancement are imported as ``unsupported`` rows instead of
being silently flattened into single-turn text.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    SCHEMA_VERSION,
    sha256_file,
    sha256_json,
    validate_manifest_shape,
)

SPLITS = ("tuning", "holdout")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_UNSUPPORTED_MEDIA = {"video", "audio", "action", "trajectory", "pointcloud"}
_UNSUPPORTED_ROLES = {"tool", "function"}


def _error(code: str, message: str, *, line: int | None = None, path: str = "$") -> dict:
    issue = {"code": code, "message": message, "path": path}
    if line is not None:
        issue["line"] = line
    return issue


def _read_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(
                    _error(
                        "bad_json",
                        f"invalid JSON at line {line_number}: {exc.msg}",
                        line=line_number,
                    )
                )
                continue
            if not isinstance(value, dict):
                errors.append(
                    _error("row_not_object", f"line {line_number} is not a JSON object.", line=line_number)
                )
                continue
            value["__line__"] = line_number
            rows.append(value)
    return rows, errors


def _safe_relative(base: Path, reference: str, *, label: str) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to {base}: got absolute path {reference!r}.")
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes {base}: {reference!r}") from exc
    return resolved


def _image_kind(path: Path) -> str | None:
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        return None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _validate_messages(messages: Any) -> tuple[list[dict], str | None]:
    if not isinstance(messages, list) or not messages:
        return [], "messages must be a non-empty list"
    cleaned: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return [], f"messages[{index}] must be an object"
        role = str(message.get("role", "")).lower()
        if role in _UNSUPPORTED_ROLES:
            return [], f"messages[{index}].role={role!r} requires tool/state support"
        if role not in ("system", "user", "assistant"):
            return [], f"messages[{index}].role={role!r} is unsupported"
        if message.get("tool_calls") or message.get("function_call"):
            return [], f"messages[{index}] contains tool calls (unsupported first round)"
        if message.get("actions") is not None:
            return [], f"messages[{index}] contains actions (VLA unsupported)"
        content = _content_text(message.get("content"))
        if not content.strip():
            return [], f"messages[{index}].content is empty"
        cleaned.append({"role": role, "content": content})
    return cleaned, None


def _validate_media(row: dict, media_root: Path) -> tuple[list[dict], list[str], str | None]:
    media = row.get("media", [])
    if media is None:
        media = []
    if not isinstance(media, list):
        return [], [], "media must be a list"
    cleaned: list[dict] = []
    hashes: list[str] = []
    for index, item in enumerate(media):
        if not isinstance(item, dict):
            return [], [], f"media[{index}] must be an object"
        media_type = str(item.get("type", "")).lower()
        if media_type in _UNSUPPORTED_MEDIA:
            return [], [], f"media[{index}].type={media_type!r} is unsupported first round"
        if media_type != "image":
            return [], [], f"media[{index}].type={media_type!r} is unsupported"
        reference = item.get("path")
        if not isinstance(reference, str) or not reference:
            return [], [], f"media[{index}].path is required"
        try:
            path = _safe_relative(media_root, reference, label=f"media[{index}].path")
        except ValueError as exc:
            return [], [], str(exc)
        if not path.is_file():
            return [], [], f"media[{index}] not found: {reference}"
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return [], [], f"media[{index}] has unsupported suffix {path.suffix!r}"
        kind = _image_kind(path)
        if kind is None:
            return [], [], f"media[{index}] is not a decodable PNG/JPEG/WebP image"
        digest = sha256_file(path)
        hashes.append(digest)
        cleaned.append({"type": "image", "path": reference, "sha256": digest, "kind": kind})
    return cleaned, hashes, None


def _row_identity(row: dict) -> dict:
    return {
        "request_id": row["request_id"],
        "group_id": row["group_id"],
        "task": row["task"],
        "messages_sha256": sha256_json(row["messages"]),
        "media_sha256": [item["sha256"] for item in row.get("media", [])],
    }


@dataclass
class Workload:
    manifest: dict
    manifest_path: Path
    manifest_dir: Path
    media_root: Path
    rows: list[dict] = field(default_factory=list)
    labels: dict[str, dict] = field(default_factory=dict)
    rows_digest: str | None = None
    labels_digest: str | None = None
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    unsupported: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def counts(self) -> dict:
        per_split: dict[str, int] = {}
        per_task: dict[str, int] = {}
        groups: dict[str, set] = {}
        for row in self.rows:
            per_split[row["split"]] = per_split.get(row["split"], 0) + 1
            per_task[row["task"]] = per_task.get(row["task"], 0) + 1
            groups.setdefault(row["split"], set()).add(row["group_id"])
        return {
            "requests": len(self.rows),
            "groups": sum(len(value) for value in groups.values()),
            "requests_by_split": per_split,
            "groups_by_split": {key: len(value) for key, value in groups.items()},
            "requests_by_task": per_task,
        }

    def request_identity(self, row: dict) -> dict:
        return _row_identity(row)


def _load_labels(path: Path, row_ids: set[str]) -> tuple[dict[str, dict], list[dict], list[dict], str]:
    labels: dict[str, dict] = {}
    errors: list[dict] = []
    warnings: list[dict] = []
    rows, parse_errors = _read_jsonl(path)
    errors.extend(parse_errors)
    raw_labels = [dict(item) for item in rows]
    for label in rows:
        line = label.pop("__line__", None)
        request_id = label.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            errors.append(_error("missing_request_id", "label row has no request_id.", line=line))
            continue
        if request_id in labels:
            errors.append(_error("duplicate_label", f"duplicate label for {request_id!r}.", line=line))
            continue
        if request_id not in row_ids:
            errors.append(
                _error("label_without_row", f"label {request_id!r} has no matching workload row.", line=line)
            )
            continue
        reference = label.get("reference")
        references = label.get("references")
        if reference is None and references is None:
            errors.append(_error("missing_reference", f"label {request_id!r} has no reference.", line=line))
            continue
        if references is not None and not isinstance(references, list):
            errors.append(_error("invalid_references", f"label {request_id!r} references must be a list.", line=line))
            continue
        labels[request_id] = {
            "reference": reference,
            "references": references,
            "scorer": label.get("scorer"),
            "task": label.get("task"),
        }
    missing = sorted(row_ids - set(labels))
    if missing:
        warnings.append(
            _error(
                "missing_labels",
                f"{len(missing)} request(s) have no label; task quality will be not_evaluated for them.",
            )
        )
    return labels, errors, warnings, sha256_json(raw_labels)


def load_workload(manifest_path: str | Path, *, strict_counts: bool = True) -> Workload:
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.resolve().parent
    errors: list[dict] = []
    warnings: list[dict] = []
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest {manifest_path} must be a JSON object.")

    shape_errors = validate_manifest_shape(manifest)
    if shape_errors:
        return Workload(
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_dir=manifest_dir,
            media_root=manifest_dir,
            errors=shape_errors,
        )

    try:
        media_root = _safe_relative(manifest_dir, str(manifest["media_root"]), label="media_root")
        rows_path = _safe_relative(manifest_dir, str(manifest["rows_path"]), label="rows_path")
    except ValueError as exc:
        return Workload(
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_dir=manifest_dir,
            media_root=manifest_dir,
            errors=[_error("unsafe_path", str(exc))],
        )
    if not media_root.is_dir():
        errors.append(_error("missing_media_root", f"media_root not found: {media_root}"))
    if not rows_path.is_file():
        errors.append(_error("missing_rows", f"rows_path not found: {rows_path}"))

    raw_rows, row_errors = ([], []) if not rows_path.is_file() else _read_jsonl(rows_path)
    errors.extend(row_errors)

    expected_hashes = manifest.get("hashes") or {}
    rows_digest = sha256_file(rows_path) if rows_path.is_file() else None
    if expected_hashes.get("rows_sha256") and rows_digest != expected_hashes["rows_sha256"]:
        errors.append(
            _error(
                "rows_hash_mismatch",
                f"rows JSONL sha256 {rows_digest} != frozen {expected_hashes['rows_sha256']}",
            )
        )
    if not expected_hashes.get("rows_sha256"):
        warnings.append(_error("unfrozen_rows", "manifest has no hashes.rows_sha256; recording computed digest."))
    sd_candidates = [item for item in manifest.get("draft_candidates") or [] if item.get("method") == "sd"]
    if len(sd_candidates) > 2:
        warnings.append(
            _error(
                "too_many_candidates",
                f"{len(sd_candidates)} speculative candidates are declared; first-round tuning compares at most two.",
            )
        )

    rows: list[dict] = []
    unsupported: list[dict] = []
    request_ids: dict[str, int] = {}
    group_split: dict[str, str] = {}
    group_task: dict[str, str] = {}
    group_line: dict[str, int] = {}
    for row in raw_rows:
        line = row.pop("__line__", None)
        request_id = row.get("request_id")
        group_id = row.get("group_id")
        task = row.get("task")
        split = row.get("split")
        if not isinstance(request_id, str) or not request_id:
            errors.append(_error("missing_request_id", "row has no stable request_id.", line=line))
            continue
        if request_id in request_ids:
            errors.append(
                _error(
                    "duplicate_request_id",
                    f"duplicate request_id {request_id!r} (first at line {request_ids[request_id]}).",
                    line=line,
                )
            )
            continue
        request_ids[request_id] = line or 0
        if not isinstance(group_id, str) or not group_id:
            errors.append(_error("missing_group_id", f"{request_id}: group_id is required.", line=line))
            continue
        if not isinstance(task, str) or not task:
            errors.append(_error("missing_task", f"{request_id}: task is required.", line=line))
            continue
        if split not in SPLITS:
            errors.append(_error("invalid_split", f"{request_id}: split must be one of {SPLITS}.", line=line))
            continue

        messages, message_error = _validate_messages(row.get("messages"))
        if message_error:
            unsupported.append(
                {"request_id": request_id, "group_id": group_id, "reason": message_error, "line": line}
            )
            continue

        if row.get("action") is not None or row.get("trajectory") is not None:
            unsupported.append(
                {"request_id": request_id, "group_id": group_id, "reason": "VLA action/trajectory", "line": line}
            )
            continue

        media, media_hashes, media_error = _validate_media(row, media_root)
        if media_error:
            if "unsupported" in media_error or media_error.startswith("media must"):
                unsupported.append(
                    {"request_id": request_id, "group_id": group_id, "reason": media_error, "line": line}
                )
            else:
                errors.append(_error("bad_media", f"{request_id}: {media_error}", line=line))
            continue
        expected = row.get("expected_output_tokens")
        if expected is not None and (not isinstance(expected, (int, float)) or isinstance(expected, bool) or expected < 0):
            errors.append(
                _error("invalid_expected_tokens", f"{request_id}: expected_output_tokens must be numeric.", line=line)
            )
            continue

        if group_id in group_split and group_split[group_id] != split:
            errors.append(
                _error(
                    "group_split_leak",
                    f"group {group_id!r} appears in both {group_split[group_id]!r} and {split!r} "
                    f"(first seen line {group_line[group_id]}).",
                    line=line,
                )
            )
            continue
        if group_id in group_task and group_task[group_id] != task:
            errors.append(
                _error(
                    "group_task_mismatch",
                    f"group {group_id!r} spans tasks {group_task[group_id]!r} and {task!r}.",
                    line=line,
                )
            )
            continue
        group_split[group_id] = split
        group_task[group_id] = task
        group_line[group_id] = line or 0

        identity = _row_identity(
            {"request_id": request_id, "group_id": group_id, "task": task, "messages": messages, "media": media}
        )
        cleaned = {
            "request_id": request_id,
            "group_id": group_id,
            "task": task,
            "split": split,
            "messages": messages,
            "media": media,
            "expected_output_tokens": expected,
            "input_sha256": sha256_json(identity),
        }
        rows.append(cleaned)

    labels: dict[str, dict] = {}
    labels_digest = None
    labels_path_value = manifest.get("labels_path")
    if labels_path_value:
        try:
            labels_path = _safe_relative(manifest_dir, str(labels_path_value), label="labels_path")
        except ValueError as exc:
            errors.append(_error("unsafe_path", str(exc)))
        else:
            if not labels_path.is_file():
                errors.append(_error("missing_labels", f"labels_path not found: {labels_path}"))
            else:
                labels_digest = sha256_file(labels_path)
                if expected_hashes.get("labels_sha256") and labels_digest != expected_hashes["labels_sha256"]:
                    errors.append(
                        _error(
                            "labels_hash_mismatch",
                            f"labels JSONL sha256 {labels_digest} != frozen {expected_hashes['labels_sha256']}",
                        )
                    )
                labels, label_errors, label_warnings, _ = _load_labels(labels_path, set(request_ids))
                errors.extend(label_errors)
                warnings.extend(label_warnings)

    split_counts = {}
    for split in SPLITS:
        split_counts[split] = len({row["group_id"] for row in rows if row["split"] == split})
    frozen_splits = manifest.get("splits") or {}
    if strict_counts and frozen_splits:
        for split in SPLITS:
            expected = frozen_splits.get(split)
            if isinstance(expected, int) and expected != split_counts.get(split, 0):
                errors.append(
                    _error(
                        "split_count_mismatch",
                        f"manifest freezes {expected} {split} groups but rows contain {split_counts.get(split, 0)}.",
                    )
                )

    if not rows and not unsupported:
        errors.append(_error("empty_workload", "workload has no runnable or unsupported rows."))

    return Workload(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_dir=manifest_dir,
        media_root=media_root,
        rows=rows,
        labels=labels,
        rows_digest=rows_digest,
        labels_digest=labels_digest,
        errors=errors,
        warnings=warnings,
        unsupported=unsupported,
    )


def validation_report(workload: Workload) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "manifest": str(workload.manifest_path),
        "workload_id": workload.manifest.get("workload_id"),
        "status": "ok" if workload.ok else "failed",
        "overall_status": "ok" if workload.ok and not workload.unsupported else "partial",
        "rows_sha256": workload.rows_digest,
        "labels_sha256": workload.labels_digest,
        "counts": workload.counts,
        "errors": workload.errors,
        "warnings": workload.warnings,
        "unsupported": workload.unsupported,
        "labels_present": len(workload.labels),
    }


def convert_sharegpt(
    source: str | Path,
    out_dir: str | Path,
    *,
    media_root: str = "images",
    task: str = "general_vqa",
    split: str = "holdout",
    group_from_id: bool = True,
    last_turn_only: bool = False,
) -> dict:
    """Deterministic legacy ShareGPT -> platform rows conversion.

    Answers are intentionally dropped: the workload row is prompt-only. Any
    legacy evaluation that only consumed the last turn is reported as
    ``legacy_last_turn_only`` so the semantic change is visible.
    """
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = _read_jsonl(source)
    converted: list[dict] = []
    label_rows: list[dict] = []
    notes: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        line = row.pop("__line__", None)
        sample_id = str(row.get("id") or (f"row-{line:05d}" if line else f"row-{len(converted):05d}"))
        if sample_id in seen_ids:
            errors.append(_error("duplicate_sample_id", f"{sample_id}: duplicate source id.", line=line))
            continue
        seen_ids.add(sample_id)
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            errors.append(_error("missing_conversations", f"{sample_id}: conversations missing.", line=line))
            continue
        turns = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            role = {"human": "user", "gpt": "assistant", "system": "system"}.get(role, role)
            turns.append({"role": role, "content": str(turn.get("value", ""))})
        full_cleaned, message_error = _validate_messages(turns)
        if message_error:
            errors.append(_error("legacy_unsupported", f"{sample_id}: {message_error}", line=line))
            continue
        last_user = max((i for i, turn in enumerate(full_cleaned) if turn["role"] == "user"), default=None)
        if last_user is None:
            errors.append(_error("no_user_turn", f"{sample_id}: no user turn.", line=line))
            continue
        trailing = full_cleaned[last_user + 1 :]
        if any(turn["role"] != "assistant" for turn in trailing):
            errors.append(
                _error(
                    "unsupported_state_transition",
                    f"{sample_id}: non-assistant turn after the last user turn would require state advancement.",
                    line=line,
                )
            )
            continue
        if len(trailing) > 1:
            errors.append(
                _error(
                    "multiple_terminal_answers",
                    f"{sample_id}: multiple assistant turns follow the last user turn; refusing to flatten them.",
                    line=line,
                )
            )
            continue
        terminal_answer = trailing[0]["content"] if trailing else None
        prompt_turns = [full_cleaned[last_user]] if last_turn_only else full_cleaned[: last_user + 1]
        if last_turn_only:
            notes.append(f"{sample_id}: converted with legacy_last_turn_only=true")
        if terminal_answer is None:
            notes.append(f"{sample_id}: no terminal assistant answer; no label sidecar row")
        else:
            label_rows.append(
                {
                    "request_id": sample_id,
                    "task": task,
                    "reference": terminal_answer,
                    "source": "sharegpt_terminal_assistant_turn",
                }
            )
        media = []
        images = row.get("images") or []
        if not isinstance(images, list):
            errors.append(_error("invalid_images", f"{sample_id}: images must be a list.", line=line))
            continue
        for image in images:
            media.append({"type": "image", "path": str(image)})
        cleaned, message_error = _validate_messages(prompt_turns)
        if message_error:
            errors.append(_error("legacy_unsupported", f"{sample_id}: {message_error}", line=line))
            continue
        converted.append(
            {
                "request_id": sample_id,
                "group_id": sample_id if group_from_id else f"group-{sample_id}",
                "task": task,
                "messages": cleaned,
                "media": media,
                "split": split,
            }
        )
    rows_path = out_dir / "rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    labels_path = out_dir / "labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for row in label_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "output": str(rows_path),
        "source_sha256": sha256_file(source),
        "rows_sha256": sha256_file(rows_path) if rows_path.exists() else None,
        "labels_path": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "converted": len(converted),
        "labels": len(label_rows),
        "errors": errors,
        "semantics": {
            "answers_included": False,
            "answer_location": "private_labels_sidecar" if label_rows else "no_terminal_answers_found",
            "prompt_ends_at_last_user_turn": True,
            "media_root": media_root,
            "legacy_last_turn_only": bool(last_turn_only),
            "history_preserved": not bool(last_turn_only),
            "terminal_answer_removed": True,
            "notes": notes,
        },
    }
    with open(out_dir / "conversion_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def iter_run_rows(workload: Workload, splits: Iterable[str] = ("holdout",)) -> Iterable[dict]:
    wanted = set(splits)
    for row in workload.rows:
        if row["split"] in wanted:
            yield row


def frozen_split_digest(workload: Workload) -> str:
    """Digest of split assignment, used to prove tuning/holdout isolation."""
    return sha256_json({row["request_id"]: row["split"] for row in workload.rows})


def redact_row(row: dict) -> dict:
    """Share-safe version: no prompts, media paths, labels or absolute paths."""
    return {
        "request_id": row["request_id"],
        "group_id": row["group_id"],
        "task": row["task"],
        "split": row["split"],
        "messages_sha256": sha256_json(row["messages"]),
        "media_sha256": [item.get("sha256") for item in row.get("media", [])],
        "input_sha256": row.get("input_sha256"),
    }
