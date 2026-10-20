"""HF backend for the workload advisor.

This is the real measurement path. It is import-safe on CPU-only hosts; torch /
transformers / PIL are imported inside the methods that need them. When the
host has no CUDA device the CLI records a blocker instead of fabricating runs.

Timing boundary (``local_client``): a request is prepared in memory, then the
perf counter starts before preprocessing/encoding/H2D and stops after the
output tokens are committed. Model load is reported separately as cold start.
TTFT/ITL are null because the HF eval path exposes no per-token stream.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any

from .backend import (
    GpuBusy,
    GpuLedger,
    GpuLease,
    capabilities_document,
    check_external_occupancy,
    detect_environment,
    readiness,
    resolve_device_uuid,
    validate_execution_contract,
)
from .records import build_latency, build_raw_counts, build_resource_usage, make_record
from .schema import request_protocol_hash, sha256_file, sha256_json


class BudgetExhausted(RuntimeError):
    pass


class DraftRejected(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _repo_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text


def _json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _json_value(text: str) -> tuple[Any, bool]:
    try:
        return json.loads(text), True
    except Exception:
        return None, False


def _field_f1(prediction: dict | None, reference: dict) -> tuple[float, dict]:
    if prediction is None:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = set(reference)
    matched = 0
    for key in keys:
        if key in prediction and _normalize(prediction[key]) == _normalize(reference[key]):
            matched += 1
    precision = matched / len(prediction) if prediction else 0.0
    recall = matched / len(keys) if keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return f1, {"precision": precision, "recall": recall, "f1": f1}


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _validate_json_schema(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Validate the frozen subset used by the workload advisor's JSON scorer.

    Supports boolean schemas, ``type`` (string or list), ``required``,
    ``properties``, ``additionalProperties``, ``items``, ``enum``, ``const``,
    numeric/string/array bounds and ``allOf``/``anyOf``/``oneOf``.  Unknown
    keywords are ignored; malformed schema declarations are rejected by the
    caller rather than treated as a pass.
    """
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: schema is false"]
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object or boolean"]
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not is_valid_schema_type_list(choices):
            return [f"{path}: schema.type is malformed"]
        if not any(_json_type_matches(value, str(choice)) for choice in choices):
            errors.append(f"{path}: expected type {choices}, got {type(value).__name__}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: value != const")
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        errors.append(f"{path}: value not in enum")
    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in value:
                    errors.extend(_validate_json_schema(value[key], subschema, f"{path}.{key}"))
        additional = schema.get("additionalProperties", True)
        if additional is False and isinstance(properties, dict):
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(additional, dict):
            for key, item in value.items():
                if key not in (properties if isinstance(properties, dict) else {}):
                    errors.extend(_validate_json_schema(item, additional, f"{path}.{key}"))
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, (dict, bool)):
            for index, item in enumerate(value):
                errors.extend(_validate_json_schema(item, items, f"{path}[{index}]"))
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength")
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: below minimum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: above maximum")
    for keyword, mode in (("allOf", "all"), ("anyOf", "any"), ("oneOf", "one")):
        options = schema.get(keyword)
        if not isinstance(options, list) or not options:
            continue
        option_errors = [_validate_json_schema(value, option, path) for option in options]
        passed = sum(1 for item in option_errors if not item)
        if mode == "all" and passed != len(options):
            errors.append(f"{path}: fails {keyword}")
        elif mode == "any" and passed == 0:
            errors.append(f"{path}: fails every {keyword} branch")
        elif mode == "one" and passed != 1:
            errors.append(f"{path}: must satisfy exactly one {keyword} branch")
    return errors


def is_valid_schema_type_list(choices: list) -> bool:
    allowed = {"object", "array", "string", "integer", "number", "boolean", "null"}
    return bool(choices) and all(isinstance(choice, str) and choice in allowed for choice in choices)


def _fallback_checkpoint_gates():
    """Torch-free copies of the eval provenance gates for CPU-only tests.

    The real path imports :mod:`vlm_dflash.eval.run` and reuses those checks.
    This fallback is only used when that module cannot be imported because
    torch is absent; a GPU run cannot reach checkpoint loading in that state.
    """

    def check_head(name: str, meta: dict, target_tied_head: bool) -> None:
        head_mode = meta.get("head", "tied_embedding")
        if not target_tied_head and head_mode != "target_lm_head":
            raise RuntimeError(
                f"{name} was trained against head={head_mode!r} but the target lm_head is untied."
            )

    def check_algorithm(name: str, meta: dict, algorithm) -> None:
        recorded = str(meta.get("algorithm", "dflash"))
        if recorded != algorithm.name:
            raise RuntimeError(
                f"{name} was trained with algorithm={recorded!r}, but this run resolves "
                f"algorithm={algorithm.name!r}."
            )

    def check_feature_mode(name: str, meta: dict) -> str:
        mode = str(meta.get("feature_mode", "offline")).lower()
        if mode not in ("offline", "online"):
            raise RuntimeError(f"{name} records an unknown feature_mode={mode!r}.")
        return mode

    return check_head, check_algorithm, check_feature_mode


def _checkpoint_gates():
    try:
        from vlm_dflash.eval.run import (  # noqa: PLC0415
            check_checkpoint_algorithm,
            check_checkpoint_feature_mode,
            check_checkpoint_head,
        )

        return check_checkpoint_head, check_checkpoint_algorithm, check_checkpoint_feature_mode
    except ImportError as exc:
        if getattr(exc, "name", None) not in (None, "torch"):
            raise
        return _fallback_checkpoint_gates()


def score_output(task: str, output: str, label: dict, scorer: dict | None) -> dict:
    """Deterministic task scorer. Free text without a scorer is not evaluated."""
    if scorer is None:
        return {"evaluated": False, "reason": "no_frozen_scorer", "scorer": None}
    name = str(scorer.get("name") or "")
    threshold = scorer.get("threshold")
    references = label.get("references") if label.get("references") is not None else [label.get("reference")]
    references = [reference for reference in references if reference is not None]
    prediction = _json_object(output)
    parsed_value, parseable = _json_value(output)
    result: dict[str, Any] = {
        "evaluated": True,
        "scorer": {"name": name, "version": scorer.get("version")},
        "scorer_sha256": sha256_json(scorer),
        "task": task,
        "normalization": "nfkc_lower_strip_outer_nonword_whitespace_collapse",
        "json_object_parseable": prediction is not None,
        "json_parseable": parseable,
    }
    if name in ("normalized_exact_match", "exact_match"):
        best = max((1.0 if _normalize(output) == _normalize(reference) else 0.0) for reference in references) if references else 0.0
        result.update({"score": best, "threshold": threshold, "status": "pass" if threshold is None or best >= threshold else "fail"})
    elif name == "field_f1":
        best = 0.0
        details = None
        matched_reference = None
        for reference in references:
            reference_object = reference if isinstance(reference, dict) else _json_object(str(reference))
            if reference_object is None:
                continue
            f1, detail = _field_f1(prediction, reference_object)
            if f1 >= best:
                best, details, matched_reference = f1, detail, reference_object
        result.update(
            {
                "score": best,
                "threshold": threshold,
                "status": "pass" if threshold is None or best >= threshold else "fail",
                "field_metrics": details,
                "reference_sha256": sha256_json(matched_reference) if matched_reference is not None else None,
            }
        )
    elif name == "json_schema_valid":
        schema = scorer.get("schema")
        if not isinstance(schema, (dict, bool)):
            return {
                "evaluated": False,
                "reason": "missing_frozen_schema",
                "scorer": {"name": name, "version": scorer.get("version")},
                "scorer_sha256": sha256_json(scorer),
                "task": task,
            }
        schema_errors = _validate_json_schema(parsed_value, schema) if parseable else ["$: invalid JSON"]
        valid = parseable and not schema_errors
        result.update(
            {
                "score": 1.0 if valid else 0.0,
                "threshold": threshold if threshold is not None else 1.0,
                "status": "pass" if valid else "fail",
                "json_schema_valid": valid,
                "json_schema_errors": schema_errors,
                "schema_sha256": sha256_json(schema),
            }
        )
    elif name == "json_object_syntax":
        valid = prediction is not None
        result.update(
            {
                "score": 1.0 if valid else 0.0,
                "threshold": threshold if threshold is not None else 1.0,
                "status": "pass" if valid else "fail",
                "json_object_valid": valid,
            }
        )
    else:
        return {"evaluated": False, "reason": f"unknown_scorer:{name}", "scorer": {"name": name}}
    return result


class HFWorkloadBackend:
    def __init__(
        self,
        workload,
        *,
        out_dir: str | Path,
        budget_gpu_hours: float = 4.0,
        sessions: int = 3,
        warmups: int = 4,
        device: str = "cuda:0",
        lock_dir: Path | None = None,
        clock=None,
    ):
        self.workload = workload
        self.manifest = workload.manifest
        self.out_dir = Path(out_dir)
        self.budget_gpu_hours = float(budget_gpu_hours)
        if not math.isfinite(self.budget_gpu_hours) or self.budget_gpu_hours <= 0:
            raise ValueError("budget_gpu_hours must be a positive finite number.")
        if int(sessions) < 1:
            raise ValueError("sessions must be >= 1.")
        if int(warmups) < 0:
            raise ValueError("warmups must be >= 0.")
        self.sessions = int(sessions)
        self.warmups = int(warmups)
        self.device = device
        self.lock_dir = lock_dir
        self.clock = clock or time.perf_counter
        self.model = None
        self.processor = None
        self.drafts: dict[str, Any] = {}
        self.draft_algorithms: dict[str, Any] = {}
        self.ledger: GpuLedger | None = None
        self.run_id = f"wa-{int(time.time())}"
        self.execution_order = 0
        self.cold_start: dict[str, float] = {}
        self.reference_tokens: dict[str, list[int] | None] = {}
        self.devices: list[dict] = []
        self.skipped_candidates: list[dict] = []
        self.blocked_candidate_ids: set[str] = set()
        self.protocol_scope_hash: str | None = None
        self.protocol_payload: dict | None = None
        self.resolved_target_identity: dict | None = None
        self.resolved_draft_identities: dict[str, dict] = {}
        self.device_record: dict | None = None
        self.environment: dict | None = None
        self.code_commit = _repo_commit()
        self.phase_intervals: list[dict] = []
        self.audit_handle = None
        self.records_path = self.out_dir / "records.jsonl"
        self.ledger_path = self.out_dir / "ledger.json"
        self._verification = "sequential"
        self._verify_backend = "parity"
        self._active_phase: tuple[str, float] | None = None
        self.occupancy: dict | None = None

    # ------------------------------------------------------------------ setup
    def _torch(self):
        import torch  # noqa: PLC0415

        return torch

    def _resolve_local_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.workload.manifest_dir / path
        return path

    def _processor_spec(self) -> tuple[str, str | None, dict | None, dict | None]:
        target = self.manifest["target"]
        model_path = target.get("model_path") or target.get("model_id")
        processor_cfg = self.manifest.get("processor") or {}
        processor_path = target.get("processor") or processor_cfg.get("path") or model_path
        processor_revision = (
            target.get("processor_revision")
            or processor_cfg.get("revision")
            or target.get("revision")
        )
        vision = target.get("vision") or {}
        resolution = self.manifest.get("resolution") or {}
        min_pixels = vision.get("min_pixels", resolution.get("min_pixels"))
        max_pixels = vision.get("max_pixels", resolution.get("max_pixels"))
        pixel_bounds = None
        if min_pixels is not None or max_pixels is not None:
            pixel_bounds = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        local_processor = self._resolve_local_path(str(processor_path))
        if local_processor is not None and local_processor.exists():
            processor_path = str(local_processor)
        return str(processor_path), processor_revision, processor_cfg, pixel_bounds

    def _load_pretrained_components(
        self,
        *,
        model_path: str,
        model_kwargs: dict,
        processor_path: str,
        processor_kwargs: dict,
    ):
        """Small seam used by CPU spy tests and by the real HF loader."""
        from transformers import AutoProcessor  # noqa: PLC0415

        try:
            from transformers import AutoModelForImageTextToText as AutoModel  # noqa: PLC0415
        except (ImportError, AttributeError):
            from transformers import AutoModelForVision2Seq as AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)
        return model, processor

    def _apply_processor_contract(self, processor) -> None:
        target = self.manifest["target"]
        processor_cfg = self.manifest.get("processor") or {}
        declared = [value for value in (target.get("chat_template"), processor_cfg.get("chat_template")) if value]
        if declared:
            normalized = {str(value) for value in declared}
            if len(normalized) != 1:
                raise DraftRejected(
                    "chat_template_conflict",
                    f"target.processor and processor.chat_template disagree: {sorted(declared)}",
                )
            template = declared[0]
            if "{{" in str(template) or "{%" in str(template):
                processor.chat_template = str(template)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _actual_pixel_bounds(self, processor) -> dict | None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return None
        min_pixels = getattr(image_processor, "min_pixels", None)
        max_pixels = getattr(image_processor, "max_pixels", None)
        if min_pixels is not None and max_pixels is not None:
            return {"min_pixels": int(min_pixels), "max_pixels": int(max_pixels), "source": "image_processor"}
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            shortest = size.get("shortest_edge")
            longest = size.get("longest_edge")
            if shortest is not None and longest is not None:
                return {
                    "min_pixels": int(shortest),
                    "max_pixels": int(longest),
                    "source": "image_processor.size",
                }
        elif size is not None:
            shortest = getattr(size, "shortest_edge", None)
            longest = getattr(size, "longest_edge", None)
            if shortest is not None and longest is not None:
                return {
                    "min_pixels": int(shortest),
                    "max_pixels": int(longest),
                    "source": "image_processor.size",
                }
        return None

    def _weights_digest(self, model_path: str | None) -> str | None:
        path = self._resolve_local_path(model_path)
        if path is None or not path.exists():
            return None
        if path.is_file():
            return sha256_file(path)
        shards = sorted(
            item
            for item in path.iterdir()
            if item.is_file()
            and (
                item.name == "model.safetensors"
                or item.name.startswith("model-")
                and item.name.endswith(".safetensors")
                or item.name.startswith("pytorch_model")
                and item.name.endswith(".bin")
            )
        )
        if not shards:
            return None
        if len(shards) == 1:
            return sha256_file(shards[0])
        return sha256_json({item.name: sha256_file(item) for item in shards})

    def _model_fingerprints(
        self,
        model_path: str | None,
        processor_path: str | None = None,
    ) -> tuple[str | None, str | None]:
        path = self._resolve_local_path(model_path)
        if path is None or not path.is_dir():
            return None, None
        processor_dir = self._resolve_local_path(processor_path) if processor_path else None
        if processor_dir is None or not processor_dir.is_dir():
            processor_dir = path
        try:
            from vlm_dflash.contract import (  # noqa: PLC0415
                _MODEL_FILES,
                _PROCESSOR_FILES,
                _file_fingerprint,
                _weight_shard_signature,
            )

            model_fingerprint = sha256_json(
                {
                    "files": _file_fingerprint(path, _MODEL_FILES),
                    "weights": _weight_shard_signature(path),
                }
            )
            processor_fingerprint = sha256_json(
                {"files": _file_fingerprint(processor_dir, _PROCESSOR_FILES)}
            )
            return model_fingerprint, processor_fingerprint
        except Exception:
            return None, None

    def _resolve_target_identity(self, model, processor, model_path: str) -> dict:
        target = self.manifest["target"]
        processor_path, processor_revision, _processor_cfg, pixel_bounds = self._processor_spec()
        weights_sha256 = self._weights_digest(model_path)
        model_fingerprint, processor_fingerprint = self._model_fingerprints(model_path, processor_path)
        declared_weights = target.get("weights_sha256")
        if declared_weights:
            if weights_sha256 is None:
                raise DraftRejected(
                    "weights_hash_unverifiable",
                    f"target.weights_sha256 is declared but no local weight file exists under {model_path!r}",
                )
            if weights_sha256 != declared_weights:
                raise DraftRejected(
                    "weights_hash_mismatch",
                    f"target weights sha256 {weights_sha256} != declared {declared_weights}",
                )
        declared_model_fp = target.get("model_fingerprint")
        if declared_model_fp:
            if model_fingerprint is None:
                raise DraftRejected(
                    "model_fingerprint_unverifiable",
                    "target.model_fingerprint is declared but the local model artifacts could not be fingerprinted",
                )
            if declared_model_fp != model_fingerprint:
                raise DraftRejected(
                    "model_fingerprint_mismatch",
                    f"resolved model fingerprint {model_fingerprint} != declared {declared_model_fp}",
                )
        declared_processor_fp = target.get("processor_fingerprint")
        if declared_processor_fp:
            if processor_fingerprint is None:
                raise DraftRejected(
                    "processor_fingerprint_unverifiable",
                    "target.processor_fingerprint is declared but the local processor artifacts could not be fingerprinted",
                )
            if declared_processor_fp != processor_fingerprint:
                raise DraftRejected(
                    "processor_fingerprint_mismatch",
                    f"resolved processor fingerprint {processor_fingerprint} != declared {declared_processor_fp}",
                )
        chat_template = getattr(processor, "chat_template", None)
        template_sha256 = sha256_json(chat_template) if chat_template is not None else None
        declared_template_hash = target.get("chat_template_sha256") or target.get("template_sha256")
        if declared_template_hash:
            if template_sha256 != declared_template_hash:
                raise DraftRejected(
                    "chat_template_hash_mismatch",
                    f"processor chat template sha256 {template_sha256} != declared {declared_template_hash}",
                )
        actual_pixels = self._actual_pixel_bounds(processor)
        if pixel_bounds:
            for key in ("min_pixels", "max_pixels"):
                declared = pixel_bounds.get(key)
                actual = (actual_pixels or {}).get(key)
                if declared is not None and actual is not None and int(declared) != int(actual):
                    raise DraftRejected(
                        "pixel_bounds_not_applied",
                        f"declared {key}={declared} but loaded processor reports {actual}",
                    )
        config = getattr(model, "config", None)
        structure = target.get("structure") or getattr(config, "model_type", None)
        if config is not None and getattr(config, "model_type", None) and target.get("structure"):
            if str(config.model_type) != str(target.get("structure")):
                raise DraftRejected(
                    "target_structure_mismatch",
                    f"target.structure={target.get('structure')!r} but loaded model_type={config.model_type!r}",
                )
        actual_revision = (
            getattr(config, "_commit_hash", None)
            or getattr(model, "_commit_hash", None)
            or target.get("revision")
        )
        resolved = {
            "source": target.get("source", "hf"),
            "model_id": target.get("model_id"),
            "model_path": target.get("model_path") or target.get("model_id"),
            "revision": target.get("revision"),
            "resolved_revision": actual_revision,
            "weights_sha256": weights_sha256,
            "declared_weights_sha256": declared_weights,
            "model_fingerprint": model_fingerprint or target.get("model_fingerprint"),
            "processor_fingerprint": processor_fingerprint or target.get("processor_fingerprint"),
            "tokenizer": target.get("tokenizer", target.get("model_id")),
            "processor": target.get("processor", target.get("processor_revision")),
            "processor_revision": processor_revision,
            "chat_template": target.get("chat_template") or (self.manifest.get("processor") or {}).get("chat_template"),
            "template_sha256": template_sha256,
            "dtype": target.get("dtype"),
            "structure": structure,
            "vision": actual_pixels or pixel_bounds,
            "attention_backend": target.get("attn_implementation", "sdpa"),
        }
        return {key: value for key, value in resolved.items() if value is not None}

    def _preflight_target_artifacts(self) -> None:
        """Check cheap declared hashes before loading weights onto a GPU."""
        target = self.manifest["target"]
        model_path = target.get("model_path") or target.get("model_id")
        processor_path, _revision, _cfg, _pixels = self._processor_spec()
        declared_weights = target.get("weights_sha256")
        if declared_weights:
            actual_weights = self._weights_digest(str(model_path))
            if actual_weights is None:
                raise DraftRejected(
                    "weights_hash_unverifiable",
                    f"target.weights_sha256 is declared but no local weight file exists under {model_path!r}",
                )
            if actual_weights != declared_weights:
                raise DraftRejected(
                    "weights_hash_mismatch",
                    f"target weights sha256 {actual_weights} != declared {declared_weights}",
                )
        declared_model_fp = target.get("model_fingerprint")
        declared_processor_fp = target.get("processor_fingerprint")
        if declared_model_fp or declared_processor_fp:
            model_fp, processor_fp = self._model_fingerprints(str(model_path), processor_path)
            if declared_model_fp and model_fp != declared_model_fp:
                raise DraftRejected(
                    "model_fingerprint_mismatch",
                    f"resolved model fingerprint {model_fp} != declared {declared_model_fp}",
                )
            if declared_processor_fp and processor_fp != declared_processor_fp:
                raise DraftRejected(
                    "processor_fingerprint_mismatch",
                    f"resolved processor fingerprint {processor_fp} != declared {declared_processor_fp}",
                )

    def _load_model(self):
        target = self.manifest["target"]
        model_path = target.get("model_path") or target.get("model_id")
        if target.get("model_path"):
            model_path = str(self._resolve_local_path(target["model_path"]))
        local_only = bool(target.get("local_files_only", True))
        dtype_name = str(target.get("dtype", "bfloat16"))
        dtype = getattr(self._torch(), dtype_name)
        kwargs = {
            "dtype": dtype,
            "attn_implementation": target.get("attn_implementation", "sdpa"),
            "local_files_only": local_only,
        }
        if target.get("revision"):
            kwargs["revision"] = target.get("revision")
        processor_path, processor_revision, _processor_cfg, pixel_bounds = self._processor_spec()
        processor_kwargs = {"local_files_only": local_only}
        if processor_revision:
            processor_kwargs["revision"] = processor_revision
        if pixel_bounds:
            if pixel_bounds.get("min_pixels") is not None:
                processor_kwargs["min_pixels"] = int(pixel_bounds["min_pixels"])
            if pixel_bounds.get("max_pixels") is not None:
                processor_kwargs["max_pixels"] = int(pixel_bounds["max_pixels"])
        model, processor = self._load_pretrained_components(
            model_path=str(model_path),
            model_kwargs=kwargs,
            processor_path=str(processor_path),
            processor_kwargs=processor_kwargs,
        )
        model.to(self.device)
        model.eval()
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        self._apply_processor_contract(processor)
        self.model = model
        self.processor = processor
        self.resolved_target_identity = self._resolve_target_identity(model, processor, str(model_path))

    def _target_head_is_tied(self) -> bool:
        model = self.model
        try:
            embeddings = model.get_input_embeddings()
            weight = getattr(getattr(model, "lm_head", None), "weight", None)
            return weight is not None and embeddings is not None and weight.data_ptr() == embeddings.weight.data_ptr()
        except Exception:
            config = getattr(model, "config", None)
            return bool(getattr(config, "tie_word_embeddings", True))

    def _expected_checkpoint_identity(self) -> dict:
        target = self.manifest.get("target") or {}
        expected = dict(target.get("identity") or {})
        for key in (
            "model_fingerprint",
            "processor_fingerprint",
            "mask_token_id",
            "spatial_rope_sections",
            "capture_layer_ids",
            "draft_layer_ids",
            "image_min_pixels",
            "image_max_pixels",
        ):
            if key in target:
                expected[key] = target[key]
        if self.resolved_target_identity:
            if self.resolved_target_identity.get("model_fingerprint"):
                expected.setdefault("model_fingerprint", self.resolved_target_identity["model_fingerprint"])
            if self.resolved_target_identity.get("processor_fingerprint"):
                expected.setdefault("processor_fingerprint", self.resolved_target_identity["processor_fingerprint"])
            vision = self.resolved_target_identity.get("vision") or {}
            if vision.get("min_pixels") is not None:
                expected.setdefault("image_min_pixels", vision["min_pixels"])
            if vision.get("max_pixels") is not None:
                expected.setdefault("image_max_pixels", vision["max_pixels"])
        return expected

    def _validate_checkpoint_metadata(self, candidate: dict, algorithm, meta: dict, checkpoint_sha256: str) -> dict:
        check_checkpoint_head, check_checkpoint_algorithm, check_checkpoint_feature_mode = _checkpoint_gates()

        if not isinstance(meta, dict):
            raise DraftRejected("checkpoint_metadata_missing", "load_decoder did not return checkpoint metadata")
        checkpoint_name = str(candidate.get("candidate_id"))
        check_checkpoint_algorithm(checkpoint_name, meta, algorithm)
        feature_mode = check_checkpoint_feature_mode(checkpoint_name, meta)
        check_checkpoint_head(checkpoint_name, meta, self._target_head_is_tied())
        declared = dict(candidate.get("draft_identity") or {})
        actual_identity = meta.get("identity")
        if actual_identity is not None and not isinstance(actual_identity, dict):
            raise DraftRejected("checkpoint_identity_malformed", "checkpoint identity metadata must be an object")
        actual_identity = actual_identity or {}
        expected_identity = self._expected_checkpoint_identity()
        if actual_identity and expected_identity:
            from vlm_dflash.provenance import assert_identity  # noqa: PLC0415

            projected_actual = {key: actual_identity.get(key) for key in expected_identity}
            try:
                assert_identity(
                    expected_identity,
                    projected_actual,
                    f"workload advisor checkpoint {checkpoint_name}",
                )
            except Exception as exc:  # noqa: BLE001
                raise DraftRejected("checkpoint_target_identity_mismatch", str(exc)) from exc
        metadata_revision = (
            meta.get("revision")
            or meta.get("checkpoint_revision")
            or actual_identity.get("revision")
            or (meta.get("draft_config") or {}).get("revision")
            or (meta.get("draft_config") or {}).get("_commit_hash")
        )
        declared_revision = declared.get("revision")
        if declared_revision and metadata_revision and str(declared_revision) != str(metadata_revision):
            raise DraftRejected(
                "checkpoint_revision_mismatch",
                f"declared draft revision {declared_revision!r} != checkpoint metadata {metadata_revision!r}",
            )
        metadata_source = meta.get("source") or actual_identity.get("source")
        declared_source = declared.get("source")
        if declared_source and metadata_source and str(declared_source) != str(metadata_source):
            raise DraftRejected(
                "checkpoint_source_mismatch",
                f"declared draft source {declared_source!r} != checkpoint metadata {metadata_source!r}",
            )
        for key, expected in declared.items():
            if key in ("checkpoint", "checkpoint_sha256", "weights_sha256", "artifact_sha256"):
                continue
            if key in ("revision", "source", "algorithm"):
                continue
            metadata_value = meta.get(key) or actual_identity.get(key)
            if metadata_value is not None and expected != metadata_value:
                raise DraftRejected(
                    f"checkpoint_{key}_mismatch",
                    f"declared draft_identity.{key}={expected!r} != checkpoint metadata {metadata_value!r}",
                )
        draft_config = meta.get("draft_config") or {}
        metadata_block_size = meta.get("block_size") or (
            draft_config.get("block_size") if isinstance(draft_config, dict) else None
        )
        declared_block_size = declared.get("block_size") or candidate.get("block_size")
        if declared_block_size is not None and metadata_block_size is not None:
            if int(declared_block_size) != int(metadata_block_size):
                raise DraftRejected(
                    "checkpoint_block_size_mismatch",
                    f"declared block_size={declared_block_size} != checkpoint metadata {metadata_block_size}",
                )
        expected_layers = expected_identity.get("draft_layer_ids") or expected_identity.get("capture_layer_ids")
        metadata_layers = meta.get("target_layer_ids")
        if expected_layers is not None and metadata_layers is not None:
            if [int(layer) for layer in expected_layers] != [int(layer) for layer in metadata_layers]:
                raise DraftRejected(
                    "checkpoint_target_layers_mismatch",
                    f"checkpoint target_layer_ids={metadata_layers} != frozen {expected_layers}",
                )
        declared_hashes = [
            ("draft_identity.checkpoint_sha256", declared.get("checkpoint_sha256")),
            ("draft_identity.weights_sha256", declared.get("weights_sha256")),
            ("draft_identity.artifact_sha256", declared.get("artifact_sha256")),
            ("candidate.checkpoint_sha256", candidate.get("checkpoint_sha256")),
        ]
        for path, expected in declared_hashes:
            if expected is None:
                continue
            if expected != checkpoint_sha256:
                raise DraftRejected(
                    "checkpoint_hash_mismatch",
                    f"{path}={expected} != resolved artifact sha256 {checkpoint_sha256}",
                )
        resolved = {
            **declared,
            "algorithm": algorithm.name,
            "checkpoint": str(candidate.get("checkpoint")),
            "checkpoint_sha256": checkpoint_sha256,
            "block_size": metadata_block_size if metadata_block_size is not None else declared_block_size,
            "head": meta.get("head", "tied_embedding"),
            "feature_mode": feature_mode,
            "revision": metadata_revision or declared_revision,
            "revision_verified": bool(metadata_revision),
            "source": metadata_source or declared_source,
            "checkpoint_identity": actual_identity or None,
            "target_layer_ids": metadata_layers,
            "train_recipe": meta.get("train_recipe"),
        }
        return {key: value for key, value in resolved.items() if value is not None}

    def _load_drafts(self) -> list[dict]:
        from vlm_dflash.algorithms import get_algorithm  # noqa: PLC0415

        errors: list[dict] = []
        for candidate in self.manifest.get("draft_candidates") or []:
            if candidate.get("method") != "sd":
                continue
            if candidate.get("candidate_id") in self.blocked_candidate_ids:
                continue
            started = time.perf_counter()
            candidate_id = str(candidate.get("candidate_id"))
            try:
                self._check_budget("draft_load")
                checkpoint = self._resolve_local_path(str(candidate.get("checkpoint")))
                if checkpoint is None or not checkpoint.is_file():
                    raise DraftRejected(
                        "checkpoint_missing",
                        f"checkpoint not found: {candidate.get('checkpoint')!r}",
                    )
                checkpoint_sha256 = sha256_file(checkpoint)
                algorithm = get_algorithm(candidate.get("algorithm"))
                draft, meta = algorithm.load_decoder(checkpoint)
                resolved = self._validate_checkpoint_metadata(candidate, algorithm, meta, checkpoint_sha256)
                draft.to(
                    device=self.device,
                    dtype=getattr(self._torch(), str(self.manifest["target"].get("dtype", "bfloat16"))),
                )
                draft.eval()
                self.drafts[candidate_id] = draft
                self.draft_algorithms[candidate_id] = algorithm
                self.resolved_draft_identities[candidate_id] = resolved
                self.cold_start[candidate_id] = time.perf_counter() - started
            except BudgetExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - candidate isolation is the point
                code = exc.code if isinstance(exc, DraftRejected) else "checkpoint_rejected"
                error = {
                    "type": type(exc).__name__,
                    "code": code,
                    "stage": "draft_load",
                    "candidate_id": candidate_id,
                    "message": str(exc),
                }
                errors.append(error)
                self.skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": code,
                        "message": str(exc),
                    }
                )
        return errors

    def _load_media(self, row: dict) -> list:
        from PIL import Image  # noqa: PLC0415

        images = []
        for item in row.get("media", []):
            path = (self.workload.media_root / item["path"]).resolve()
            images.append(Image.open(path).convert("RGB"))
        return images

    def _prepare(self, row: dict, images: list) -> dict:
        from vlm_dflash.target.prompts import encode_conversation  # noqa: PLC0415

        messages = []
        attached = False
        for message in row["messages"]:
            if message["role"] == "user" and images and not attached:
                parts = [{"type": "image"} for _ in images]
                parts.append({"type": "text", "text": message["content"]})
                messages.append({"role": "user", "content": parts})
                attached = True
            else:
                messages.append({"role": message["role"], "content": message["content"]})
        inputs = encode_conversation(self.processor, messages, images, add_generation_prompt=True)
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _target_visual_tokens(self, inputs: dict) -> int | None:
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return None
        merge = getattr(getattr(self.processor, "image_processor", None), "merge_size", 2) or 2
        try:
            return int((grid.prod(dim=1) / (merge * merge)).sum().item())
        except Exception:
            return None

    # ------------------------------------------------------------------- runs
    def _sync_device(self) -> None:
        self._torch().cuda.synchronize(self.device)

    def _run_ar(self, inputs: dict, max_new_tokens: int, stop_ids: list[int]):
        torch = self._torch()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=stop_ids[0] if len(stop_ids) == 1 else stop_ids,
            )
        return generated

    def _run_sd(self, candidate_id: str, inputs: dict, max_new_tokens: int, stop_ids: list[int]):
        torch = self._torch()
        draft = self.drafts[candidate_id]
        with torch.inference_mode():
            output_ids, stats = draft.spec_generate(
                target=self.model,
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                stop_token_ids=sorted(stop_ids),
                temperature=0.0,
                target_kwargs={key: value for key, value in inputs.items() if key != "input_ids"},
                return_stats=True,
                sequential_target_verification=True,
            )
        return output_ids, dict(stats)

    def _decode_tokens(self, generated, inputs: dict) -> list[int]:
        prompt_len = int(inputs["input_ids"].shape[1])
        return [int(token) for token in generated[0, prompt_len:].tolist()]

    def _execute_request(
        self,
        row: dict,
        candidate: dict,
        max_new_tokens: int,
        stop_ids: list[int],
    ) -> dict:
        """Run one request with the declared local-client timing boundary.

        Media disk read is setup.  The e2e timer starts only after the media is
        resident and the device is idle; it includes chat rendering, processor
        work, H2D copies, generation, synchronization of the actual device and
        the token commit (``.tolist()``).  ``generate_only_s`` starts after
        preprocessing and also includes the device synchronization.
        """
        torch = self._torch()
        outcome: dict[str, Any] = {
            "tokens": [],
            "stats": None,
            "inputs": None,
            "error": None,
            "timings": {},
            "v_tokens": None,
        }
        media_started = self.clock()
        try:
            images = self._load_media(row)
            outcome["timings"]["media_load_s"] = max(0.0, self.clock() - media_started)
            self._sync_device()
        except Exception as exc:  # noqa: BLE001
            outcome["timings"]["media_load_s"] = max(0.0, self.clock() - media_started)
            outcome["error"] = {
                "type": type(exc).__name__,
                "stage": "media_load_or_device_sync",
                "message": str(exc),
            }
            return outcome

        started = self.clock()
        generate_started = None
        try:
            inputs = self._prepare(row, images)
            outcome["inputs"] = inputs
            outcome["timings"]["preprocess_s"] = max(0.0, self.clock() - started)
            generate_started = self.clock()
            if candidate["method"] == "ar":
                generated = self._run_ar(inputs, max_new_tokens, stop_ids)
                stats = None
            else:
                generated, stats = self._run_sd(candidate["candidate_id"], inputs, max_new_tokens, stop_ids)
            self._sync_device()
            generate_finished = self.clock()
            outcome["timings"]["generate_only_s"] = max(0.0, generate_finished - generate_started)
            tokens = self._decode_tokens(generated, inputs)
            outcome["timings"]["commit_s"] = max(0.0, self.clock() - generate_finished)
            outcome["tokens"] = tokens
            outcome["stats"] = stats
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - GPU-only
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            outcome["error"] = {"type": "oom", "stage": "run", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = {"type": type(exc).__name__, "stage": "run", "message": str(exc)}
        finally:
            elapsed = max(0.0, self.clock() - started)
            if outcome["timings"].get("e2e_s") is None:
                outcome["timings"]["e2e_s"] = elapsed
            if generate_started is not None and "generate_only_s" not in outcome["timings"]:
                outcome["timings"]["generate_only_s"] = max(0.0, self.clock() - generate_started)
        try:
            outcome["v_tokens"] = self._target_visual_tokens(outcome["inputs"]) if outcome["inputs"] else None
        except Exception:
            outcome["v_tokens"] = None
        return outcome

    def _peak_memory(self) -> dict:
        torch = self._torch()
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def _reset_peak(self) -> None:
        self._torch().cuda.reset_peak_memory_stats(self.device)

    def _resource(self, fit: bool | None = None, purpose: str = "run") -> dict:
        devices = self.devices[:1]
        peaks = self._peak_memory() if self.model is not None else {}
        device_rows = [
            {
                "uuid": device["uuid"],
                "name": device.get("name"),
                "peak_allocated_bytes": peaks.get("peak_allocated_bytes"),
                "peak_reserved_bytes": peaks.get("peak_reserved_bytes"),
                "peak_scope": "request_includes_resident_model_weights",
            }
            for device in devices
        ]
        ledger = self.ledger
        if ledger is None:
            return build_resource_usage(
                all_allocated_gpu_hours=None,
                measured_run_gpu_hours=None,
                devices=device_rows,
                gpu_memory_fit=fit,
                note="no ledger (dry-run)",
            )
        return build_resource_usage(
            all_allocated_gpu_hours=ledger.all_allocated_gpu_hours,
            measured_run_gpu_hours=ledger.measured_run_gpu_hours,
            devices=device_rows,
            allocation_intervals=ledger.intervals,
            gpu_memory_fit=fit,
            note="all_allocated_gpu_hours includes load, warmup, failures and idle allocation",
        )

    def _quality(self, row: dict, output_tokens: list[int]) -> dict:
        label = self.workload.labels.get(row["request_id"])
        if label is None:
            return {"evaluated": False, "reason": "no_label_for_request", "scorer": None}
        scorer = (self.manifest.get("scorers") or {}).get(row["task"])
        try:
            if self.processor is None:
                raise RuntimeError("processor is not loaded")
            text = self.processor.batch_decode([output_tokens], skip_special_tokens=True)[0]
            return score_output(row["task"], text, label, scorer)
        except Exception as exc:  # noqa: BLE001 - quality errors must not lose the measured row
            return {
                "evaluated": False,
                "reason": "scorer_error",
                "scorer": {"name": (scorer or {}).get("name"), "version": (scorer or {}).get("version")},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    def _exact(self, candidate_tokens: list[int], reference_tokens: list[int], stop_ids: set[int]) -> bool:
        from vlm_dflash.eval.equivalence import sequences_equal  # noqa: PLC0415

        return sequences_equal(candidate_tokens, reference_tokens, stop_ids)

    def _base_identity(self) -> tuple[dict, dict]:
        env = self.environment or detect_environment()
        target = self.manifest["target"]
        fallback_target = {
            "source": target.get("source", "hf"),
            "model_id": target.get("model_id"),
            "model_path": target.get("model_path"),
            "revision": target.get("revision"),
            "weights_sha256": target.get("weights_sha256"),
            "tokenizer": target.get("tokenizer", target.get("model_id")),
            "processor": target.get("processor", target.get("processor_revision")),
            "chat_template": target.get("chat_template"),
            "dtype": target.get("dtype"),
            "structure": target.get("structure", "qwen2_5_vl"),
            "vision": target.get("vision"),
        }
        target_identity = dict(self.resolved_target_identity or {})
        for key, value in fallback_target.items():
            target_identity.setdefault(key, value)
        backend_identity = {
            "name": (self.manifest.get("backend") or {}).get("name", "hf-transformers"),
            "version": (env.get("transformers") or {}).get("version"),
            "torch_version": (env.get("torch") or {}).get("version"),
            "code_commit": self.code_commit,
            "device_topology": {
                "devices": self.devices[:1],
                "concurrency": self.manifest.get("concurrency"),
            },
            "verification": self._verification,
            "verify_backend": self._verify_backend,
            "implementation": "hf-transformers",
        }
        return target_identity, backend_identity

    def _candidate_identity(self, candidate: dict) -> dict | None:
        if candidate.get("method") == "ar":
            return None
        identity = dict(self.resolved_draft_identities.get(candidate["candidate_id"]) or candidate.get("draft_identity") or {})
        identity.setdefault("algorithm", candidate.get("algorithm"))
        identity.setdefault("checkpoint", candidate.get("checkpoint"))
        identity.setdefault("block_size", candidate.get("block_size"))
        return identity

    def _record(
        self,
        *,
        row: dict,
        candidate: dict,
        session_id: str,
        repeat_index: int,
        tokens: list[int],
        wall_s: float,
        stats: dict | None,
        reference: list[int] | None,
        stop_ids: set[int],
        v_tokens: int | None,
        error: dict | None,
        timings: dict | None = None,
    ) -> dict:
        target_identity, backend_identity = self._base_identity()
        method = candidate["method"]
        accepted = int(stats.get("accepted_tokens", 0)) if stats else None
        proposed = int(stats.get("draft_tokens", 0)) if stats else None
        rounds = int(stats.get("verify_steps", 0)) if stats else None
        counts = build_raw_counts(
            emitted_tokens=len(tokens),
            proposed_draft_tokens=proposed,
            accepted_draft_tokens=accepted,
            verify_rounds=rounds,
            target_forward_calls=int(stats.get("target_forward_calls", 0)) if stats else None,
            draft_forward_calls=None,
            method=method,
        )
        e2e_s = (timings or {}).get("e2e_s") if timings else wall_s
        if e2e_s is not None and (not isinstance(e2e_s, (int, float)) or not math.isfinite(float(e2e_s))):
            e2e_s = None
        latency, reasons = build_latency(e2e_s=e2e_s, n_samples=1)
        if e2e_s is None and error is not None:
            reasons["latency.e2e_s"] = f"{error.get('stage', 'run')}_failed"
        if timings:
            for key in ("media_load_s", "preprocess_s", "generate_only_s", "commit_s"):
                value = timings.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0:
                    latency[key] = float(value)
        if method == "ar":
            reasons["raw_counts.target_forward_calls"] = "generate_call_count_not_exposed"
            reasons["raw_counts.draft_forward_calls"] = "not_applicable_ar"
        else:
            reasons["raw_counts.draft_forward_calls"] = "spec_generate_batched_calls_not_exposed"
        reasons["visual_tokens.draft_V"] = "draft_visual_tokens_not_exposed"
        correctness = None
        status = "not_checked"
        if reference is not None and error is None:
            exact = self._exact(tokens, reference, stop_ids)
            correctness = {
                "reference_hash": sha256_json(reference),
                "candidate_hash": sha256_json(tokens),
                "input_sha256": row.get("input_sha256"),
                "n_requests": 1,
                "failed_request_ids": [] if exact else [row["request_id"]],
                "compared_tokens": min(len(tokens), len(reference)),
                "protocol_hash": request_protocol_hash(self.protocol_payload or {}, self.workload.request_identity(row)),
            }
            status = "verified_greedy_exact" if exact else "failed"
            if not exact:
                error = {"type": "token_mismatch", "stage": "correctness", "message": "candidate tokens differ from AR reference"}
        if error is not None:
            status = "failed"
        record = make_record(
            run_id=self.run_id,
            request_id=row["request_id"],
            group_id=row["group_id"],
            task=row["task"],
            split=row["split"],
            method=method,
            candidate_id=candidate["candidate_id"],
            session_id=session_id,
            repeat_index=repeat_index,
            execution_order=self.execution_order,
            target_identity=target_identity,
            draft_identity=self._candidate_identity(candidate),
            backend_identity=backend_identity,
            protocol_hash=request_protocol_hash(self.protocol_payload or {}, self.workload.request_identity(row)),
            protocol_scope_hash=self.protocol_scope_hash,
            correctness_status=status,
            acceptance_policy="strict_greedy_token_exact" if method == "sd" else "not_applicable",
            correctness=correctness,
            quality=self._quality(row, tokens) if tokens else {"evaluated": False, "reason": "no_output_tokens"},
            measurement_scope="local_client",
            raw_counts=counts,
            latency=latency,
            resource_usage=self._resource(fit=error is None),
            output_tokens_R=len(tokens),
            visual_tokens={"target_V": v_tokens, "draft_V": None},
            error=error,
            missing_reasons=reasons,
            measurement={
                "timing_boundary": "request_ready_to_committed_output",
                "includes": ["chat_template", "processor", "encode", "h2d", "generate", "device_sync", "token_commit"],
                "excludes": ["media_disk_read", "model_load", "warmup"],
                "device": self.device,
                "generate_only_s": latency.get("generate_only_s"),
                "media_load_s": latency.get("media_load_s"),
            },
            artifacts={
                "candidate_token_ids": tokens,
                "reference_token_ids": reference,
                "candidate_token_ids_sha256": sha256_json(tokens),
                "reference_token_ids_sha256": sha256_json(reference) if reference is not None else None,
            },
            extra={"concurrency": self.manifest.get("concurrency"), "counts_verified": True},
        )
        return record

    # --------------------------------------------------------------- orchestrate
    def _mark_phase(self, purpose: str, started_at: float) -> None:
        ended_at = time.time()
        interval = {
            "phase": purpose,
            "start_unix_s": started_at,
            "end_unix_s": ended_at,
            "duration_s": max(0.0, ended_at - started_at),
        }
        self.phase_intervals.append(interval)
        if self.ledger is not None:
            self.ledger.record_interval(purpose, start_unix_s=started_at, end_unix_s=ended_at)

    def _check_budget(self, phase: str) -> dict:
        if self.ledger is None:
            return {"exceeded": False}
        budget = self.ledger.check_budget()
        if budget["exceeded"]:
            raise BudgetExhausted(
                f"GPU budget {self.budget_gpu_hours} h reached before {phase} "
                f"({budget['all_allocated_gpu_hours']:.3f} h allocated)."
            )
        return budget

    def _ledger_payload(self, *, status: str, records: int = 0) -> dict:
        if self.ledger is None:
            return {
                "schema_version": "veladraft-platform-v1",
                "status": status,
                "run_id": self.run_id,
                "budget_gpu_hours": self.budget_gpu_hours,
                "all_allocated_gpu_hours": 0.0,
                "measured_run_gpu_hours": 0.0,
                "uuids": [],
                "records": records,
                "phases": self.phase_intervals,
                "intervals": [],
            }
        payload = self.ledger.checkpoint()
        payload.update(
            {
                "status": status,
                "run_id": self.run_id,
                "records": records,
                "phases": self.phase_intervals,
                "device": self.device,
                "physical_uuid": (self.device_record or {}).get("uuid"),
                "occupancy": self.occupancy,
            }
        )
        return payload

    def _write_ledger_checkpoint(self, *, status: str, records: int) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        payload = self._ledger_payload(status=status, records=records)
        temp_path = self.ledger_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.ledger_path)

    def _open_audit(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audit_handle = open(self.records_path, "w", encoding="utf-8")
        self._write_ledger_checkpoint(status="running", records=0)

    def _append_record(self, record: dict, records_so_far: int) -> None:
        if self.audit_handle is None:
            return
        self.audit_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.audit_handle.flush()
        os.fsync(self.audit_handle.fileno())
        self._write_ledger_checkpoint(status="running", records=records_so_far)

    def _close_audit(self) -> None:
        if self.audit_handle is not None:
            self.audit_handle.close()
            self.audit_handle = None

    def _blocked_result(self, *, reason: str, ready: dict, errors: list[dict] | None = None) -> dict:
        return {
            "status": "blocked",
            "stop_reason": reason,
            "readiness": ready,
            "records": [],
            "errors": list(errors or []),
            "skipped_candidates": self.skipped_candidates,
            "cold_start": self.cold_start,
            "ledger": self._ledger_payload(status="blocked"),
            "occupancy": self.occupancy,
            "capabilities": capabilities_document(state="blocked"),
        }

    def _aggregate_status(self, records: list[dict], errors: list[dict], stop_reason: str) -> str:
        if not records:
            return "failed"
        record_errors = [record for record in records if record.get("error")]
        quality_failures = [
            record for record in records if (record.get("quality") or {}).get("status") == "fail"
        ]
        if len(record_errors) == len(records):
            return "failed"
        if record_errors or quality_failures or errors or self.skipped_candidates:
            return "partial"
        return "ok"

    def run(self, *, rows: list[dict] | None = None, max_requests: int | None = None) -> dict:
        ready = readiness(self.manifest, manifest_dir=self.workload.manifest_dir)
        if ready["status"] == "blocked":
            return self._blocked_result(reason="backend_not_ready", ready=ready)
        rows = list(rows or [row for row in self.workload.rows if row["split"] == "holdout"])
        if max_requests is not None:
            rows = rows[:max_requests]
        if not rows:
            raise ValueError("no holdout rows to run")
        environment = ready.get("environment") or {}
        self.environment = environment
        try:
            self.device_record = resolve_device_uuid(
                self.device,
                environment.get("devices") or [],
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        except ValueError as exc:
            error = {"type": "device_resolution_error", "message": str(exc)}
            return self._blocked_result(reason="device_resolution_error", ready=ready, errors=[error])
        self.devices = [self.device_record]
        self.device = str(self.device_record.get("logical_device") or self.device)
        self.occupancy = check_external_occupancy(str(self.device_record["uuid"]))
        if self.occupancy.get("busy"):
            error = {
                "type": "GpuBusy",
                "message": f"external process(es) are using {self.device_record['uuid']}",
                "processes": self.occupancy.get("processes") or [],
            }
            return self._blocked_result(reason="gpu_busy", ready=ready, errors=[error])
        lease: GpuLease | None = None
        records: list[dict] = []
        errors: list[dict] = []
        stop_reason = "completed"
        execution_stage = "run"
        status = "ok"
        cleanup_needed = False
        self.blocked_candidate_ids = {
            item.get("candidate_id") for item in ready.get("candidate_blockers") or []
        }
        try:
            lease = GpuLease(
                [self.device_record["uuid"]],
                **({"lock_dir": self.lock_dir} if self.lock_dir else {}),
            ).acquire()
            self.ledger = GpuLedger(self.budget_gpu_hours, lease)
            post_lease_occupancy = check_external_occupancy(str(self.device_record["uuid"]))
            if self.occupancy is not None:
                self.occupancy = {**self.occupancy, "post_lease": post_lease_occupancy}
            else:
                self.occupancy = {"post_lease": post_lease_occupancy}
            if post_lease_occupancy.get("busy"):
                raise GpuBusy(
                    str(self.device_record["uuid"]),
                    lock_dir=self.lock_dir,
                    detail="external process appeared after the advisory lock was acquired",
                )
            self._open_audit()
            self._check_budget("pre_load")
            phase_started = time.time()
            self._torch().cuda.reset_peak_memory_stats(self.device)
            started = self.clock()
            execution_stage = "target_preflight"
            self._preflight_target_artifacts()
            execution_stage = "model_load"
            cleanup_needed = True
            self._load_model()
            self.cold_start["target"] = self.clock() - started
            self._mark_phase("model_load", phase_started)
            self._check_budget("post_model_load")
            phase_started = time.time()
            execution_stage = "draft_load"
            draft_errors = self._load_drafts()
            errors.extend(draft_errors)
            self._mark_phase("draft_load", phase_started)
            execution_stage = "protocol"
            self._verification = str(self.manifest.get("decode", {}).get("verification", "sequential"))
            self._verify_backend = str(self.manifest.get("decode", {}).get("verify_backend", "parity"))
            self._build_protocol()
            blocked_ids = set(self.blocked_candidate_ids)
            candidates = []
            for candidate in self.manifest.get("draft_candidates") or []:
                if candidate.get("candidate_id") in blocked_ids:
                    self.skipped_candidates.append(
                        {"candidate_id": candidate.get("candidate_id"), "reason": "backend_not_ready"}
                    )
                else:
                    candidates.append(candidate)
            if not candidates:
                status = "blocked"
                stop_reason = "no_ready_candidates"
                return self._blocked_result(reason=stop_reason, ready=ready, errors=errors)
            try:
                ar_candidate = next(candidate for candidate in candidates if candidate["method"] == "ar")
            except StopIteration as exc:
                raise RuntimeError("manifest has no AR candidate after readiness filtering") from exc
            sd_candidates = [
                candidate
                for candidate in candidates
                if candidate["method"] == "sd" and candidate["candidate_id"] in self.drafts
            ]
            active_candidates = [ar_candidate] + sd_candidates
            stop_ids = set(int(value) for value in self.manifest["decode"]["stop_ids"])
            max_new_tokens = int(self.manifest["decode"]["max_new_tokens"])
            for candidate in sd_candidates:
                self.draft_algorithms[candidate["candidate_id"]].configure_verify_backend(
                    self._verify_backend == "reference"
                )
            # Protocol-matched AR tokens may be reused across sessions; AR timing is re-measured.
            phase_started = time.time()
            execution_stage = "ar_reference"
            for row in rows:
                self._check_budget("ar_reference")
                try:
                    self.reference_tokens[row["request_id"]] = self._reference(
                        row, ar_candidate, max_new_tokens, stop_ids
                    )
                except Exception as exc:  # noqa: BLE001
                    self.reference_tokens[row["request_id"]] = None
                    errors.append(
                        {
                            "type": type(exc).__name__,
                            "stage": "ar_reference",
                            "request_id": row["request_id"],
                            "message": str(exc),
                        }
                    )
            self._mark_phase("ar_references", phase_started)
            self.ledger.start_measured()
            for session_index in range(self.sessions):
                session_id = f"{self.run_id}-s{session_index}"
                execution_stage = "warmup"
                for row in rows[: self.warmups]:
                    for candidate in active_candidates:
                        self._check_budget("warmup")
                        phase_started = time.time()
                        outcome = self._execute_request(row, candidate, max_new_tokens, sorted(stop_ids))
                        self._mark_phase("warmup", phase_started)
                        if outcome["error"] is not None:
                            errors.append(
                                {
                                    **outcome["error"],
                                    "stage": "warmup",
                                    "session_id": session_id,
                                    "request_id": row.get("request_id"),
                                    "candidate_id": candidate.get("candidate_id"),
                                }
                            )
                execution_stage = "request"
                for row_index, row in enumerate(rows):
                    order = active_candidates
                    if order:
                        shift = (row_index + session_index) % len(order)
                        order = order[shift:] + order[:shift]
                    for candidate in order:
                        self._check_budget("request")
                        self.execution_order += 1
                        phase_started = time.time()
                        record = self._run_once(
                            row, candidate, max_new_tokens, stop_ids, session_id, session_index, record=True
                        )
                        self._mark_phase("request_record" if record is not None else "request_warmup", phase_started)
                        if record is not None:
                            records.append(record)
                            self._append_record(record, len(records))
            self.ledger.stop_measured()
        except GpuBusy as exc:
            stop_reason = "gpu_busy"
            errors.append({"type": type(exc).__name__, "message": str(exc)})
            status = "blocked"
        except BudgetExhausted as exc:
            stop_reason = "budget_exhausted"
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        except DraftRejected as exc:
            stop_reason = f"{execution_stage}_rejected"
            errors.append(
                {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "message": str(exc),
                    "stage": execution_stage,
                }
            )
        except Exception as exc:  # noqa: BLE001
            stop_reason = "runtime_error"
            errors.append({"type": type(exc).__name__, "message": str(exc), "stage": execution_stage})
        finally:
            if self.ledger is not None:
                self.ledger.stop_measured()
            if cleanup_needed:
                try:
                    self._release_models()
                except Exception as exc:  # cleanup failure must remain visible in the audit
                    errors.append({"type": type(exc).__name__, "message": str(exc), "stage": "gpu_cleanup"})
                    stop_reason = "gpu_cleanup_failed"
            if status != "blocked":
                status = self._aggregate_status(records, errors, stop_reason)
            if status == "failed" and stop_reason == "completed":
                stop_reason = "all_records_failed"
            if lease is not None:
                lease.release()
            try:
                self._write_ledger_checkpoint(status=status, records=len(records))
            finally:
                self._close_audit()
        if status == "blocked":
            return self._blocked_result(reason=stop_reason, ready=ready, errors=errors)
        return {
            "status": status,
            "stop_reason": stop_reason,
            "records": records,
            "errors": errors,
            "skipped_candidates": self.skipped_candidates,
            "cold_start": self.cold_start,
            "ledger": {
                "budget_gpu_hours": self.budget_gpu_hours,
                "all_allocated_gpu_hours": self.ledger.all_allocated_gpu_hours,
                "measured_run_gpu_hours": self.ledger.measured_run_gpu_hours,
                "intervals": self.ledger.intervals,
                "phases": self.phase_intervals,
                "checkpoint_path": str(self.ledger_path),
            },
            "occupancy": self.occupancy,
            "capabilities": capabilities_document(
                state="gpu_smoke" if records and status != "failed" else "blocked"
            ),
        }

    def _release_models(self) -> None:
        """Drop owned CUDA tensors before releasing the lease or stopping its ledger."""
        self.model = None
        self.processor = None
        self.drafts.clear()
        self.draft_algorithms.clear()
        gc.collect()
        self._sync_device()
        self._torch().cuda.empty_cache()

    def _build_protocol(self) -> None:
        from .schema import protocol_payload, protocol_scope_hash  # noqa: PLC0415

        self.protocol_payload = protocol_payload(
            self.manifest, self.workload.rows_digest, self.workload.labels_digest
        )
        # The scope is the frozen manifest contract also checked by report/recommend.
        # Request hashes additionally bind the actual loaded runtime identities.
        self.protocol_scope_hash = protocol_scope_hash(self.protocol_payload)
        self.protocol_payload["resolved_target_identity"] = self.resolved_target_identity
        self.protocol_payload["resolved_draft_identities"] = self.resolved_draft_identities
        self.protocol_payload["verify"] = {
            "verification": self._verification,
            "verify_backend": self._verify_backend,
        }
        self.protocol_payload["timing_boundary"] = "request_ready_to_committed_output"

    def _reference(self, row: dict, candidate: dict, max_new_tokens: int, stop_ids: set[int]) -> list[int]:
        outcome = self._execute_request(row, candidate, max_new_tokens, sorted(stop_ids))
        if outcome["error"] is not None:
            raise RuntimeError(
                f"AR reference failed for {row.get('request_id')}: "
                f"{outcome['error'].get('type')}: {outcome['error'].get('message')}"
            )
        return outcome["tokens"]

    def _run_once(
        self,
        row: dict,
        candidate: dict,
        max_new_tokens: int,
        stop_ids: set[int],
        session_id: str,
        session_index: int,
        *,
        record: bool,
    ) -> dict | None:
        self._reset_peak()
        outcome = self._execute_request(row, candidate, max_new_tokens, sorted(stop_ids))
        if not record:
            return None
        tokens = outcome["tokens"]
        reference = tokens if candidate["method"] == "ar" else self.reference_tokens.get(row["request_id"])
        return self._record(
            row=row,
            candidate=candidate,
            session_id=session_id,
            repeat_index=session_index,
            tokens=tokens,
            wall_s=0.0,
            stats=outcome["stats"],
            reference=reference,
            stop_ids=stop_ids,
            v_tokens=outcome["v_tokens"],
            error=outcome["error"],
            timings=outcome["timings"],
        )
