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

import json
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any

from .backend import GpuBusy, GpuLedger, GpuLease, capabilities_document, detect_environment, readiness
from .records import build_latency, build_raw_counts, build_resource_usage, make_record
from .schema import canonical_json, request_protocol_hash, sha256_json


class BudgetExhausted(RuntimeError):
    pass


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


def score_output(task: str, output: str, label: dict, scorer: dict | None) -> dict:
    """Deterministic task scorer. Free text without a scorer is not evaluated."""
    if scorer is None:
        return {"evaluated": False, "reason": "no_frozen_scorer", "scorer": None}
    name = str(scorer.get("name") or "")
    threshold = scorer.get("threshold")
    references = label.get("references") if label.get("references") is not None else [label.get("reference")]
    references = [reference for reference in references if reference is not None]
    result: dict[str, Any] = {"evaluated": True, "scorer": {"name": name, "version": scorer.get("version")}, "task": task}
    if name in ("normalized_exact_match", "exact_match"):
        best = max((1.0 if _normalize(output) == _normalize(reference) else 0.0) for reference in references) if references else 0.0
        result.update({"score": best, "threshold": threshold, "status": "pass" if threshold is None or best >= threshold else "fail"})
    elif name == "field_f1":
        best = 0.0
        details = None
        prediction = _json_object(output)
        for reference in references:
            reference_object = reference if isinstance(reference, dict) else _json_object(str(reference))
            if reference_object is None:
                continue
            f1, detail = _field_f1(prediction, reference_object)
            if f1 >= best:
                best, details = f1, detail
        result.update(
            {
                "score": best,
                "threshold": threshold,
                "status": "pass" if threshold is None or best >= threshold else "fail",
                "field_metrics": details,
            }
        )
    elif name == "json_schema_valid":
        valid = prediction is not None
        result.update({"score": 1.0 if valid else 0.0, "threshold": 1.0, "status": "pass" if valid else "fail"})
    else:
        return {"evaluated": False, "reason": f"unknown_scorer:{name}", "scorer": {"name": name}}
    result["json_schema_valid"] = _json_object(output) is not None
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
    ):
        self.workload = workload
        self.manifest = workload.manifest
        self.out_dir = Path(out_dir)
        self.budget_gpu_hours = float(budget_gpu_hours)
        self.sessions = int(sessions)
        self.warmups = int(warmups)
        self.device = device
        self.lock_dir = lock_dir
        self.model = None
        self.processor = None
        self.drafts: dict[str, Any] = {}
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

    # ------------------------------------------------------------------ setup
    def _torch(self):
        import torch  # noqa: PLC0415

        return torch

    def _load_model(self):
        from transformers import AutoProcessor  # noqa: PLC0415

        target = self.manifest["target"]
        model_path = target.get("model_path") or target.get("model_id")
        local_only = bool(target.get("local_files_only", True))
        dtype_name = str(target.get("dtype", "bfloat16"))
        dtype = getattr(self._torch(), dtype_name)
        kwargs = {
            "dtype": dtype,
            "attn_implementation": target.get("attn_implementation", "sdpa"),
            "local_files_only": local_only,
        }
        try:
            from transformers import AutoModelForImageTextToText as AutoModel  # noqa: PLC0415

            model = AutoModel.from_pretrained(model_path, **kwargs)
        except (ImportError, ValueError):
            from transformers import AutoModelForVision2Seq as AutoModel  # noqa: PLC0415

            model = AutoModel.from_pretrained(model_path, **kwargs)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)
        self.model = model
        self.processor = processor

    def _load_drafts(self):
        from vlm_dflash.algorithms import get_algorithm  # noqa: PLC0415

        for candidate in self.manifest.get("draft_candidates") or []:
            if candidate.get("method") != "sd":
                continue
            if candidate.get("candidate_id") in self.blocked_candidate_ids:
                continue
            started = time.perf_counter()
            algorithm = get_algorithm(candidate.get("algorithm"))
            draft, _ = algorithm.load_decoder(candidate["checkpoint"])
            draft.to(device=self.device, dtype=getattr(self._torch(), str(self.manifest["target"].get("dtype", "bfloat16"))))
            draft.eval()
            self.drafts[candidate["candidate_id"]] = draft
            self.cold_start[candidate["candidate_id"]] = time.perf_counter() - started

    def _prepare(self, row: dict) -> dict:
        from PIL import Image  # noqa: PLC0415

        from vlm_dflash.target.prompts import encode_conversation  # noqa: PLC0415

        images = []
        for item in row.get("media", []):
            path = (self.workload.media_root / item["path"]).resolve()
            images.append(Image.open(path).convert("RGB"))
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
    def _run_ar(self, inputs: dict, max_new_tokens: int, stop_ids: list[int]) -> tuple[list[int], float]:
        torch = self._torch()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=stop_ids[0] if len(stop_ids) == 1 else stop_ids,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        prompt_len = int(inputs["input_ids"].shape[1])
        return generated[0, prompt_len:].tolist(), wall

    def _run_sd(self, candidate_id: str, inputs: dict, max_new_tokens: int, stop_ids: list[int]) -> tuple[list[int], dict, float]:
        torch = self._torch()
        draft = self.drafts[candidate_id]
        torch.cuda.synchronize()
        started = time.perf_counter()
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
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        prompt_len = int(inputs["input_ids"].shape[1])
        return output_ids[0, prompt_len:].tolist(), dict(stats), wall

    def _peak_memory(self) -> dict:
        torch = self._torch()
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }

    def _reset_peak(self) -> None:
        self._torch().cuda.reset_peak_memory_stats(self.device)

    def _resource(self, fit: bool | None = None, purpose: str = "run") -> dict:
        devices = self.devices
        peaks = self._peak_memory() if self.model is not None else {}
        device_rows = [
            {
                "uuid": device["uuid"],
                "name": device.get("name"),
                "peak_allocated_bytes": peaks.get("peak_allocated_bytes"),
                "peak_reserved_bytes": peaks.get("peak_reserved_bytes"),
                "peak_scope": "request_includes_model_weights",
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
            allocation_intervals=[],
            gpu_memory_fit=fit,
            note="all_allocated_gpu_hours includes load, warmup, failures and idle allocation",
        )

    def _quality(self, row: dict, output_tokens: list[int]) -> dict:
        label = self.workload.labels.get(row["request_id"])
        if label is None:
            return {"evaluated": False, "reason": "no_label_for_request", "scorer": None}
        scorer = (self.manifest.get("scorers") or {}).get(row["task"])
        text = self.processor.batch_decode([output_tokens], skip_special_tokens=True)[0]
        return score_output(row["task"], text, label, scorer)

    def _exact(self, candidate_tokens: list[int], reference_tokens: list[int], stop_ids: set[int]) -> bool:
        from vlm_dflash.eval.equivalence import sequences_equal  # noqa: PLC0415

        return sequences_equal(candidate_tokens, reference_tokens, stop_ids)

    def _base_identity(self) -> tuple[dict, dict]:
        env = detect_environment()
        target = self.manifest["target"]
        target_identity = {
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
        backend_identity = {
            "name": (self.manifest.get("backend") or {}).get("name", "hf-transformers"),
            "version": (env.get("transformers") or {}).get("version"),
            "torch_version": (env.get("torch") or {}).get("version"),
            "code_commit": _repo_commit(),
            "device_topology": {
                "devices": env.get("devices") or [],
                "concurrency": self.manifest.get("concurrency"),
            },
        }
        return target_identity, backend_identity

    def _candidate_identity(self, candidate: dict) -> dict | None:
        if candidate.get("method") == "ar":
            return None
        identity = dict(candidate.get("draft_identity") or {})
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
        latency, reasons = build_latency(e2e_s=wall_s, n_samples=1)
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
            quality=self._quality(row, tokens) if error is None else {"evaluated": False, "reason": "error"},
            measurement_scope="local_client",
            raw_counts=counts,
            latency=latency,
            resource_usage=self._resource(fit=error is None),
            output_tokens_R=len(tokens),
            visual_tokens={"target_V": v_tokens, "draft_V": None},
            error=error,
            missing_reasons=reasons,
            extra={"concurrency": self.manifest.get("concurrency"), "counts_verified": True},
        )
        return record

    # --------------------------------------------------------------- orchestrate
    def run(self, *, rows: list[dict] | None = None, max_requests: int | None = None) -> dict:
        ready = readiness(self.manifest)
        if ready["status"] == "blocked":
            return {
                "status": "blocked",
                "readiness": ready,
                "records": [],
                "capabilities": capabilities_document(state="blocked"),
            }
        rows = list(rows or [row for row in self.workload.rows if row["split"] == "holdout"])
        if max_requests is not None:
            rows = rows[:max_requests]
        if not rows:
            raise ValueError("no holdout rows to run")
        uuids = [device["uuid"] for device in ready["environment"].get("devices") or []]
        self.devices = ready["environment"].get("devices") or []
        lease = GpuLease(uuids, **({"lock_dir": self.lock_dir} if self.lock_dir else {})).acquire()
        self.ledger = GpuLedger(self.budget_gpu_hours, lease)
        records: list[dict] = []
        errors: list[dict] = []
        stop_reason = "completed"
        self.blocked_candidate_ids = {
            item.get("candidate_id") for item in ready.get("candidate_blockers") or []
        }
        try:
            self._torch().cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            self._load_model()
            self.cold_start["target"] = time.perf_counter() - started
            self._load_drafts()
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
                return {
                    "status": "blocked",
                    "readiness": ready,
                    "records": [],
                    "capabilities": capabilities_document(state="blocked"),
                }
            sd_candidates = [candidate for candidate in candidates if candidate["method"] == "sd"]
            ar_candidate = next(candidate for candidate in candidates if candidate["method"] == "ar")
            stop_ids = set(int(value) for value in self.manifest["decode"]["stop_ids"])
            max_new_tokens = int(self.manifest["decode"]["max_new_tokens"])
            # Protocol-matched AR tokens may be reused across sessions; AR timing is re-measured.
            for row in rows:
                try:
                    self.reference_tokens[row["request_id"]] = self._reference(row, max_new_tokens, stop_ids)
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
            self.ledger.start_measured()
            for session_index in range(self.sessions):
                session_id = f"{self.run_id}-s{session_index}"
                for row in rows[: self.warmups]:
                    for candidate in candidates:
                        self._run_once(row, candidate, max_new_tokens, stop_ids, session_id, session_index, record=False)
                for row_index, row in enumerate(rows):
                    order = [ar_candidate] + sd_candidates
                    if order:
                        shift = (row_index + session_index) % len(order)
                        order = order[shift:] + order[:shift]
                    for candidate in order:
                        budget = self.ledger.check_budget()
                        if budget["exceeded"]:
                            stop_reason = "budget_exhausted"
                            raise BudgetExhausted(
                                f"GPU budget {self.budget_gpu_hours} h reached ({budget['all_allocated_gpu_hours']:.3f} h)."
                            )
                        self.execution_order += 1
                        record = self._run_once(
                            row, candidate, max_new_tokens, stop_ids, session_id, session_index, record=True
                        )
                        if record is not None:
                            records.append(record)
            self.ledger.stop_measured()
        except (GpuBusy, BudgetExhausted) as exc:
            stop_reason = "gpu_busy" if isinstance(exc, GpuBusy) else "budget_exhausted"
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            stop_reason = "runtime_error"
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        finally:
            self.ledger.stop_measured()
            self.ledger.record_interval("run")
            lease.release()
        return {
            "status": "partial" if errors or self.skipped_candidates else "ok",
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
            },
            "capabilities": capabilities_document(state="gpu_smoke" if records else "blocked"),
        }

    def _build_protocol(self) -> None:
        from .schema import protocol_payload, protocol_scope_hash  # noqa: PLC0415

        self.protocol_payload = protocol_payload(
            self.manifest, self.workload.rows_digest, self.workload.labels_digest
        )
        self.protocol_scope_hash = protocol_scope_hash(self.protocol_payload)

    def _reference(self, row: dict, max_new_tokens: int, stop_ids: set[int]) -> list[int]:
        inputs = self._prepare(row)
        tokens, _ = self._run_ar(inputs, max_new_tokens, sorted(stop_ids))
        return tokens

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
        torch = self._torch()
        error = None
        tokens: list[int] = []
        wall = 0.0
        stats = None
        inputs = None
        self._reset_peak()
        try:
            inputs = self._prepare(row)
            if candidate["method"] == "ar":
                tokens, wall = self._run_ar(inputs, max_new_tokens, sorted(stop_ids))
            else:
                tokens, stats, wall = self._run_sd(candidate["candidate_id"], inputs, max_new_tokens, sorted(stop_ids))
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - GPU-only
            torch.cuda.empty_cache()
            error = {"type": "oom", "stage": "run", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            error = {"type": type(exc).__name__, "stage": "run", "message": str(exc)}
        if not record:
            return None
        v_tokens = self._target_visual_tokens(inputs) if inputs is not None else None
        reference = tokens if candidate["method"] == "ar" else self.reference_tokens.get(row["request_id"])
        return self._record(
            row=row,
            candidate=candidate,
            session_id=session_id,
            repeat_index=session_index,
            tokens=tokens,
            wall_s=wall,
            stats=stats,
            reference=reference,
            stop_ids=stop_ids,
            v_tokens=v_tokens,
            error=error,
        )
