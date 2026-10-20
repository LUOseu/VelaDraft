from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vlm_dflash.workload_advisor.backend import (
    CAPABILITIES,
    GpuBusy,
    GpuLedger,
    GpuLease,
    capabilities_document,
    check_external_occupancy,
    readiness,
    resolve_device_uuid,
    validate_execution_contract,
)
from vlm_dflash.workload_advisor.backend_hf import (
    BudgetExhausted,
    DraftRejected,
    HFWorkloadBackend,
    score_output,
)
from vlm_dflash.workload_advisor.plan import build_plan
from vlm_dflash.workload_advisor.schema import sha256_file, validate_record
from vlm_dflash.workload_advisor.workload import load_workload, validation_report

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "workload_advisor" / "mini"


def _manifest() -> dict:
    return json.loads((MINI / "manifest.json").read_text(encoding="utf-8"))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def synchronize(self, device=None) -> None:
        return None

    def reset_peak_memory_stats(self, device=None) -> None:
        return None

    def max_memory_allocated(self, device=None) -> int:
        return 0

    def max_memory_reserved(self, device=None) -> int:
        return 0

    def empty_cache(self) -> None:
        return None


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"

    def inference_mode(self):
        import contextlib

        return contextlib.nullcontext()


class _FakeWeight:
    def __init__(self) -> None:
        self.token = object()

    def data_ptr(self):
        return id(self.token)


class _FakeEmbeddings:
    def __init__(self, weight: _FakeWeight) -> None:
        self.weight = weight


class _FakeModel:
    def __init__(self, model_type: str = "qwen2_5_vl") -> None:
        self.config = SimpleNamespace(model_type=model_type, _commit_hash="resolved-rev", tie_word_embeddings=True)
        self._weight = _FakeWeight()
        self.lm_head = SimpleNamespace(weight=self._weight)
        self.moved_to = None
        self.evaluated = False
        self.frozen = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def requires_grad_(self, value):
        self.frozen = not value
        return self

    def get_input_embeddings(self):
        return _FakeEmbeddings(self._weight)


class _FakeProcessor:
    def __init__(self, min_pixels=200704, max_pixels=12845056) -> None:
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.image_processor = SimpleNamespace(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            merge_size=2,
        )
        self.chat_template = "{{ messages }}"
        self.decoded = "answer"

    def batch_decode(self, token_rows, skip_special_tokens=True):
        return [self.decoded for _ in token_rows]


def _backend_for_loader(tmp: Path, *, device: str = "cuda:1") -> HFWorkloadBackend:
    workload = load_workload(MINI / "manifest.json")
    manifest = _manifest()
    backend = HFWorkloadBackend(
        workload,
        out_dir=tmp / "out",
        sessions=1,
        warmups=0,
        device=device,
    )
    backend.manifest = manifest
    workload.manifest = manifest
    return backend


class BackendTests(unittest.TestCase):
    def test_capabilities_do_not_claim_streaming_or_ttft(self) -> None:
        self.assertFalse(CAPABILITIES["generation"]["streaming"])
        self.assertFalse(CAPABILITIES["generation"]["cancel"])
        self.assertFalse(CAPABILITIES["metrics"]["ttft"])
        self.assertFalse(CAPABILITIES["metrics"]["itl"])
        self.assertFalse(CAPABILITIES["visibility"]["proposal_commit_events"])
        document = capabilities_document(state="blocked", evidence_path="blocker.json")
        self.assertEqual(document["evidence"]["state"], "blocked")
        self.assertFalse(document["capabilities"]["generation"]["streaming"])

    def test_readiness_is_blocked_without_a_local_gpu(self) -> None:
        report = readiness(_manifest())
        self.assertEqual(report["status"], "blocked")
        codes = {blocker["code"] for blocker in report["blockers"]}
        self.assertTrue({"torch_missing", "no_gpu"} & codes)

    def test_gpu_lease_acquires_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lease = GpuLease(["GPU-test-b", "GPU-test-a"], lock_dir=Path(tmp))
            lease.acquire()
            self.assertTrue((Path(tmp) / "GPU-test-a.lock").exists())
            self.assertTrue((Path(tmp) / "GPU-test-b.lock").exists())
            self.assertGreaterEqual(lease.allocated_hours, 0.0)
            lease.release()
            again = GpuLease(["GPU-test-a"], lock_dir=Path(tmp)).acquire()
            again.release()

    def test_gpu_lease_rejects_a_concurrent_holder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "import fcntl, pathlib, sys, time\n"
                "path = pathlib.Path(sys.argv[1]) / (sys.argv[2] + '.lock')\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                "handle = open(path, 'a+')\n"
                "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
                "handle.write('held\\n'); handle.flush()\n"
                "time.sleep(30)\n"
            )
            holder = subprocess.Popen([sys.executable, "-c", script, tmp, "GPU-contended"])
            try:
                deadline = time.time() + 5
                lock_path = Path(tmp) / "GPU-contended.lock"
                while time.time() < deadline:
                    if lock_path.exists() and lock_path.read_text(encoding="utf-8").strip():
                        break
                    time.sleep(0.05)
                with self.assertRaises(GpuBusy):
                    GpuLease(["GPU-contended"], lock_dir=Path(tmp)).acquire()
            finally:
                holder.terminate()
                holder.wait(timeout=5)

    def test_budget_ledger_counts_all_devices_and_intervals(self) -> None:
        lease = GpuLease(["GPU-a", "GPU-b"], lock_dir=Path(tempfile.gettempdir()))
        now = time.time()
        lease.acquired_at = now - 3600.0
        lease.released_at = now
        ledger = GpuLedger(4.0, lease)
        ledger.measured_started_at = now - 1800.0
        ledger.measured_released_at = now
        self.assertAlmostEqual(ledger.all_allocated_gpu_hours, 2.0, places=3)
        self.assertAlmostEqual(ledger.measured_run_gpu_hours, 1.0, places=3)
        budget = ledger.check_budget()
        self.assertEqual(budget["budget_gpu_hours"], 4.0)
        self.assertFalse(budget["exceeded"])

    def test_plan_counts_and_refuses_to_predict(self) -> None:
        workload = load_workload(MINI / "manifest.json")
        validation = validation_report(workload)
        plan = build_plan(_manifest(), validation=validation, readiness=readiness(_manifest()), budget_gpu_hours=4.0)
        self.assertEqual(plan["counts"]["holdout_groups"], 24)
        self.assertEqual(plan["counts"]["measured_runs"], 216)
        self.assertEqual(plan["counts"]["warmup_runs"], 36)
        self.assertIsNone(plan["predicted_speedup"])
        self.assertIsNone(plan["predicted_gpu_hours"])
        self.assertFalse(plan["ranking_allowed"])
        self.assertEqual(plan["overall_status"], "blocked")

    def test_plan_marks_missing_optional_candidate_as_partial(self) -> None:
        workload = load_workload(MINI / "manifest.json")
        validation = validation_report(workload)
        partial = {
            "status": "partial",
            "blockers": [{"code": "checkpoint_missing", "candidate_id": "dfly-fast-wrong"}],
            "candidate_blockers": [{"code": "checkpoint_missing", "candidate_id": "dfly-fast-wrong"}],
            "warnings": [],
        }
        plan = build_plan(_manifest(), validation=validation, readiness=partial, budget_gpu_hours=4.0)
        self.assertEqual(plan["overall_status"], "partial")
        self.assertEqual(plan["blocked_candidates"], ["dfly-fast-wrong"])
        matrix = {entry["candidate_id"]: entry for entry in plan["candidate_matrix"]}
        self.assertEqual(matrix["ar"]["status"], "ready")
        self.assertEqual(matrix["dflash-exact"]["status"], "ready")
        self.assertEqual(matrix["dfly-fast-wrong"]["status"], "blocked")

    def test_run_cli_writes_blocker_and_no_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "workload_advisor.py"),
                    "run",
                    "--manifest",
                    str(MINI / "manifest.json"),
                    "--out-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=120,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertTrue((out / "blocker.json").exists())
            self.assertTrue((out / "validation.json").exists())
            self.assertTrue((out / "plan.json").exists())
            self.assertFalse((out / "records.jsonl").exists())
            blocker = json.loads((out / "blocker.json").read_text(encoding="utf-8"))
            self.assertEqual(blocker["status"], "blocked")
            self.assertEqual(blocker["reason"], "backend_not_ready")

    def test_run_cli_rejects_invalid_budget_before_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "workload_advisor.py"),
                    "run",
                    "--manifest",
                    str(MINI / "manifest.json"),
                    "--out-dir",
                    str(out),
                    "--budget-gpu-hours",
                    "0",
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=120,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            error = json.loads((out / "run_error.json").read_text(encoding="utf-8"))
            self.assertEqual(error["reason"], "invalid_run_configuration")
            self.assertFalse((out / "blocker.json").exists())

    def test_cuda_visible_devices_resolves_only_the_used_physical_uuid(self) -> None:
        devices = [
            {"uuid": "GPU-a", "name": "A"},
            {"uuid": "GPU-b", "name": "B"},
            {"uuid": "GPU-c", "name": "C"},
        ]
        self.assertEqual(resolve_device_uuid("cuda:1", devices)["uuid"], "GPU-b")
        self.assertEqual(
            resolve_device_uuid("cuda:1", devices, cuda_visible_devices="GPU-c,GPU-a")["uuid"],
            "GPU-a",
        )
        self.assertEqual(
            resolve_device_uuid("cuda:0", devices, cuda_visible_devices="2")["uuid"],
            "GPU-c",
        )
        with self.assertRaises(ValueError):
            resolve_device_uuid("cuda:2", devices, cuda_visible_devices="GPU-a,GPU-b")

    def test_unsupported_execution_modes_are_rejected_before_gpu(self) -> None:
        manifest = _manifest()
        manifest["concurrency"] = 2
        manifest["measurement_scope"] = "server_endpoint"
        manifest["backend"]["server_endpoint"] = "http://example.invalid"
        manifest["backend"]["streaming"] = True
        manifest["decode"]["verification"] = "parallel"
        issues = validate_execution_contract(manifest)
        codes = {issue["code"] for issue in issues}
        self.assertTrue(
            {
                "unsupported_concurrency",
                "unsupported_measurement_scope",
                "unsupported_server_endpoint",
                "unsupported_streaming",
                "unsupported_verification",
            }
            <= codes
        )


class ScorerTests(unittest.TestCase):
    schema = {
        "type": "object",
        "required": ["name", "count"],
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "additionalProperties": False,
    }

    def _score(self, text: str) -> dict:
        return score_output(
            "qa",
            text,
            {"reference": {}},
            {"name": "json_schema_valid", "version": "1", "schema": self.schema},
        )

    def test_json_schema_scorer_checks_required_and_types(self) -> None:
        valid = self._score('{"name": "ok", "count": 2}')
        self.assertEqual(valid["status"], "pass")
        self.assertTrue(valid["json_schema_valid"])
        self.assertFalse(self._score("{}")["json_schema_valid"])
        self.assertFalse(self._score('{"name": 1, "count": 2}')["json_schema_valid"])
        self.assertFalse(self._score('{"name": "ok", "count": 2, "extra": true}')["json_schema_valid"])
        self.assertFalse(self._score("[]")["json_schema_valid"])
        self.assertFalse(self._score("not-json")["json_schema_valid"])

    def test_schema_scorer_without_frozen_schema_is_not_evaluated(self) -> None:
        result = score_output(
            "qa",
            "{}",
            {"reference": {}},
            {"name": "json_schema_valid", "version": "1"},
        )
        self.assertFalse(result["evaluated"])
        self.assertEqual(result["reason"], "missing_frozen_schema")

    def test_syntax_only_scorer_is_named_honestly(self) -> None:
        result = score_output(
            "qa",
            "{}",
            {"reference": {}},
            {"name": "json_object_syntax", "version": "1"},
        )
        self.assertTrue(result["json_object_valid"])


class LoaderTests(unittest.TestCase):
    def _prepare_model_dir(self, tmp: Path, backend: HFWorkloadBackend) -> None:
        model_dir = tmp / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "qwen2_5_vl"}', encoding="utf-8")
        weights = model_dir / "model.safetensors"
        weights.write_bytes(b"checkpoint-bytes")
        backend.manifest["target"].update(
            {
                "model_path": str(model_dir),
                "revision": "rev-7",
                "processor": str(model_dir),
                "processor_revision": "proc-rev-3",
                "weights_sha256": sha256_file(weights),
                "vision": {"min_pixels": 200704, "max_pixels": 12845056},
            }
        )
        backend.manifest["processor"] = {"chat_template": "qwen2_5_vl", "revision": "proc-rev-3"}

    def test_loader_applies_revision_processor_and_pixel_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = _backend_for_loader(tmp)
            self._prepare_model_dir(tmp, backend)
            backend.manifest["target"]["chat_template"] = "{{ custom_chat_template }}"
            backend.manifest["processor"]["chat_template"] = "{{ custom_chat_template }}"
            backend._torch = lambda: _FakeTorch()
            calls = {}

            def load_components(*, model_path, model_kwargs, processor_path, processor_kwargs):
                calls.update(
                    {
                        "model_path": model_path,
                        "model_kwargs": model_kwargs,
                        "processor_path": processor_path,
                        "processor_kwargs": processor_kwargs,
                    }
                )
                return _FakeModel(), _FakeProcessor()

            backend._load_pretrained_components = load_components
            backend._load_model()
            self.assertEqual(calls["model_kwargs"]["revision"], "rev-7")
            self.assertEqual(calls["processor_kwargs"]["revision"], "proc-rev-3")
            self.assertEqual(calls["processor_kwargs"]["min_pixels"], 200704)
            self.assertEqual(calls["processor_kwargs"]["max_pixels"], 12845056)
            self.assertEqual(backend.processor.chat_template, "{{ custom_chat_template }}")
            self.assertEqual(backend.processor.tokenizer.padding_side, "left")
            self.assertEqual(backend.model.moved_to, "cuda:1")

    def test_loader_rejects_declared_weight_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = _backend_for_loader(tmp)
            self._prepare_model_dir(tmp, backend)
            backend.manifest["target"]["weights_sha256"] = "0" * 64
            backend._torch = lambda: _FakeTorch()
            backend._load_pretrained_components = lambda **kwargs: (_FakeModel(), _FakeProcessor())
            with self.assertRaises(DraftRejected) as raised:
                backend._load_model()
            self.assertEqual(raised.exception.code, "weights_hash_mismatch")

    def test_loader_rejects_a_template_or_pixel_value_it_did_not_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = _backend_for_loader(tmp)
            self._prepare_model_dir(tmp, backend)
            backend._torch = lambda: _FakeTorch()
            backend._load_pretrained_components = lambda **kwargs: (
                _FakeModel(),
                _FakeProcessor(min_pixels=1, max_pixels=2),
            )
            with self.assertRaises(DraftRejected) as raised:
                backend._load_model()
            self.assertEqual(raised.exception.code, "pixel_bounds_not_applied")


class CheckpointTests(unittest.TestCase):
    def _backend(self, tmp: Path) -> HFWorkloadBackend:
        backend = _backend_for_loader(tmp)
        backend._torch = lambda: _FakeTorch()
        backend.model = _FakeModel()
        backend.processor = _FakeProcessor()
        backend.resolved_target_identity = {}
        return backend

    def _checkpoint(self, tmp: Path, name: str = "draft.pt") -> tuple[Path, str]:
        path = tmp / name
        path.write_bytes(b"draft-weight-bytes")
        return path, sha256_file(path)

    def test_checkpoint_metadata_is_compared_with_declared_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            path, digest = self._checkpoint(tmp)
            candidate = {
                "candidate_id": "candidate",
                "method": "sd",
                "algorithm": "dflash",
                "checkpoint": str(path),
                "block_size": 8,
                "draft_identity": {
                    "source": "hf-run",
                    "revision": "rev-2",
                    "checkpoint_sha256": digest,
                },
            }
            algorithm = SimpleNamespace(name="dflash")
            meta = {
                "algorithm": "dflash",
                "head": "tied_embedding",
                "feature_mode": "offline",
                "revision": "rev-2",
                "source": "hf-run",
                "block_size": 8,
                "draft_config": {"block_size": 8},
            }
            resolved = backend._validate_checkpoint_metadata(candidate, algorithm, meta, digest)
            self.assertEqual(resolved["checkpoint_sha256"], digest)
            self.assertTrue(resolved["revision_verified"])

            bad_revision = dict(candidate)
            bad_revision["draft_identity"] = {**candidate["draft_identity"], "revision": "rev-other"}
            with self.assertRaises(DraftRejected) as raised:
                backend._validate_checkpoint_metadata(bad_revision, algorithm, meta, digest)
            self.assertEqual(raised.exception.code, "checkpoint_revision_mismatch")

    def test_corrupt_checkpoint_is_isolated_from_ar_and_good_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            bad_path, bad_digest = self._checkpoint(tmp, "bad.pt")
            good_path, good_digest = self._checkpoint(tmp, "good.pt")
            backend.manifest["draft_candidates"] = [
                {"candidate_id": "ar", "method": "ar", "draft_identity": None},
                {
                    "candidate_id": "bad",
                    "method": "sd",
                    "algorithm": "dflash",
                    "checkpoint": str(bad_path),
                    "draft_identity": {"source": "hf-run", "revision": "rev-2", "checkpoint_sha256": bad_digest},
                },
                {
                    "candidate_id": "good",
                    "method": "sd",
                    "algorithm": "dflash",
                    "checkpoint": str(good_path),
                    "draft_identity": {"source": "hf-run", "revision": "rev-2", "checkpoint_sha256": good_digest},
                },
            ]
            meta = {
                "algorithm": "dflash",
                "head": "tied_embedding",
                "feature_mode": "offline",
                "revision": "rev-2",
                "source": "hf-run",
                "block_size": 8,
                "draft_config": {"block_size": 8},
            }

            class FakeDraft:
                def to(self, **kwargs):
                    return self

                def eval(self):
                    return self

                def configure_verify_backend(self, reference):
                    return None

            class FakeAlgorithm:
                name = "dflash"

                def load_decoder(self, path):
                    if str(path).endswith("bad.pt"):
                        raise RuntimeError("corrupt checkpoint")
                    return FakeDraft(), meta

            with mock.patch("vlm_dflash.algorithms.get_algorithm", return_value=FakeAlgorithm()):
                errors = backend._load_drafts()
            self.assertEqual(set(backend.drafts), {"good"})
            self.assertEqual([item["candidate_id"] for item in backend.skipped_candidates], ["bad"])
            self.assertEqual(len(errors), 1)


class TimingTests(unittest.TestCase):
    def _backend(self, tmp: Path) -> tuple[HFWorkloadBackend, _Clock, list[str]]:
        backend = _backend_for_loader(tmp, device="cuda:1")
        clock = _Clock()
        syncs: list[str] = []
        backend.clock = clock
        backend._torch = lambda: _FakeTorch()

        def load_media(row):
            clock.advance(0.05)
            return []

        def prepare(row, images):
            clock.advance(0.10)
            return {"input_ids": SimpleNamespace(shape=(1, 3))}

        def run_ar(inputs, max_new_tokens, stop_ids):
            clock.advance(0.20)
            return object()

        def sync_device():
            syncs.append(backend.device)
            clock.advance(0.01)

        def decode_tokens(generated, inputs):
            clock.advance(0.02)
            return [10, 11]

        backend._load_media = load_media
        backend._prepare = prepare
        backend._run_ar = run_ar
        backend._sync_device = sync_device
        backend._decode_tokens = decode_tokens
        backend._target_visual_tokens = lambda inputs: 7
        return backend, clock, syncs

    def test_e2e_includes_preprocess_h2d_generate_sync_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            backend, _clock, syncs = self._backend(Path(tmp_name))
            row = {"request_id": "r1", "media": [], "messages": []}
            outcome = backend._execute_request(
                row,
                {"candidate_id": "ar", "method": "ar"},
                8,
                [1],
            )
            self.assertIsNone(outcome["error"])
            self.assertEqual(outcome["tokens"], [10, 11])
            self.assertAlmostEqual(outcome["timings"]["media_load_s"], 0.05)
            self.assertAlmostEqual(outcome["timings"]["preprocess_s"], 0.10)
            self.assertAlmostEqual(outcome["timings"]["generate_only_s"], 0.21)
            self.assertAlmostEqual(outcome["timings"]["commit_s"], 0.02)
            self.assertAlmostEqual(outcome["timings"]["e2e_s"], 0.33)
            self.assertEqual(syncs, ["cuda:1", "cuda:1"])

    def test_failed_request_keeps_elapsed_time_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            backend, clock, _syncs = self._backend(Path(tmp_name))

            def broken_prepare(row, images):
                clock.advance(0.03)
                raise RuntimeError("processor failed")

            backend._prepare = broken_prepare
            outcome = backend._execute_request(
                {"request_id": "r1"},
                {"candidate_id": "ar", "method": "ar"},
                8,
                [1],
            )
            self.assertEqual(outcome["error"]["stage"], "run")
            self.assertAlmostEqual(outcome["timings"]["e2e_s"], 0.03)
            self.assertNotIn("generate_only_s", outcome["timings"])


class LifecycleTests(unittest.TestCase):
    def _ready(self) -> dict:
        return {
            "status": "ready",
            "environment": {"devices": [{"uuid": "GPU-test", "name": "fake", "source": "test"}]},
            "candidate_blockers": [],
        }

    def _backend(self, tmp: Path, **kwargs) -> HFWorkloadBackend:
        workload = load_workload(MINI / "manifest.json")
        return HFWorkloadBackend(
            workload,
            out_dir=tmp / "out",
            sessions=1,
            warmups=0,
            device="cuda:0",
            lock_dir=tmp / "locks",
            **kwargs,
        )

    def test_invalid_budget_is_rejected_before_any_lease_or_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self._backend(Path(tmp_name), budget_gpu_hours=value)

    def test_lease_busy_is_a_blocker_not_an_uncaught_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            backend = self._backend(Path(tmp_name))
            backend._torch = lambda: _FakeTorch()
            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=self._ready()
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                return_value={"status": "checked", "busy": False, "processes": []},
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.GpuLease"
            ) as lease_class:
                lease_class.return_value.acquire.side_effect = GpuBusy("GPU-test")
                result = backend.run(max_requests=1)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stop_reason"], "gpu_busy")
            self.assertEqual(result["records"], [])

    def test_external_process_blocks_before_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            backend = self._backend(Path(tmp_name))
            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=self._ready()
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                return_value={"status": "checked", "busy": True, "processes": [{"pid": 42}]},
            ):
                result = backend.run(max_requests=1)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stop_reason"], "gpu_busy")
            self.assertEqual(result["records"], [])

    def test_only_the_logical_gpu_is_locked_and_ledgered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            backend.device = "cuda:1"
            backend._torch = lambda: _FakeTorch()
            backend._preflight_target_artifacts = lambda: None
            backend._load_model = lambda: None
            backend._load_drafts = lambda: []
            backend._build_protocol = lambda: setattr(backend, "protocol_scope_hash", "b" * 64)
            backend._reference = lambda row, candidate, max_new_tokens, stop_ids: [1]
            backend._run_once = lambda *args, **kwargs: {"error": None, "quality": {"status": "pass"}}
            ready = self._ready()
            ready["environment"]["devices"] = [
                {"uuid": "GPU-a", "name": "A"},
                {"uuid": "GPU-b", "name": "B"},
            ]
            captured = {}

            def lease_factory(uuids, **kwargs):
                captured["uuids"] = list(uuids)
                return GpuLease(uuids, lock_dir=tmp / "locks")

            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=ready
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                return_value={"status": "checked", "busy": False, "processes": []},
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.GpuLease", side_effect=lease_factory
            ):
                result = backend.run(max_requests=1)
            self.assertEqual(captured["uuids"], ["GPU-b"])
            self.assertEqual([device["uuid"] for device in backend.devices], ["GPU-b"])
            self.assertEqual(result["ledger"]["intervals"][0]["uuids"], ["GPU-b"])

    def test_budget_is_checked_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            backend._torch = lambda: _FakeTorch()
            loaded = []
            backend._load_model = lambda: loaded.append(True)

            def check(phase):
                if phase == "pre_load":
                    raise BudgetExhausted("no budget before load")
                return {"exceeded": False}

            backend._check_budget = check
            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=self._ready()
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                return_value={"status": "checked", "busy": False, "processes": []},
            ):
                result = backend.run(max_requests=1)
            self.assertEqual(loaded, [])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stop_reason"], "budget_exhausted")

    def test_external_process_after_lease_releases_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            backend._torch = lambda: _FakeTorch()
            loaded = []
            backend._load_model = lambda: loaded.append(True)
            occupancy = [
                {"status": "checked", "busy": False, "processes": []},
                {"status": "checked", "busy": True, "processes": [{"pid": 7}]},
            ]
            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=self._ready()
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                side_effect=occupancy,
            ):
                result = backend.run(max_requests=1)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stop_reason"], "gpu_busy")
            self.assertEqual(loaded, [])
            GpuLease(["GPU-test"], lock_dir=tmp / "locks").acquire().release()

    def test_target_preflight_rejection_keeps_the_specific_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            backend._torch = lambda: _FakeTorch()
            backend._preflight_target_artifacts = mock.Mock(
                side_effect=DraftRejected("weights_hash_mismatch", "declared weights do not match")
            )
            backend._load_model = mock.Mock()
            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=self._ready()
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                return_value={"status": "checked", "busy": False, "processes": []},
            ):
                result = backend.run(max_requests=1)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stop_reason"], "target_preflight_rejected")
            self.assertEqual(result["errors"][0]["code"], "weights_hash_mismatch")
            self.assertEqual(result["errors"][0]["stage"], "target_preflight")
            self.assertEqual(result["records"], [])
            backend._load_model.assert_not_called()
            ledger = json.loads((tmp / "out" / "ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["status"], "failed")

    def test_all_failed_records_cannot_return_ok_and_are_flushed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            backend._torch = lambda: _FakeTorch()
            backend._preflight_target_artifacts = lambda: None
            backend._load_model = lambda: None
            backend._load_drafts = lambda: []
            backend._build_protocol = lambda: setattr(backend, "protocol_scope_hash", "a" * 64)
            backend._reference = lambda row, candidate, max_new_tokens, stop_ids: [1]

            def run_once(row, candidate, max_new_tokens, stop_ids, session_id, session_index, *, record):
                return {"error": {"type": "boom"}, "quality": {"status": "fail"}}

            backend._run_once = run_once
            with mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.readiness", return_value=self._ready()
            ), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy",
                return_value={"status": "checked", "busy": False, "processes": []},
            ):
                result = backend.run(max_requests=2)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stop_reason"], "all_records_failed")
            written = (tmp / "out" / "records.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(written), 2)
            self.assertTrue((tmp / "out" / "ledger.json").exists())
            phases = {item["phase"] for item in result["ledger"]["phases"]}
            self.assertTrue({"model_load", "draft_load", "ar_references", "request_record"} <= phases)

    def test_scorer_exception_is_auditable_and_keeps_the_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            backend = self._backend(tmp)
            backend.model = _FakeModel()
            backend.processor = _FakeProcessor()
            backend.resolved_target_identity = {}
            backend.environment = {"devices": [], "transformers": {}, "torch": {}}
            backend.devices = [{"uuid": "GPU-test", "name": "fake"}]
            backend._resource = lambda **kwargs: {
                "all_allocated_gpu_hours": 0.0,
                "measured_run_gpu_hours": 0.0,
                "devices": [],
                "allocation_intervals": [],
                "gpu_memory_fit": True,
                "note": "test",
            }
            row = {
                "request_id": "r1",
                "group_id": "g1",
                "task": "chart_qa",
                "split": "holdout",
                "input_sha256": "f" * 64,
                "messages": [{"role": "user", "content": "question"}],
                "media": [],
            }
            backend.workload.labels["r1"] = {"reference": "answer"}
            with mock.patch("vlm_dflash.workload_advisor.backend_hf.score_output", side_effect=RuntimeError("bad scorer")):
                record = backend._record(
                    row=row,
                    candidate={"candidate_id": "ar", "method": "ar"},
                    session_id="s0",
                    repeat_index=0,
                    tokens=[1, 2],
                    wall_s=0.0,
                    stats=None,
                    reference=[1, 2],
                    stop_ids=set(),
                    v_tokens=1,
                    error=None,
                    timings={"e2e_s": 0.1, "generate_only_s": 0.08},
                )
            self.assertEqual(record["quality"]["reason"], "scorer_error")
            self.assertEqual(record["quality"]["error"]["type"], "RuntimeError")
            self.assertEqual(record["missing_reasons"]["quality"], "scorer_error")
            self.assertEqual(validate_record(record), [])


if __name__ == "__main__":
    unittest.main()
