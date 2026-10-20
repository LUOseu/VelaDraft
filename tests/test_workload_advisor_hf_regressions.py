"""CPU regressions for HF orchestration, without claiming GPU validation."""
from __future__ import annotations

import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

from tests import test_workload_advisor_backend as fixtures
from vlm_dflash.workload_advisor.backend import GpuLease, readiness
from vlm_dflash.workload_advisor.schema import protocol_payload, protocol_scope_hash, request_protocol_hash


class HFRuntimeRegressions(unittest.TestCase):
    def test_resolved_runtime_identity_keeps_frozen_manifest_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = fixtures._backend_for_loader(Path(tmp))
            workload = backend.workload
            expected = protocol_scope_hash(protocol_payload(
                backend.manifest, workload.rows_digest, workload.labels_digest,
            ))
            backend.resolved_target_identity = {"weights_sha256": "a" * 64}
            backend._build_protocol()
            self.assertEqual(backend.protocol_scope_hash, expected)
            row = workload.request_identity(workload.rows[0])
            first = request_protocol_hash(backend.protocol_payload, row)
            backend.resolved_target_identity = {"weights_sha256": "b" * 64}
            backend._build_protocol()
            self.assertEqual(backend.protocol_scope_hash, expected)
            self.assertNotEqual(request_protocol_hash(backend.protocol_payload, row), first)

    def test_actual_decoder_contract_reaches_measurement(self):
        """configure_verify_backend is an algorithm API, not a decoder API."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = fixtures.LifecycleTests()._backend(root)
            backend._torch = lambda: fixtures._FakeTorch()
            backend._preflight_target_artifacts = lambda: None
            backend._load_model = lambda: None
            backend._reference = lambda *args: [1]
            backend._run_once = lambda *args, **kwargs: {"error": None}
            checkpoint = root / "draft.pt"
            checkpoint.write_bytes(b"weights")
            backend.manifest["draft_candidates"] = [
                {"candidate_id": "ar", "method": "ar"},
                {"candidate_id": "sd", "method": "sd", "algorithm": "dflash", "checkpoint": str(checkpoint)},
            ]

            class Decoder:
                def to(self, **kwargs):
                    return self

                def eval(self):
                    return self

            algorithm = mock.Mock(name="algorithm")
            algorithm.name = "dflash"
            algorithm.load_decoder.return_value = (Decoder(), {"algorithm": "dflash"})
            backend._validate_checkpoint_metadata = lambda *args: {"algorithm": "dflash"}
            with mock.patch("vlm_dflash.workload_advisor.backend_hf.readiness", return_value=fixtures.LifecycleTests()._ready()), mock.patch(
                "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy", return_value={"busy": False}
            ), mock.patch("vlm_dflash.algorithms.get_algorithm", return_value=algorithm):
                result = backend.run(max_requests=1)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(len(result["records"]), 2)
            algorithm.configure_verify_backend.assert_called_once_with(False)

    def test_models_are_freed_before_unlock_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                backend = fixtures.LifecycleTests()._backend(root)
                torch = fixtures._FakeTorch()
                cleanup_events = []
                torch.cuda.empty_cache = lambda: cleanup_events.append("cache_released")
                backend._torch = lambda: torch
                backend._preflight_target_artifacts = lambda: None
                model_refs = []

                def load_model():
                    backend.model = fixtures._FakeModel()
                    model_refs.append(weakref.ref(backend.model))

                backend._load_model = load_model
                backend._load_drafts = lambda: []
                backend._reference = lambda *args: [1]
                backend._run_once = mock.Mock(side_effect=RuntimeError("measurement failed")) if fail else (
                    lambda *args, **kwargs: {"error": None}
                )
                unlocked = []

                class CheckedLease(GpuLease):
                    def release(lease):
                        unlocked.append(
                            backend.model is None and not backend.drafts
                            and all(ref() is None for ref in model_refs)
                            and cleanup_events == ["cache_released"]
                        )
                        super().release()

                with mock.patch("vlm_dflash.workload_advisor.backend_hf.readiness", return_value=fixtures.LifecycleTests()._ready()), mock.patch(
                    "vlm_dflash.workload_advisor.backend_hf.check_external_occupancy", return_value={"busy": False}
                ), mock.patch("vlm_dflash.workload_advisor.backend_hf.GpuLease", CheckedLease):
                    result = backend.run(max_requests=1)
                self.assertEqual(unlocked, [True])
                self.assertEqual(result["status"], "failed" if fail else "ok")
                self.assertIsNone(backend.model)

    def test_readiness_and_loader_resolve_manifest_relative_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = fixtures._backend_for_loader(root)
            fixtures.LoaderTests()._prepare_model_dir(root, backend)
            backend.workload.manifest_dir = root
            backend.manifest["target"]["model_path"] = "model"
            backend.manifest["target"]["processor"] = "model"
            (root / "draft.pt").write_bytes(b"draft")
            backend.manifest["draft_candidates"] = [
                {"candidate_id": "ar", "method": "ar"},
                {"candidate_id": "sd", "method": "sd", "checkpoint": "draft.pt"},
            ]
            env = {"torch": {"version": "test", "cuda_available": True}, "devices": [{"uuid": "GPU-test"}]}
            with mock.patch("vlm_dflash.workload_advisor.backend.detect_environment", return_value=env):
                ready = readiness(backend.manifest, manifest_dir=root)
            self.assertEqual(ready["status"], "ready", ready)
            backend._torch = lambda: fixtures._FakeTorch()
            loader = mock.Mock(return_value=(fixtures._FakeModel(), fixtures._FakeProcessor()))
            backend._load_pretrained_components = loader
            backend._load_model()
            self.assertEqual(loader.call_args.kwargs["model_path"], str(root / "model"))
            self.assertEqual(loader.call_args.kwargs["processor_path"], str(root / "model"))


if __name__ == "__main__":
    unittest.main()
