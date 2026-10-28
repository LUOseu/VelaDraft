from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from vlm_dflash.workload_advisor.backend import (
    CAPABILITIES,
    GpuBusy,
    GpuLedger,
    GpuLease,
    capabilities_document,
    readiness,
)
from vlm_dflash.workload_advisor.plan import build_plan
from vlm_dflash.workload_advisor.workload import load_workload, validation_report

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "workload_advisor" / "mini"


def _manifest() -> dict:
    return json.loads((MINI / "manifest.json").read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    unittest.main()
