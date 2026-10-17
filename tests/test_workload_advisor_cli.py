from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "workload_advisor" / "mini"
FIXTURES = ROOT / "examples" / "workload_advisor" / "fixtures"
CLI = ROOT / "scripts" / "workload_advisor.py"


def _run(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args], capture_output=True, text=True, cwd=ROOT, timeout=timeout
    )


class CliTests(unittest.TestCase):
    def test_validate_exit_codes(self) -> None:
        good = _run("validate", "--manifest", str(MINI / "manifest.json"))
        self.assertEqual(good.returncode, 0, good.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "manifest.json"
            bad.write_text("{not json", encoding="utf-8")
            result = _run("validate", "--manifest", str(bad))
            self.assertNotEqual(result.returncode, 0)

    def test_report_replay_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            args = [
                "report",
                "--records",
                str(FIXTURES / "records.jsonl"),
                "--manifest",
                str(MINI / "manifest.json"),
            ]
            self.assertEqual(_run(*args, "--out-dir", str(first)).returncode, 0)
            self.assertEqual(_run(*args, "--out-dir", str(second)).returncode, 0)
            self.assertEqual((first / "decision.json").read_bytes(), (second / "decision.json").read_bytes())
            self.assertEqual((first / "summary.json").read_bytes(), (second / "summary.json").read_bytes())
            report = (first / "report.md").read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC FIXTURE", report)
            self.assertIn("insufficient_evidence", report)

    def test_import_legacy_is_aggregate_only_and_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "legacy.jsonl"
            imported = _run(
                "import-legacy", "--report", str(FIXTURES / "legacy_report.json"), "--out", str(records)
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            self.assertTrue(records.exists())
            out = Path(tmp) / "out"
            recommended = _run("recommend", "--records", str(records), "--out-dir", str(out))
            self.assertEqual(recommended.returncode, 0, recommended.stderr)
            decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["recommended_action"], "insufficient_evidence")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            candidate = summary["candidates"]["legacy"]
            self.assertIsNone(candidate["speed"]["ci95_low"])

    def test_share_package_defaults_to_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            result = _run(
                "report",
                "--records",
                str(FIXTURES / "records.jsonl"),
                "--manifest",
                str(MINI / "manifest.json"),
                "--out-dir",
                str(out),
                "--share",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "share" / "SHARE_MANIFEST.json").exists())
            for path in (out / "share").glob("*"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("Extract the invoice total", text)
                self.assertNotIn("/Users/Admin", text)


if __name__ == "__main__":
    unittest.main()
