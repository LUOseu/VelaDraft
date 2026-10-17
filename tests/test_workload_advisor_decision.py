from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlm_dflash.workload_advisor.decision import decide
from vlm_dflash.workload_advisor.records import aggregate_record_from_legacy, write_records
from vlm_dflash.workload_advisor.report import render_markdown, write_share_package
from vlm_dflash.workload_advisor.schema import sha256_json
from vlm_dflash.workload_advisor.stats import summarize

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "workload_advisor" / "mini"
FIXTURES = ROOT / "examples" / "workload_advisor" / "fixtures"


def _records(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest() -> dict:
    return json.loads((MINI / "manifest.json").read_text(encoding="utf-8"))


def _as_real(records: list[dict]) -> list[dict]:
    copy = json.loads(json.dumps(records))
    for record in copy:
        record["synthetic"] = False
    return copy


class DecisionTests(unittest.TestCase):
    def test_fixture_never_produces_real_recommendation(self) -> None:
        summary = summarize(_records("records.jsonl"), bucket_rules=_manifest()["bucket_rules"], include_synthetic=True)
        self.assertTrue(summary["n_synthetic_records"] > 0)
        summary["synthetic"] = True
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "insufficient_evidence")
        self.assertTrue(decision["synthetic"])
        excluded = {item["candidate_id"]: item for item in decision["excluded_candidates"]}
        self.assertIn("dflash-exact", excluded)
        self.assertTrue(any(reason["code"] == "synthetic_fixture" for reason in excluded["dflash-exact"]["reasons"]))

    def test_accelerate_when_real_and_exact(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "accelerate")
        self.assertEqual(decision["recommended_method"], "dflash-exact")
        self.assertIsNotNone(decision["protocol_hash"])
        excluded = {item["candidate_id"]: item for item in decision["excluded_candidates"]}
        self.assertIn("dfly-fast-wrong", excluded)
        codes = {reason["code"] for reason in excluded["dfly-fast-wrong"]["reasons"]}
        self.assertIn("correctness_failed", codes)
        self.assertIn("quality_failed", codes)

    def test_retain_ar_when_candidate_is_slower(self) -> None:
        records = _as_real(_records("records_ar_better.jsonl"))
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "retain_ar")
        self.assertIsNone(decision["recommended_method"])
        excluded = {item["candidate_id"]: item for item in decision["excluded_candidates"]}
        codes = {reason["code"] for item in excluded.values() for reason in item["reasons"]}
        self.assertIn("speedup_ci_not_above_threshold", codes)

    def test_fast_but_wrong_candidate_is_excluded_from_recommendation(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        decision = decide(summary)
        self.assertNotEqual(decision["recommended_method"], "dfly-fast-wrong")

    def test_aggregate_only_has_no_ci_and_is_insufficient(self) -> None:
        report = json.loads((FIXTURES / "legacy_report.json").read_text(encoding="utf-8"))
        record = aggregate_record_from_legacy(
            report,
            "textvqa",
            run_id="legacy",
            protocol_scope_hash=sha256_json({"legacy": True}),
            target_identity={"model_id": "legacy", "dtype": "bfloat16", "processor": "legacy", "revision": "1"},
            backend_identity={"name": "legacy", "version": "1", "device_topology": {}},
        )
        summary = summarize([record], bucket_rules=_manifest()["bucket_rules"])
        candidate = summary["candidates"]["legacy"]
        self.assertIsNone(candidate["speed"]["ci95_low"])
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "insufficient_evidence")
        codes = {reason["code"] for item in decision["excluded_candidates"] for reason in item["reasons"]}
        self.assertIn("aggregate_only", codes)

    def test_protocol_mismatch_blocks_pairing(self) -> None:
        records = _as_real(_records("records.jsonl"))
        records[0]["protocol_scope_hash"] = "f" * 64
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        self.assertTrue(summary["protocol_mismatch"])
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "insufficient_evidence")

    def test_decision_is_deterministic(self) -> None:
        records = _as_real(_records("records.jsonl"))
        first = decide(summarize(records, bucket_rules=_manifest()["bucket_rules"]))
        second = decide(summarize(records, bucket_rules=_manifest()["bucket_rules"]))
        self.assertEqual(first["decision_id"], second["decision_id"])

    def test_no_streaming_is_reported_as_null_not_fabricated(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        ar = summary["candidates"]["ar"]
        self.assertEqual(ar["latency"]["ttft_s"]["n"], 0)
        self.assertIsNone(ar["latency"]["ttft_s"]["mean"])
        self.assertEqual(ar["latency"]["itl"], "null_no_token_visibility")
        decision = decide(summary)
        report = render_markdown(summary, decision, manifest=_manifest())
        self.assertIn("null", report)
        self.assertIn("TTFT/ITL stay null", report)


class ReportTests(unittest.TestCase):
    def test_report_has_required_decision_fields_and_buckets(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        decision = decide(summary)
        report = render_markdown(summary, decision, manifest=_manifest())
        self.assertIn("protocol scope hash", report)
        self.assertIn("CI", report)
        self.assertIn("Excluded candidates", report)
        self.assertIn("insufficient_evidence", report)
        self.assertIn("dflash-exact", report)
        self.assertIn("## Bucket evidence", report)

    def test_share_package_removes_prompts_paths_and_answers(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = summarize(records, bucket_rules=_manifest()["bucket_rules"])
        decision = decide(summary)
        report = render_markdown(summary, decision, manifest=_manifest(), share=True)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_share_package(Path(tmp), records=records, summary=summary, decision=decision, report_md=report)
            self.assertTrue(manifest["share_package"])
            for path in Path(tmp).glob("*"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/Users/Admin", text)
                self.assertNotIn("Extract the invoice total", text)
                self.assertNotIn("synthetic/not-a-real-checkpoint", text)

    def test_synthetic_report_has_banner_and_explicit_label(self) -> None:
        summary = summarize(_records("records.jsonl"), bucket_rules=_manifest()["bucket_rules"])
        summary["synthetic"] = True
        summary["overall_status"] = "synthetic_fixture"
        decision = decide(summary)
        report = render_markdown(summary, decision, manifest=_manifest())
        self.assertIn("SYNTHETIC FIXTURE", report)
        self.assertIn("must not be copied", report)


if __name__ == "__main__":
    unittest.main()
