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


def _assignment() -> dict[str, dict]:
    rows = [json.loads(line) for line in (MINI / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        row["request_id"]: {
            "group_id": row["group_id"],
            "task": row["task"],
            "split": row["split"],
        }
        for row in rows
    }


def _labels() -> set[str]:
    return {
        json.loads(line)["request_id"]
        for line in (MINI / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _summarize(records: list[dict], *, assignment=None, labels=None, include_synthetic=False) -> dict:
    return summarize(
        records,
        bucket_rules=_manifest()["bucket_rules"],
        assignment=_assignment() if assignment is None else assignment,
        expected_labels=_labels() if labels is None else labels,
        manifest=_manifest(),
        include_synthetic=include_synthetic,
    )


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
        summary = _summarize(records)
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
        summary = _summarize(records)
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "retain_ar")
        self.assertIsNone(decision["recommended_method"])
        excluded = {item["candidate_id"]: item for item in decision["excluded_candidates"]}
        codes = {reason["code"] for item in excluded.values() for reason in item["reasons"]}
        self.assertIn("speedup_ci_not_above_threshold", codes)

    def test_fast_but_wrong_candidate_is_excluded_from_recommendation(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = _summarize(records)
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
        summary = _summarize(records)
        self.assertTrue(summary["protocol_mismatch"])
        decision = decide(summary)
        self.assertEqual(decision["recommended_action"], "insufficient_evidence")

    def test_decision_is_deterministic(self) -> None:
        records = _as_real(_records("records.jsonl"))
        first = decide(_summarize(records))
        second = decide(_summarize(records))
        self.assertEqual(first["decision_id"], second["decision_id"])

    def test_no_streaming_is_reported_as_null_not_fabricated(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = _summarize(records)
        ar = summary["candidates"]["ar"]
        self.assertEqual(ar["latency"]["ttft_s"]["n"], 0)
        self.assertIsNone(ar["latency"]["ttft_s"]["mean"])
        self.assertEqual(ar["latency"]["itl"], "null_no_token_visibility")
        self.assertEqual(
            ar["correctness"]["paired_coverage"]["requests"],
            ar["correctness"]["n_requests"],
        )
        decision = decide(summary)
        report = render_markdown(summary, decision, manifest=_manifest())
        self.assertIn("null", report)
        self.assertIn("TTFT/ITL stay null", report)


class ReportTests(unittest.TestCase):
    def test_report_has_required_decision_fields_and_buckets(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = _summarize(records)
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
        summary = _summarize(records)
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
        summary = _summarize(_records("records.jsonl"), include_synthetic=True)
        summary["synthetic"] = True
        summary["overall_status"] = "synthetic_fixture"
        decision = decide(summary)
        report = render_markdown(summary, decision, manifest=_manifest())
        self.assertIn("SYNTHETIC FIXTURE", report)
        self.assertIn("must not be copied", report)

    def test_share_package_structural_allowlist_removes_nested_sentinels(self) -> None:
        records = _as_real(_records("records.jsonl"))
        summary = _summarize(records)
        decision = decide(summary)
        private_record = json.loads(json.dumps(records[0]))
        private_record["target_identity"]["processor"] = {"local_path": "/srv/ACME_PRIVATE_PROCESSOR"}
        private_record["quality"] = {
            "evaluated": True,
            "score": 1.0,
            "threshold": 1.0,
            "status": "pass",
            "reference": "ACME_PRIVATE_ANSWER",
            "prediction": "ACME_PRIVATE_PREDICTION",
        }
        private_record["error"] = {
            "type": "timeout",
            "stage": "generation",
            "message": "/mnt/ACME_PRIVATE_INPUT.png",
        }
        private_record["artifacts"] = {
            "weights_sha256": "a" * 64,
            "local_log": "C:\\ACME_PRIVATE\\run.log",
        }
        private_record["missing_reasons"]["latency.ttft_s"] = "/srv/ACME_PRIVATE_PROMPT"
        summary["run"] = {
            "status": "ok",
            "readiness": {"status": "ok", "target_path": "/srv/ACME_PRIVATE_TARGET"},
            "note": "ACME_PRIVATE_FREE_TEXT",
        }
        summary["issues"] = [
            {
                "code": "protocol_mismatch",
                "message": "/mnt/ACME_PRIVATE_INPUT.png",
                "path": "C:\\ACME_PRIVATE\\manifest.json",
            }
        ]
        decision["reasons"] = ["/srv/ACME_PRIVATE_REASON"]
        with tempfile.TemporaryDirectory() as tmp:
            write_share_package(
                Path(tmp),
                records=[private_record],
                summary=summary,
                decision=decision,
                report_md="/srv/ACME_PRIVATE_REPORT",
            )
            for path in Path(tmp).iterdir():
                self.assertNotIn("ACME_PRIVATE", path.read_text(encoding="utf-8"), str(path))


class HoldoutPairingTests(unittest.TestCase):
    def test_equal_speed_baseline_does_not_accelerate(self) -> None:
        records = _as_real(_records("records.jsonl"))
        ar = {
            (record["request_id"], record["session_id"], record["repeat_index"]): record
            for record in records
            if record["candidate_id"] == "ar"
        }
        for record in records:
            if record["candidate_id"] == "dflash-exact":
                key = (record["request_id"], record["session_id"], record["repeat_index"])
                record["latency"]["e2e_s"] = ar[key]["latency"]["e2e_s"]
        summary = _summarize(records)
        decision = decide(summary)
        self.assertNotEqual(decision["recommended_action"], "accelerate")
        candidate = summary["candidates"]["dflash-exact"]
        self.assertLessEqual(candidate["speed"]["ci95_low"], 1.05)
        self.assertEqual(candidate["n_pairs"], 72)

    def test_deleting_two_candidate_sessions_is_not_complete_evidence(self) -> None:
        records = _as_real(_records("records.jsonl"))
        missing_sessions = {"fixture-s1", "fixture-s2"}
        records = [
            record
            for record in records
            if not (record["candidate_id"] == "dflash-exact" and record["session_id"] in missing_sessions)
        ]
        summary = _summarize(records)
        decision = decide(summary)
        candidate = summary["candidates"]["dflash-exact"]
        self.assertEqual(len(candidate["pairing"]["missing_candidate"]), 48)
        self.assertFalse(candidate["correctness"]["gate"])
        self.assertFalse(candidate["correctness"]["pairing_complete"])
        self.assertNotEqual(decision["recommended_action"], "accelerate")

    def test_duplicate_ar_rows_are_rejected_not_inflated(self) -> None:
        records = _as_real(_records("records.jsonl"))
        duplicates = [json.loads(json.dumps(record)) for record in records if record["candidate_id"] == "ar"]
        summary = _summarize(records + duplicates)
        decision = decide(summary)
        self.assertGreater(summary["duplicate_records"], 0)
        self.assertGreater(len(summary["pairing"]["duplicate_ids"]), 0)
        self.assertEqual(summary["candidates"]["dflash-exact"]["n_pairs"], 0)
        self.assertNotEqual(decision["recommended_action"], "accelerate")

    def test_request_and_hash_mismatch_breaks_pairing(self) -> None:
        records = _as_real(_records("records.jsonl"))
        target = next(
            record
            for record in records
            if record["candidate_id"] == "dflash-exact"
            and record["request_id"] == "req-06"
            and record["session_id"] == "fixture-s0"
        )
        target["correctness"]["input_sha256"] = "f" * 64
        summary = _summarize(records)
        decision = decide(summary)
        candidate = summary["candidates"]["dflash-exact"]
        self.assertTrue(candidate["pairing"]["protocol_mismatch"])
        self.assertRegex(candidate["pairing"]["protocol_mismatch"][0]["reason"], "input_sha256_mismatch")
        self.assertEqual(candidate["n_pairs"], 71)
        self.assertNotEqual(decision["recommended_action"], "accelerate")

    def test_all_tuning_has_no_recommendation(self) -> None:
        records = _as_real(_records("records.jsonl"))
        assignment = _assignment()
        for record in records:
            record["split"] = "tuning"
        for value in assignment.values():
            value["split"] = "tuning"
        summary = _summarize(records, assignment=assignment)
        decision = decide(summary)
        self.assertEqual(summary["n_holdout_records"], 0)
        self.assertEqual(decision["recommended_action"], "insufficient_evidence")
        self.assertNotEqual(decision["recommended_action"], "accelerate")

    def test_tuning_additions_do_not_change_holdout_ci(self) -> None:
        records = _as_real(_records("records.jsonl"))
        baseline = _summarize(records)
        assignment = _assignment()
        tuning = []
        for record in records:
            copy = json.loads(json.dumps(record))
            original_request = copy["request_id"]
            copy["request_id"] = f"tune-{original_request}"
            copy["group_id"] = f"tune-{copy['group_id']}"
            copy["session_id"] = f"tune-{copy['session_id']}"
            copy["split"] = "tuning"
            tuning.append(copy)
            assignment[copy["request_id"]] = {
                "group_id": copy["group_id"],
                "task": copy["task"],
                "split": "tuning",
            }
        augmented = _summarize(records + tuning, assignment=assignment)
        before = baseline["candidates"]["dflash-exact"]["speed"]
        after = augmented["candidates"]["dflash-exact"]["speed"]
        self.assertEqual(before["n_groups"], after["n_groups"])
        self.assertEqual(before["n_pairs"], after["n_pairs"])
        self.assertEqual(before["ci95_low"], after["ci95_low"])
        self.assertEqual(before["ci95_high"], after["ci95_high"])
        self.assertGreater(augmented["candidates"]["dflash-exact"]["tuning"]["n_pairs"], 0)


class GateCoverageTests(unittest.TestCase):
    def test_one_labeled_group_cannot_pass_quality_and_sessions_do_not_inflate(self) -> None:
        records = _as_real(_records("records.jsonl"))
        labels = {"req-06"}
        summary = _summarize(records, labels=labels)
        decision = decide(summary)
        quality = summary["candidates"]["dflash-exact"]["quality"]
        self.assertEqual(quality["status"], "conditional")
        self.assertEqual(quality["n"], 1)
        self.assertEqual(quality["coverage"]["groups"], {"evaluated": 1, "denominator": 24})
        self.assertNotEqual(decision["recommended_action"], "accelerate")
        self.assertEqual(decision["recommended_action"], "conditional_accelerate")

    def test_task_with_no_labels_blocks_quality_pass(self) -> None:
        records = _as_real(_records("records.jsonl"))
        rows = [json.loads(line) for line in (MINI / "rows.jsonl").read_text().splitlines() if line.strip()]
        labels = {row["request_id"] for row in rows if row["task"] != "chart_qa"}
        summary = _summarize(records, labels=labels)
        decision = decide(summary)
        quality = summary["candidates"]["dflash-exact"]["quality"]
        self.assertEqual(quality["status"], "conditional")
        self.assertEqual(quality["coverage"]["tasks"], {"evaluated": 2, "denominator": 3})
        self.assertNotEqual(decision["recommended_action"], "accelerate")

    def test_failed_ar_reference_cannot_support_candidate_exactness(self) -> None:
        records = _as_real(_records("records.jsonl"))
        for record in records:
            if record["candidate_id"] == "ar":
                record["correctness_status"] = "failed"
                record["error"] = {"code": "timeout", "stage": "generation"}
                record["correctness"]["failed_request_ids"] = [record["request_id"]]
        summary = _summarize(records)
        decision = decide(summary)
        self.assertFalse(summary["candidates"]["ar"]["gates"]["correctness"])
        self.assertEqual(summary["candidates"]["dflash-exact"]["n_pairs"], 0)
        self.assertTrue(summary["candidates"]["dflash-exact"]["pairing"]["failed_ar"])
        self.assertEqual(decision["recommended_action"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
