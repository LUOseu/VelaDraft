from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from vlm_dflash.workload_advisor.decision import decide
from vlm_dflash.workload_advisor.records import (
    ar_reference_reuse_error,
    build_latency,
    load_records,
    normalize_external_profile,
    write_records,
)
from vlm_dflash.workload_advisor.schema import canonical_json, validate_record
from vlm_dflash.workload_advisor.stats import summarize
from vlm_dflash.workload_advisor.workload import convert_sharegpt, load_workload, validation_report

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "examples" / "workload_advisor" / "mini"
FIXTURES = ROOT / "examples" / "workload_advisor" / "fixtures"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _copy_mini(tmp: Path) -> Path:
    target = tmp / "mini"
    shutil.copytree(MINI, target)
    return target


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict) -> None:
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class WorkloadValidationTests(unittest.TestCase):
    def test_example_manifest_validates(self) -> None:
        workload = load_workload(MINI / "manifest.json")
        self.assertTrue(workload.ok, workload.errors)
        report = validation_report(workload)
        self.assertEqual(report["counts"]["groups_by_split"], {"tuning": 6, "holdout": 24})
        self.assertEqual(report["labels_present"], 30)
        self.assertEqual(report["overall_status"], "ok")

    def test_bad_json_line_is_reported_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            with open(mini / "rows.jsonl", "a", encoding="utf-8") as handle:
                handle.write("{not json}\n")
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertFalse(workload.ok)
            self.assertTrue(any(error["code"] == "bad_json" and error.get("line") == 31 for error in workload.errors))

    def test_media_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            (Path(tmp) / "outside.png").write_bytes((mini / "media" / "grp-00.png").read_bytes())
            rows = _read_jsonl(mini / "rows.jsonl")
            rows[0]["media"] = [{"type": "image", "path": "../outside.png"}]
            _write_jsonl(mini / "rows.jsonl", rows)
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertFalse(workload.ok)
            self.assertTrue(any(error["code"] == "bad_media" and "escapes" in error["message"] for error in workload.errors))

    def test_missing_media_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            rows = _read_jsonl(mini / "rows.jsonl")
            rows[0]["media"] = [{"type": "image", "path": "missing.png"}]
            _write_jsonl(mini / "rows.jsonl", rows)
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertFalse(workload.ok)
            self.assertTrue(any(error["code"] == "bad_media" and "not found" in error["message"] for error in workload.errors))

    def test_duplicate_request_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            rows = _read_jsonl(mini / "rows.jsonl")
            rows.append(dict(rows[0]))
            _write_jsonl(mini / "rows.jsonl", rows)
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertTrue(any(error["code"] == "duplicate_request_id" for error in workload.errors))

    def test_group_split_leak_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            rows = _read_jsonl(mini / "rows.jsonl")
            leaked = dict(rows[0])
            leaked["request_id"] = "leaked-request"
            leaked["split"] = "holdout"
            rows.append(leaked)
            _write_jsonl(mini / "rows.jsonl", rows)
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertTrue(any(error["code"] == "group_split_leak" for error in workload.errors))

    def test_label_without_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            labels = _read_jsonl(mini / "labels.jsonl")
            labels.append({"request_id": "does-not-exist", "reference": "x"})
            _write_jsonl(mini / "labels.jsonl", labels)
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertTrue(any(error["code"] == "label_without_row" for error in workload.errors))

    def test_video_and_tool_rows_are_unsupported_not_silently_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest.pop("hashes")
            _write_manifest(mini, manifest)
            rows = _read_jsonl(mini / "rows.jsonl")
            video = dict(rows[0])
            video["request_id"] = "video-row"
            video["group_id"] = "video-group"
            video["media"] = [{"type": "video", "path": "grp-00.png"}]
            tool = dict(rows[1])
            tool["request_id"] = "tool-row"
            tool["group_id"] = "tool-group"
            tool["messages"] = [{"role": "tool", "content": "result"}]
            rows.extend([video, tool])
            _write_jsonl(mini / "rows.jsonl", rows)
            workload = load_workload(mini / "manifest.json", strict_counts=False)
            self.assertTrue(workload.ok, workload.errors)
            reasons = {item["request_id"]: item["reason"] for item in workload.unsupported}
            self.assertIn("video-row", reasons)
            self.assertIn("tool-row", reasons)
            self.assertNotIn("video-row", {row["request_id"] for row in workload.rows})
            self.assertNotIn("tool-row", {row["request_id"] for row in workload.rows})

    def test_unknown_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            manifest = _manifest(mini)
            manifest["schema_version"] = "veladraft-platform-v0"
            _write_manifest(mini, manifest)
            workload = load_workload(mini / "manifest.json")
            self.assertFalse(workload.ok)
            self.assertTrue(any(error["code"] == "unknown_schema_version" for error in workload.errors))

    def test_rows_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mini = _copy_mini(Path(tmp))
            with open(mini / "rows.jsonl", "a", encoding="utf-8") as handle:
                handle.write("\n")
            workload = load_workload(mini / "manifest.json")
            self.assertFalse(workload.ok)
            self.assertTrue(any(error["code"] == "rows_hash_mismatch" for error in workload.errors))

    def test_sharegpt_conversion_is_deterministic_and_documents_last_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "conv-1",
                        "conversations": [
                            {"from": "human", "value": "first question"},
                            {"from": "gpt", "value": "first answer"},
                            {"from": "human", "value": "second question"},
                        ],
                        "images": ["conv-1.png"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            first = convert_sharegpt(source, Path(tmp) / "out1", last_turn_only=True)
            second = convert_sharegpt(source, Path(tmp) / "out2", last_turn_only=True)
            self.assertEqual(first["rows_sha256"], second["rows_sha256"])
            self.assertTrue(first["semantics"]["legacy_last_turn_only"])
            self.assertFalse(first["semantics"]["answers_included"])
            rows = _read_jsonl(Path(tmp) / "out1" / "rows.jsonl")
            self.assertEqual(rows[0]["messages"], [{"role": "user", "content": "second question"}])

    def test_sharegpt_single_turn_answer_is_private_label_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "single-1",
                        "conversations": [
                            {"from": "human", "value": "What is the total?"},
                            {"from": "gpt", "value": "PRIVATE_LABEL_TOTAL_123"},
                        ],
                        "images": ["single-1.png"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = convert_sharegpt(source, Path(tmp) / "out")
            rows = _read_jsonl(Path(tmp) / "out" / "rows.jsonl")
            self.assertEqual(rows[0]["messages"], [{"role": "user", "content": "What is the total?"}])
            self.assertFalse(report["semantics"]["answers_included"])
            self.assertEqual(report["semantics"]["answer_location"], "private_labels_sidecar")
            labels = _read_jsonl(Path(tmp) / "out" / "labels.jsonl")
            self.assertEqual({row["request_id"] for row in labels}, {row["request_id"] for row in rows})
            self.assertEqual(labels[0]["reference"], "PRIVATE_LABEL_TOTAL_123")
            self.assertNotIn("PRIVATE_LABEL_TOTAL_123", json.dumps(rows))

    def test_sharegpt_multiturn_preserves_history_but_not_terminal_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "multi-1",
                        "conversations": [
                            {"from": "human", "value": "first question"},
                            {"from": "gpt", "value": "first answer"},
                            {"from": "human", "value": "second question"},
                            {"from": "gpt", "value": "PRIVATE_SECOND_ANSWER"},
                        ],
                        "images": ["multi-1.png"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = convert_sharegpt(source, Path(tmp) / "out")
            rows = _read_jsonl(Path(tmp) / "out" / "rows.jsonl")
            self.assertEqual(
                rows[0]["messages"],
                [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                ],
            )
            self.assertTrue(report["semantics"]["history_preserved"])
            self.assertTrue(report["semantics"]["terminal_answer_removed"])
            labels = _read_jsonl(Path(tmp) / "out" / "labels.jsonl")
            self.assertEqual(labels[0]["reference"], "PRIVATE_SECOND_ANSWER")
            self.assertNotIn("PRIVATE_SECOND_ANSWER", json.dumps(rows))


class RunRecordValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = json.loads((FIXTURES / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])

    def test_fixture_record_is_valid(self) -> None:
        self.assertEqual(validate_record(self.record), [])

    def test_missing_required_key_is_rejected(self) -> None:
        record = dict(self.record)
        record.pop("resource_usage")
        issues = validate_record(record)
        self.assertTrue(any(issue["code"] == "missing_key" and issue["path"] == "resource_usage" for issue in issues))

    def test_ar_draft_identity_needs_reason(self) -> None:
        record = dict(self.record)
        record["missing_reasons"] = {}
        issues = validate_record(record)
        self.assertTrue(any(issue["code"] == "missing_reason" for issue in issues))

    def test_relaxed_policy_cannot_claim_exact(self) -> None:
        record = dict(self.record)
        record["acceptance_policy"] = "relaxed"
        issues = validate_record(record)
        self.assertTrue(any(issue["code"] == "relaxed_not_exact" for issue in issues))

    def test_exact_status_with_failures_is_rejected(self) -> None:
        record = json.loads(json.dumps(self.record))
        record["correctness"]["failed_request_ids"] = ["req-x"]
        issues = validate_record(record)
        self.assertTrue(any(issue["code"] == "exact_with_failures" for issue in issues))

    def test_acceptance_length_must_match_counts(self) -> None:
        record = json.loads(
            [line for line in (FIXTURES / "records.jsonl").read_text(encoding="utf-8").splitlines() if '"candidate_id": "dflash-exact"' in line][0]
        )
        record["raw_counts"]["acceptance_length_draft"] = 99.0
        issues = validate_record(record)
        self.assertTrue(any(issue["code"] == "al_draft_mismatch" for issue in issues))

    def test_seconds_suffix_is_required(self) -> None:
        record = json.loads(json.dumps(self.record))
        record["latency"]["wall_seconds"] = 1.0
        issues = validate_record(record)
        self.assertTrue(any(issue["code"] == "seconds_suffix" for issue in issues))

    def test_ar_reference_reuse_requires_protocol_match(self) -> None:
        reference = json.loads(json.dumps(self.record))
        record = json.loads(json.dumps(self.record))
        self.assertIsNone(ar_reference_reuse_error(record, reference))
        record["protocol_hash"] = "0" * 64
        self.assertIn("protocol_hash mismatch", ar_reference_reuse_error(record, reference))

    def test_zero_over_n_is_accepted_and_keeps_denominator(self) -> None:
        reference = json.loads(json.dumps(self.record))
        reference["correctness"]["failed_request_ids"] = []
        reference["correctness"]["n_requests"] = 24
        self.assertEqual(validate_record(reference), [])
        self.assertEqual(reference["correctness"]["n_requests"], 24)

    def test_bad_json_record_line_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text("{bad}\n" + canonical_json(self.record) + "\n", encoding="utf-8")
            records, issues = load_records(path)
            self.assertEqual(len(records), 1)
            self.assertTrue(any(issue["code"] == "bad_json" for issue in issues))


class ExternalProfileAndLatencyTests(unittest.TestCase):
    def test_structured_latency_round_trips(self) -> None:
        record = json.loads((FIXTURES / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
        latency, reasons = build_latency(
            e2e_s=1.5,
            itl_s={"mean": 0.05, "p50": 0.04, "p95": 0.08, "n": 5},
            chunk_gap_s={"mean": 0.10, "p50": 0.09, "p95": 0.20, "n": 4},
            itl_reason="token_visibility_available",
        )
        record["latency"] = latency
        record["missing_reasons"].pop("latency.itl_s", None)
        record["missing_reasons"].update(reasons)
        self.assertEqual(validate_record(record), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            write_records(path, [record])
            loaded, issues = load_records(path)
            self.assertEqual(issues, [])
            self.assertEqual(loaded[0]["latency"]["itl_s"], latency["itl_s"])
            self.assertEqual(loaded[0]["latency"]["chunk_gap_s"], latency["chunk_gap_s"])

    def test_structured_latency_rejects_bad_shapes_and_values(self) -> None:
        record = json.loads((FIXTURES / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
        record["latency"]["itl_s"] = {"mean": -0.1, "n": -2, "unknown": 1}
        issues = validate_record(record)
        codes = {issue["code"] for issue in issues}
        self.assertIn("invalid_latency_distribution", codes)
        self.assertIn("invalid_seconds", codes)
        record["latency"]["itl_s"] = {"mean": math.nan, "n": 1}
        issues = validate_record(record)
        self.assertTrue(any(issue["path"] == "latency.itl_s.mean" for issue in issues))

    def test_known_serving_profile_normalizes_without_inventing_evidence(self) -> None:
        source = json.loads((FIXTURES / "serving_record.json").read_text(encoding="utf-8"))
        record = normalize_external_profile(source)
        self.assertEqual(record["measurement_scope"], "server_endpoint")
        self.assertEqual(record["latency"]["e2e_s"], 3.0)
        self.assertEqual(record["raw_counts"]["emitted_tokens"], 24)
        self.assertEqual(record["raw_counts"]["proposed_draft_tokens"], 50)
        self.assertEqual(record["raw_counts"]["accepted_draft_tokens"], 30)
        self.assertEqual(record["raw_counts"]["count_source"], "serving_raw_counts")
        self.assertIsNone(record["resource_usage"]["all_allocated_gpu_hours"])
        self.assertEqual(record["serving_gpu_hours"]["kind"], "request_or_window_exposure")
        self.assertEqual(record["split"], None)
        self.assertEqual(record["missing_reasons"]["split"], "missing_holdout_assignment")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "serving.jsonl"
            write_records(path, [record])
            loaded, issues = load_records(path)
            self.assertEqual(issues, [])
            summary = summarize(loaded)
            decision = decide(summary)
            self.assertFalse(summary["assignment_verified"])
            self.assertEqual(decision["recommended_action"], "insufficient_evidence")

    def test_generate_only_total_is_never_relabelled_e2e(self) -> None:
        source = json.loads((FIXTURES / "serving_record.json").read_text(encoding="utf-8"))
        source["measurement_scope"]["timing_boundary"] = "generate_only_decode"
        record = normalize_external_profile(source)
        self.assertIsNone(record["latency"]["e2e_s"])
        self.assertEqual(record["missing_reasons"]["latency.e2e_s"], "serving_boundary_is_not_end_to_end")

    def test_unknown_surface_and_ambiguous_boundary_are_rejected(self) -> None:
        source = json.loads((FIXTURES / "serving_record.json").read_text(encoding="utf-8"))
        source["measurement_scope"]["surface"] = "mystery_transport"
        with self.assertRaisesRegex(ValueError, "unsupported serving measurement surface"):
            normalize_external_profile(source)
        source = json.loads((FIXTURES / "serving_record.json").read_text(encoding="utf-8"))
        source["measurement_scope"].pop("timing_boundary")
        with self.assertRaisesRegex(ValueError, "timing_boundary"):
            normalize_external_profile(source)


if __name__ == "__main__":
    unittest.main()
