"""Compare AR and N-Gram offline token transcripts.

The gate is intentionally strict: expected request x repeat pairs come from
the frozen fixture or run plan, never from whichever records happen to exist in
the candidate document.  Missing, duplicate, extra, mismatched-identity and
failed records are reported as coverage errors and can never produce a PASS.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .contracts import write_json_atomic

_REPEAT_RE = re.compile(r"^(?P<base>.+)-r(?P<repeat>\d+)$")
_SUCCESS_FINISH_REASONS = {"stop", "length"}
_FAILED_CORRECTNESS = {"failed", "unsupported"}


def _split_request_id(request_id: str) -> tuple[str, Optional[int]]:
    match = _REPEAT_RE.match(str(request_id))
    if not match:
        return str(request_id), None
    return match.group("base"), int(match.group("repeat"))


def _record_key(record: dict[str, Any]) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Return ``(base_request_id, repeat, error)`` for one offline record."""

    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        return None, None, "record has no non-empty request_id"
    base, id_repeat = _split_request_id(request_id)
    field_repeat = record.get("repeat_index")
    if field_repeat is not None:
        if not isinstance(field_repeat, int) or field_repeat < 0:
            return base, None, f"{request_id}: repeat_index must be a non-negative int"
        if id_repeat is not None and id_repeat != field_repeat:
            return base, None, (
                f"{request_id}: request-id repeat {id_repeat} does not match "
                f"repeat_index {field_repeat}"
            )
        return base, field_repeat, None
    if id_repeat is not None:
        return base, id_repeat, None
    return base, 0, None


def _expected_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = list(manifest.get("requests", [])) + list(manifest.get("diagnostic", []))
    expected: dict[str, dict[str, Any]] = {}
    for entry in entries:
        request_id = entry.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("manifest fixture has no request_id")
        if request_id in expected:
            raise ValueError(f"duplicate request_id in manifest: {request_id}")
        expected[request_id] = {
            "category": entry.get("category"),
            "prompt_sha256": entry.get("prompt_sha256"),
            "image_sha256": entry.get("image_sha256"),
        }
    if not expected:
        raise ValueError("manifest contains no correctness fixtures")
    return expected


def _expected_from_ar(ar_records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    expected: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for record in ar_records:
        base, _, error = _record_key(record)
        if error or base is None:
            errors.append(error or "invalid AR record key")
            continue
        identity = {
            "category": record.get("category"),
            "prompt_sha256": record.get("prompt_sha256"),
            "image_sha256": record.get("image_sha256"),
        }
        previous = expected.get(base)
        if previous is None:
            expected[base] = identity
            continue
        if previous != identity:
            errors.append(f"{base}: AR repeats disagree on fixture identity")
    return expected, errors


def _expected_repeats(ar_doc: dict[str, Any], ngram_docs: list[dict[str, Any]], explicit: Optional[list[int]]) -> list[int]:
    if explicit:
        return sorted({int(value) for value in explicit})
    for doc in [ar_doc, *ngram_docs]:
        value = (doc.get("meta") or {}).get("repeats")
        if isinstance(value, int) and value >= 1:
            return list(range(value))
        if isinstance(value, list) and value and all(isinstance(item, int) and item >= 0 for item in value):
            return sorted(set(value))
    observed = set()
    for record in ar_doc.get("records", []):
        _, repeat, error = _record_key(record)
        if not error and repeat is not None:
            observed.add(repeat)
    return sorted(observed) if observed else [0]


def _index_records(
    records: list[dict[str, Any]],
    side: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], int], list[str]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: dict[tuple[str, int], int] = defaultdict(int)
    errors: list[str] = []
    for record in records:
        base, repeat, error = _record_key(record)
        if error or base is None or repeat is None:
            errors.append(f"{side}: {error or 'invalid record key'}")
            continue
        key = (base, repeat)
        if key in indexed:
            duplicates[key] += 1
        else:
            indexed[key] = record
    return indexed, dict(duplicates), errors


def _output_errors(record: dict[str, Any], side: str, key: tuple[str, int]) -> list[str]:
    errors: list[str] = []
    label = f"{side}:{key[0]}-r{key[1]}"
    if record.get("error"):
        errors.append(f"{label}: generation error: {record.get('error')}")
    status = record.get("correctness_status")
    if status in _FAILED_CORRECTNESS:
        errors.append(f"{label}: correctness_status={status}")
    finish = record.get("finish_reason")
    if finish not in _SUCCESS_FINISH_REASONS:
        errors.append(f"{label}: non-success finish_reason {finish!r}")
    tokens = record.get("token_ids")
    if not isinstance(tokens, list) or any(not isinstance(token, int) or token < 0 for token in tokens):
        errors.append(f"{label}: token_ids must be a list of non-negative ints")
    if not isinstance(record.get("text"), str):
        errors.append(f"{label}: text must be a string")
    return errors


def _identity_errors(
    expected: dict[str, dict[str, Any]],
    record: dict[str, Any],
    side: str,
    key: tuple[str, int],
) -> list[str]:
    errors: list[str] = []
    label = f"{side}:{key[0]}-r{key[1]}"
    want = expected.get(key[0])
    if want is None:
        return [f"{label}: request is not in the expected fixture set"]
    for field in ("category", "prompt_sha256", "image_sha256"):
        expected_value = want.get(field)
        actual = record.get(field)
        if expected_value is not None and actual != expected_value:
            errors.append(f"{label}: {field} mismatch ({actual!r} != {expected_value!r})")
    return errors


def _pairwise(
    indexed: dict[tuple[str, int], dict[str, Any]],
    expected_keys: set[tuple[str, int]],
    repeats: list[int],
) -> dict[str, Any]:
    counts = {"pairs": 0, "exact": 0, "finish_reason_exact": 0}
    for i, left_repeat in enumerate(repeats):
        for right_repeat in repeats[i + 1 :]:
            for base in sorted({base for base, _ in expected_keys}):
                left_key = (base, left_repeat)
                right_key = (base, right_repeat)
                left = indexed.get(left_key)
                right = indexed.get(right_key)
                if left is None or right is None:
                    continue
                counts["pairs"] += 1
                counts["exact"] += int(left.get("token_ids") == right.get("token_ids"))
                counts["finish_reason_exact"] += int(left.get("finish_reason") == right.get("finish_reason"))
    for key in ("exact", "finish_reason_exact"):
        counts[key + "_rate"] = counts[key] / counts["pairs"] if counts["pairs"] else None
    counts["rate"] = counts["exact_rate"]
    expected_pair_count = len(expected_keys) * (len(repeats) - 1) // 2
    counts["expected_pairs"] = expected_pair_count
    counts["complete"] = counts["pairs"] == expected_pair_count
    return counts


def _compare_one(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_tokens = reference.get("token_ids") or []
    cand_tokens = candidate.get("token_ids") or []
    first = None
    for index in range(min(len(ref_tokens), len(cand_tokens))):
        if ref_tokens[index] != cand_tokens[index]:
            first = index
            break
    if first is None and len(ref_tokens) != len(cand_tokens):
        first = min(len(ref_tokens), len(cand_tokens))
    return {
        "exact": ref_tokens == cand_tokens,
        "ref_len": len(ref_tokens),
        "cand_len": len(cand_tokens),
        "first_divergence": first,
        "finish_reason_match": reference.get("finish_reason") == candidate.get("finish_reason"),
        "text_exact": reference.get("text") == candidate.get("text"),
    }


def compare_documents(
    ar_doc: dict[str, Any],
    ngram_docs: list[dict[str, Any]],
    *,
    manifest: Optional[dict[str, Any]] = None,
    expected_repeats: Optional[list[int]] = None,
) -> dict[str, Any]:
    ar_records = list(ar_doc.get("records", []))
    ngram_records = [record for doc in ngram_docs for record in doc.get("records", [])]
    repeats = _expected_repeats(ar_doc, ngram_docs, expected_repeats)
    repeat_set = set(repeats)

    coverage_errors: list[str] = []
    if manifest is not None:
        expected = _expected_from_manifest(manifest)
        expected_source = "manifest"
    elif isinstance(ar_doc.get("run_plan"), dict) and ar_doc["run_plan"].get("requests"):
        expected = _expected_from_manifest(ar_doc["run_plan"])
        expected_source = "run_plan"
    else:
        expected, identity_errors = _expected_from_ar(ar_records)
        expected_source = "ar_observed"
        coverage_errors.extend(identity_errors)

    expected_keys = {
        (request_id, repeat)
        for request_id in expected
        for repeat in repeats
    }
    ar_index, ar_duplicates, ar_index_errors = _index_records(ar_records, "ar")
    sd_index, sd_duplicates, sd_index_errors = _index_records(ngram_records, "ngram")
    coverage_errors.extend(ar_index_errors)
    coverage_errors.extend(sd_index_errors)

    ar_keys = set(ar_index)
    sd_keys = set(sd_index)
    missing_ar = sorted(expected_keys - ar_keys, key=lambda item: (item[1], item[0]))
    missing_sd = sorted(expected_keys - sd_keys, key=lambda item: (item[1], item[0]))
    extra_ar = sorted(ar_keys - expected_keys, key=lambda item: (item[1], item[0]))
    extra_sd = sorted(sd_keys - expected_keys, key=lambda item: (item[1], item[0]))
    unexpected_repeats = sorted({repeat for _, repeat in (ar_keys | sd_keys) if repeat not in repeat_set})
    for key in missing_ar:
        coverage_errors.append(f"ar: missing expected pair {key[0]}-r{key[1]}")
    for key in missing_sd:
        coverage_errors.append(f"ngram: missing expected pair {key[0]}-r{key[1]}")
    for key in extra_ar:
        coverage_errors.append(f"ar: extra pair outside expected set {key[0]}-r{key[1]}")
    for key in extra_sd:
        coverage_errors.append(f"ngram: extra pair outside expected set {key[0]}-r{key[1]}")
    for key, count in sorted(ar_duplicates.items()):
        coverage_errors.append(f"ar: duplicate pair {key[0]}-r{key[1]} ({count + 1} records)")
    for key, count in sorted(sd_duplicates.items()):
        coverage_errors.append(f"ngram: duplicate pair {key[0]}-r{key[1]} ({count + 1} records)")
    for repeat in unexpected_repeats:
        coverage_errors.append(f"unexpected repeat index {repeat} outside {repeats}")

    ar_invalid: list[str] = []
    sd_invalid: list[str] = []
    identity_mismatches: list[str] = []
    for key in sorted(expected_keys & ar_keys, key=lambda item: (item[1], item[0])):
        record = ar_index[key]
        ar_invalid.extend(_output_errors(record, "ar", key))
        identity_mismatches.extend(_identity_errors(expected, record, "ar", key))
    for key in sorted(expected_keys & sd_keys, key=lambda item: (item[1], item[0])):
        record = sd_index[key]
        sd_invalid.extend(_output_errors(record, "ngram", key))
        identity_mismatches.extend(_identity_errors(expected, record, "ngram", key))

    ar_ar = _pairwise(ar_index, expected_keys, repeats)
    sd_ar = {"pairs": 0, "exact": 0, "sequence_exact": 0, "finish_reason_exact": 0}
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "exact": 0})
    details: list[dict[str, Any]] = []
    failed_ids: list[str] = []
    for key in sorted(expected_keys, key=lambda item: (item[1], item[0])):
        reference = ar_index.get(key)
        candidate = sd_index.get(key)
        if reference is None or candidate is None:
            failed_ids.append(f"{key[0]}-r{key[1]}")
            continue
        diff = _compare_one(reference, candidate)
        sd_ar["pairs"] += 1
        sd_ar["exact"] += int(diff["exact"])
        sd_ar["sequence_exact"] += int(diff["exact"])
        sd_ar["finish_reason_exact"] += int(diff["finish_reason_match"])
        category = expected[key[0]].get("category") or candidate.get("category") or "unknown"
        categories[category]["pairs"] += 1
        categories[category]["exact"] += int(diff["exact"])
        if not diff["exact"] or not diff["finish_reason_match"] or candidate.get("error"):
            failed_ids.append(f"{key[0]}-r{key[1]}")
        if not diff["exact"] or not diff["finish_reason_match"]:
            details.append(
                {
                    "repeat": key[1],
                    "request_id": key[0],
                    "category": category,
                    **diff,
                }
            )
    for counts in (sd_ar,):
        for key in ("exact", "sequence_exact", "finish_reason_exact"):
            counts[key + "_rate"] = counts[key] / counts["pairs"] if counts["pairs"] else None
        counts["rate"] = counts["exact_rate"]
    sd_ar["expected_pairs"] = len(expected_keys)
    sd_ar["complete"] = sd_ar["pairs"] == len(expected_keys)
    for category, counts in categories.items():
        counts["rate"] = counts["exact"] / counts["pairs"] if counts["pairs"] else None

    diagnostic_keys = {key for key in expected_keys if expected[key[0]].get("category") == "diagnostic"}
    diagnostics = [sd_index[key] for key in sorted(diagnostic_keys) if key in sd_index]
    diagnostic_proposed = sum(int(record["raw_counts"].get("proposed") or 0) for record in diagnostics)
    diagnostic_rounds = sum(int(record["raw_counts"].get("verify_rounds") or 0) for record in diagnostics)
    diagnostic_accepted = sum(int(record["raw_counts"].get("accepted") or 0) for record in diagnostics)
    diagnostic_unexecuted = sorted(
        f"{key[0]}-r{key[1]}"
        for key in diagnostic_keys
        if key not in sd_index
        or int(sd_index[key].get("raw_counts", {}).get("proposed") or 0) <= 0
        or int(sd_index[key].get("raw_counts", {}).get("verify_rounds") or 0) <= 0
    )
    natural_records = [
        record
        for key, record in sd_index.items()
        if key in expected_keys and expected[key[0]].get("category") != "diagnostic"
    ]
    natural_proposed = sum(int(record["raw_counts"].get("proposed") or 0) for record in natural_records)
    natural_rounds = sum(int(record["raw_counts"].get("verify_rounds") or 0) for record in natural_records)
    sd_executed = bool(diagnostic_keys) and not diagnostic_unexecuted and diagnostic_proposed > 0 and diagnostic_rounds > 0

    ar_coverage_complete = (
        not missing_ar
        and not extra_ar
        and not ar_duplicates
        and not ar_index_errors
        and not unexpected_repeats
        and not ar_invalid
        and not identity_mismatches_for_side(identity_mismatches, "ar:")
        and ar_ar["complete"]
    )
    sd_coverage_complete = (
        not missing_sd
        and not extra_sd
        and not sd_duplicates
        and not sd_index_errors
        and not unexpected_repeats
        and not sd_invalid
        and not identity_mismatches_for_side(identity_mismatches, "ngram:")
        and sd_ar["complete"]
    )
    ar_errors = [item for item in ar_invalid if item.startswith("ar:")]
    sd_errors = [item for item in sd_invalid if item.startswith("ngram:")]
    expected_source_verified = expected_source in {"manifest", "run_plan"}
    if not expected_source_verified:
        coverage_errors.append("expected request set is not from a frozen manifest or run plan")
    ar_offline_pass = bool(
        expected
        and repeats
        and expected_source_verified
        and ar_coverage_complete
        and ar_ar.get("rate") == 1.0
        and ar_ar.get("finish_reason_exact_rate") == 1.0
        and not ar_errors
    )
    sd_correctness_pass = bool(
        ar_offline_pass
        and sd_coverage_complete
        and sd_executed
        and sd_ar.get("rate") == 1.0
        and sd_ar.get("finish_reason_exact_rate") == 1.0
        and not sd_errors
        and not coverage_errors
        and not identity_mismatches
    )

    if ar_offline_pass:
        ar_status = "verified_greedy_exact"
    elif ar_ar.get("rate") is not None and ar_ar.get("rate") < 1.0:
        ar_status = "experimental_numeric_divergence"
    else:
        ar_status = "failed"
    if sd_correctness_pass:
        sd_status = "verified_greedy_exact"
    elif not sd_executed:
        sd_status = "failed"
    elif (
        expected_source_verified
        and sd_coverage_complete
        and sd_ar.get("rate") is not None
        and sd_ar.get("rate") > 0
        and _sd_repeat_exact(ngram_docs, sd_index, expected_keys, repeats)
    ):
        sd_status = "experimental_numeric_divergence"
    else:
        sd_status = "failed"

    coverage = {
        "expected_source": expected_source,
        "manifest_verified": expected_source_verified,
        "expected_requests": len(expected),
        "expected_repeats": repeats,
        "expected_pairs": len(expected_keys),
        "ar_pairs": len(ar_keys & expected_keys),
        "sd_pairs": len(sd_keys & expected_keys),
        "missing_ar": [f"{base}-r{repeat}" for base, repeat in missing_ar][:200],
        "missing_sd": [f"{base}-r{repeat}" for base, repeat in missing_sd][:200],
        "extra_ar": [f"{base}-r{repeat}" for base, repeat in extra_ar][:200],
        "extra_sd": [f"{base}-r{repeat}" for base, repeat in extra_sd][:200],
        "duplicate_ar": [f"{base}-r{repeat}" for base, repeat in sorted(ar_duplicates)][:200],
        "duplicate_sd": [f"{base}-r{repeat}" for base, repeat in sorted(sd_duplicates)][:200],
        "unexpected_repeats": unexpected_repeats,
        "invalid_ar_ids": ar_invalid[:200],
        "invalid_sd_ids": sd_invalid[:200],
        "identity_mismatches": identity_mismatches[:200],
        "ar_complete": ar_coverage_complete,
        "sd_complete": sd_coverage_complete,
        "errors": coverage_errors[:200],
        "num_errors": len(coverage_errors) + len(ar_invalid) + len(sd_invalid) + len(identity_mismatches),
        "failed_ids": failed_ids[:500],
    }

    completion_levels = {
        "AR_OFFLINE_PASS": bool(ar_offline_pass),
        "AR_API_PASS": False,
        "SD_EXECUTED": bool(sd_executed),
        "SD_CORRECTNESS_PASS": bool(sd_correctness_pass),
        "SERVING_LIFECYCLE_PASS": False,
        "ar_status": ar_status,
        "sd_status": sd_status,
    }
    return {
        "ar_ar": ar_ar,
        "sd_ar": sd_ar,
        "categories": dict(categories),
        "diagnostic": {
            "requests": len(diagnostics),
            "expected_requests": len(diagnostic_keys),
            "proposed_tokens": diagnostic_proposed,
            "verify_rounds": diagnostic_rounds,
            "accepted_tokens": diagnostic_accepted,
            "proposed_positive": diagnostic_proposed > 0,
            "verification_entered": diagnostic_rounds > 0,
            "unexecuted_ids": diagnostic_unexecuted[:200],
        },
        "natural": {
            "requests": len(natural_records),
            "proposed_tokens": natural_proposed,
            "verify_rounds": natural_rounds,
        },
        "coverage": coverage,
        "completion_levels": completion_levels,
        "divergences": details[:200],
        "num_divergences": len(details),
        "failed_ids": failed_ids[:500],
    }


def _sd_repeat_exact(
    ngram_docs: list[dict[str, Any]],
    sd_index: dict[tuple[str, int], dict[str, Any]],
    expected_keys: set[tuple[str, int]],
    repeats: list[int],
) -> bool:
    for base in {base for base, _ in expected_keys}:
        records = [sd_index.get((base, repeat)) for repeat in repeats]
        if any(record is None for record in records):
            return False
        tokens = {tuple(record.get("token_ids") or []) for record in records}
        finishes = {record.get("finish_reason") for record in records if record is not None}
        if len(tokens) != 1 or len(finishes) != 1:
            return False
    return True


def identity_mismatches_for_side(items: list[str], prefix: str) -> list[str]:
    return [item for item in items if item.startswith(prefix)]


def load_docs(paths: list[str]) -> list[dict[str, Any]]:
    import json

    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            docs.append(json.load(handle))
    return docs


def run_compare(
    ar_paths: list[str],
    ngram_paths: list[str],
    out: Optional[str] = None,
    *,
    manifest_path: Optional[str] = None,
) -> dict[str, Any]:
    ar_docs = load_docs(ar_paths)
    if not ar_docs:
        raise ValueError("at least one AR document is required")
    ngram_docs = load_docs(ngram_paths)
    if not ngram_docs:
        raise ValueError("at least one N-Gram document is required")
    manifest = None
    if manifest_path:
        import json

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ar_doc = copy.deepcopy(ar_docs[0])
    if len(ar_docs) > 1:
        ar_doc["records"] = [record for doc in ar_docs for record in doc.get("records", [])]
    summary = compare_documents(ar_doc, ngram_docs, manifest=manifest)
    summary["inputs"] = {"ar": ar_paths, "ngram": ngram_paths, "manifest": manifest_path}
    summary["ar_meta"] = [doc.get("meta", {}) for doc in ar_docs]
    summary["ngram_meta"] = [doc.get("meta", {}) for doc in ngram_docs]
    summary["correction_provenance"] = {
        "gate": "strict_coverage_v2",
        "source": "raw offline documents; no token timestamps or engine traces invented",
    }
    if out:
        write_json_atomic(out, summary)
    return summary
