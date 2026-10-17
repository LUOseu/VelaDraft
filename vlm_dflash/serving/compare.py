"""Compare AR and N-Gram offline token transcripts."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .contracts import write_json_atomic

_REPEAT_RE = re.compile(r"-r(\d+)$")


def _base_request_id(request_id: str) -> str:
    return _REPEAT_RE.sub("", request_id)


def _index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for record in document.get("records", []):
        out[_base_request_id(record["request_id"])] = record
    return out


def _pairwise(records_by_repeat: list[dict[str, dict[str, Any]]], request_ids: list[str]) -> dict[str, Any]:
    counts = {"pairs": 0, "exact": 0}
    for i in range(len(records_by_repeat)):
        for j in range(i + 1, len(records_by_repeat)):
            for request_id in request_ids:
                left = records_by_repeat[i].get(request_id)
                right = records_by_repeat[j].get(request_id)
                if left is None or right is None:
                    continue
                counts["pairs"] += 1
                if left.get("token_ids") == right.get("token_ids"):
                    counts["exact"] += 1
    counts["rate"] = counts["exact"] / counts["pairs"] if counts["pairs"] else None
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


def compare_documents(ar_doc: dict[str, Any], ngram_docs: list[dict[str, Any]]) -> dict[str, Any]:
    ar_records = ar_doc.get("records", [])
    ar_by_repeat = defaultdict(dict)
    for record in ar_records:
        match = _REPEAT_RE.search(record["request_id"])
        repeat = int(match.group(1)) if match else 0
        ar_by_repeat[repeat][_base_request_id(record["request_id"])] = record
    repeat_keys = sorted(ar_by_repeat)
    ngram_by_repeat = []
    for doc in ngram_docs:
        indexed = {}
        for record in doc.get("records", []):
            match = _REPEAT_RE.search(record["request_id"])
            repeat = int(match.group(1)) if match else 0
            indexed.setdefault(repeat, {})[_base_request_id(record["request_id"])] = record
        ngram_by_repeat.append(indexed)
    request_ids = sorted(set().union(*[set(index) for index in ar_by_repeat.values()])) if ar_by_repeat else []
    ar_ar = _pairwise([ar_by_repeat[repeat] for repeat in repeat_keys], request_ids)
    sd_ar_counts = {"pairs": 0, "exact": 0, "sequence_exact": 0, "finish_reason_exact": 0}
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"pairs": 0, "exact": 0})
    details = []
    for doc in ngram_docs:
        for repeat, indexed in _iter_repeat_indexes(doc):
            reference_index = ar_by_repeat.get(repeat) or ar_by_repeat[repeat_keys[0]]
            for request_id, candidate in indexed.items():
                reference = reference_index.get(request_id)
                if reference is None:
                    continue
                diff = _compare_one(reference, candidate)
                sd_ar_counts["pairs"] += 1
                sd_ar_counts["exact"] += int(diff["exact"])
                sd_ar_counts["sequence_exact"] += int(diff["exact"])
                sd_ar_counts["finish_reason_exact"] += int(diff["finish_reason_match"])
                category = candidate.get("category", "unknown")
                categories[category]["pairs"] += 1
                categories[category]["exact"] += int(diff["exact"])
                if not diff["exact"]:
                    details.append({"repeat": repeat, "request_id": request_id, "category": category, **diff})
    for counts in (sd_ar_counts,):
        for key in ("exact", "sequence_exact", "finish_reason_exact"):
            counts[key + "_rate"] = counts[key] / counts["pairs"] if counts["pairs"] else None
    for category, counts in categories.items():
        counts["rate"] = counts["exact"] / counts["pairs"] if counts["pairs"] else None

    # Speculative-execution evidence: diagnostics must actually propose and verify.
    diagnostics = []
    for doc in ngram_docs:
        for record in doc.get("records", []):
            if record.get("category") == "diagnostic":
                diagnostics.append(record)
    diagnostic_proposed = sum(int(record["raw_counts"].get("proposed") or 0) for record in diagnostics)
    diagnostic_rounds = sum(int(record["raw_counts"].get("verify_rounds") or 0) for record in diagnostics)
    diagnostic_accepted = sum(int(record["raw_counts"].get("accepted") or 0) for record in diagnostics)
    natural_proposed = sum(
        int(record["raw_counts"].get("proposed") or 0)
        for doc in ngram_docs
        for record in doc.get("records", [])
        if record.get("category") != "diagnostic"
    )
    natural_rounds = sum(
        int(record["raw_counts"].get("verify_rounds") or 0)
        for doc in ngram_docs
        for record in doc.get("records", [])
        if record.get("category") != "diagnostic"
    )
    sd_executed = diagnostic_proposed > 0 and diagnostic_rounds > 0
    ar_exact = ar_ar.get("rate")
    sd_exact = sd_ar_counts.get("exact_rate")
    ar_errors = sum(1 for record in ar_records if record.get("error"))
    if ar_exact is not None and ar_exact == 1.0 and sd_exact == 1.0:
        sd_status = "verified_greedy_exact"
    elif not sd_executed:
        sd_status = "failed"
    elif ar_exact is not None and sd_exact is not None and sd_exact + 1e-12 >= ar_exact:
        sd_status = "verified_greedy_exact" if sd_exact == 1.0 and ar_exact == 1.0 else "experimental_numeric_divergence"
    elif ar_exact is not None and sd_exact is not None and sd_exact >= ar_exact - 0.05:
        sd_status = "experimental_numeric_divergence"
    else:
        sd_status = "failed"
    ar_status = "verified_greedy_exact" if ar_exact == 1.0 else "experimental_numeric_divergence"
    completion_levels = {
        "AR_OFFLINE_PASS": bool(ar_records) and ar_errors == 0 and ar_exact == 1.0,
        "AR_API_PASS": False,
        "SD_EXECUTED": bool(sd_executed),
        "SD_CORRECTNESS_PASS": sd_status == "verified_greedy_exact",
        "SERVING_LIFECYCLE_PASS": False,  # filled by the HTTP measurement summary
        "ar_status": ar_status,
        "sd_status": sd_status,
    }
    return {
        "ar_ar": ar_ar,
        "sd_ar": sd_ar_counts,
        "categories": dict(categories),
        "diagnostic": {
            "requests": len(diagnostics),
            "proposed_tokens": diagnostic_proposed,
            "verify_rounds": diagnostic_rounds,
            "accepted_tokens": diagnostic_accepted,
            "proposed_positive": diagnostic_proposed > 0,
            "verification_entered": diagnostic_rounds > 0,
        },
        "natural": {
            "proposed_tokens": natural_proposed,
            "verify_rounds": natural_rounds,
        },
        "completion_levels": completion_levels,
        "divergences": details[:200],
        "num_divergences": len(details),
    }


def _iter_repeat_indexes(document: dict[str, Any]):
    by_repeat = defaultdict(dict)
    for record in document.get("records", []):
        match = _REPEAT_RE.search(record["request_id"])
        repeat = int(match.group(1)) if match else 0
        by_repeat[repeat][_base_request_id(record["request_id"])] = record
    return sorted(by_repeat.items())


def load_docs(paths: list[str]) -> list[dict[str, Any]]:
    import json

    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            docs.append(json.load(handle))
    return docs


def run_compare(ar_paths: list[str], ngram_paths: list[str], out: Optional[str] = None) -> dict[str, Any]:
    ar_docs = load_docs(ar_paths)
    if not ar_docs:
        raise ValueError("at least one AR document is required")
    ngram_docs = load_docs(ngram_paths)
    if not ngram_docs:
        raise ValueError("at least one N-Gram document is required")
    summary = compare_documents(ar_docs[0], ngram_docs)
    summary["inputs"] = {"ar": ar_paths, "ngram": ngram_paths}
    summary["ar_meta"] = [doc.get("meta", {}) for doc in ar_docs]
    summary["ngram_meta"] = [doc.get("meta", {}) for doc in ngram_docs]
    if out:
        write_json_atomic(out, summary)
    return summary
