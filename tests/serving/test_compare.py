from __future__ import annotations

from vlm_dflash.serving.compare import compare_documents


def _record(
    request_id,
    tokens,
    category="ocr",
    proposed=0,
    rounds=0,
    accepted=0,
    *,
    prompt_sha256="prompt",
    image_sha256="image",
    finish_reason="stop",
    error=None,
    correctness_status="not_checked",
):
    record = {
        "request_id": request_id,
        "token_ids": tokens,
        "text": str(tokens),
        "finish_reason": finish_reason,
        "category": category,
        "raw_counts": {"proposed": proposed, "verify_rounds": rounds, "accepted": accepted},
        "prompt_sha256": prompt_sha256,
        "image_sha256": image_sha256,
        "correctness_status": correctness_status,
    }
    if error:
        record["error"] = error
    return record


def _doc(records):
    return {"records": records}


def test_exact_match_passes():
    ar = _doc([
        _record("a-r0", [1, 2, 3]),
        _record("a-r1", [1, 2, 3]),
        _record("d-r0", [9, 9], "diagnostic"),
        _record("d-r1", [9, 9], "diagnostic"),
    ])
    ngram = _doc([
        _record("a-r0", [1, 2, 3], proposed=2, rounds=1, accepted=1),
        _record("a-r1", [1, 2, 3], proposed=2, rounds=1, accepted=1),
        _record("d-r0", [9, 9], "diagnostic", proposed=8, rounds=2, accepted=2),
        _record("d-r1", [9, 9], "diagnostic", proposed=8, rounds=2, accepted=2),
    ])
    manifest = {
        "requests": [
            {"request_id": "a", "category": "ocr", "prompt_sha256": "prompt", "image_sha256": "image"},
        ],
        "diagnostic": [
            {"request_id": "d", "category": "diagnostic", "prompt_sha256": "prompt", "image_sha256": "image"},
        ],
    }
    summary = compare_documents(ar, [ngram], manifest=manifest)
    assert summary["ar_ar"]["rate"] == 1.0
    assert summary["sd_ar"]["exact_rate"] == 1.0
    assert summary["diagnostic"]["proposed_positive"] is True
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is True


def test_exact_transcript_without_frozen_manifest_cannot_pass():
    ar = _doc([_record("a-r0", [1, 2, 3])])
    ngram = _doc([_record("a-r0", [1, 2, 3], proposed=2, rounds=1, accepted=1)])

    summary = compare_documents(ar, [ngram])

    assert summary["coverage"]["manifest_verified"] is False
    assert summary["completion_levels"]["AR_OFFLINE_PASS"] is False
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False
    assert "expected request set is not from a frozen manifest or run plan" in summary["coverage"]["errors"]


def test_no_proposals_fails_sd_executed():
    ar = _doc([_record("a-r0", [1, 2, 3])])
    ngram = _doc([_record("a-r0", [1, 2, 3], proposed=0, rounds=0, accepted=0)])
    summary = compare_documents(ar, [ngram])
    assert summary["completion_levels"]["SD_EXECUTED"] is False
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False


def test_systematic_divergence_is_failed():
    ar = _doc([_record("a-r0", [1, 2, 3]), _record("b-r0", [4, 5])])
    ngram = _doc([
        _record("a-r0", [7, 8, 9], proposed=8, rounds=2, accepted=2),
        _record("b-r0", [7, 8], proposed=8, rounds=2, accepted=2),
    ])
    summary = compare_documents(ar, [ngram])
    assert summary["completion_levels"]["sd_status"] == "failed"


def test_ar_request_error_blocks_offline_pass():
    ar = _doc([_record("a-r0", [1, 2, 3]), _record("a-r1", [1, 2, 3])])
    ar["records"][0]["error"] = "transport error"
    ngram = _doc([_record("a-r0", [1, 2, 3], proposed=2, rounds=1, accepted=1)])

    summary = compare_documents(ar, [ngram])

    assert summary["ar_ar"]["rate"] == 1.0
    assert summary["completion_levels"]["AR_OFFLINE_PASS"] is False


def test_diagnostic_only_sd_fails_coverage():
    ar = _doc([
        _record("a-r0", [1, 2, 3]),
        _record("d-r0", [9, 9], "diagnostic"),
    ])
    ngram = _doc([_record("d-r0", [9, 9], "diagnostic", proposed=8, rounds=2, accepted=2)])

    summary = compare_documents(ar, [ngram])

    assert summary["coverage"]["missing_sd"] == ["a-r0"]
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False


def test_sd_error_blocks_correctness_even_with_identical_tokens():
    ar = _doc([_record("a-r0", [1, 2, 3]), _record("d-r0", [9, 9], "diagnostic")])
    ngram = _doc([
        _record("a-r0", [1, 2, 3], proposed=2, rounds=1, accepted=1, error="boom", correctness_status="failed"),
        _record("d-r0", [9, 9], "diagnostic", proposed=8, rounds=2, accepted=2),
    ])

    summary = compare_documents(ar, [ngram])

    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False
    assert summary["coverage"]["invalid_sd_ids"]


def test_zero_execution_blocks_correctness():
    ar = _doc([_record("a-r0", [1, 2, 3]), _record("d-r0", [9, 9], "diagnostic")])
    ngram = _doc([
        _record("a-r0", [1, 2, 3], proposed=0, rounds=0, accepted=0),
        _record("d-r0", [9, 9], "diagnostic", proposed=0, rounds=0, accepted=0),
    ])

    summary = compare_documents(ar, [ngram])

    assert summary["completion_levels"]["SD_EXECUTED"] is False
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False
    assert summary["diagnostic"]["unexecuted_ids"] == ["d-r0"]


def test_duplicate_sd_pair_is_rejected():
    ar = _doc([_record("a-r0", [1]), _record("d-r0", [2], "diagnostic")])
    ngram = _doc([
        _record("a-r0", [1], proposed=1, rounds=1, accepted=1),
        _record("a-r0", [1], proposed=1, rounds=1, accepted=1),
        _record("d-r0", [2], "diagnostic", proposed=1, rounds=1, accepted=1),
    ])

    summary = compare_documents(ar, [ngram])

    assert summary["coverage"]["duplicate_sd"] == ["a-r0"]
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False


def test_missing_repeat_does_not_borrow_another_repeat():
    ar = _doc([
        _record("a-r0", [1, 2, 3]),
        _record("a-r1", [1, 2, 3]),
        _record("d-r0", [9], "diagnostic"),
        _record("d-r1", [9], "diagnostic"),
    ])
    ngram = _doc([
        _record("a-r0", [1, 2, 3], proposed=1, rounds=1, accepted=1),
        _record("d-r0", [9], "diagnostic", proposed=1, rounds=1, accepted=1),
    ])

    summary = compare_documents(ar, [ngram])

    assert summary["coverage"]["missing_sd"] == ["a-r1", "d-r1"]
    assert summary["sd_ar"]["pairs"] == 2
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False


def test_identity_mismatch_blocks_pass():
    manifest = {
        "requests": [
            {"request_id": "a", "category": "ocr", "prompt_sha256": "p", "image_sha256": "i"},
        ],
        "diagnostic": [
            {"request_id": "d", "category": "diagnostic", "prompt_sha256": "pd", "image_sha256": "id"},
        ],
    }
    ar = _doc([
        _record("a-r0", [1], prompt_sha256="p", image_sha256="i"),
        _record("d-r0", [2], "diagnostic", prompt_sha256="pd", image_sha256="id"),
    ])
    ngram = _doc([
        _record("a-r0", [1], proposed=1, rounds=1, accepted=1, image_sha256="other"),
        _record("d-r0", [2], "diagnostic", proposed=1, rounds=1, accepted=1, prompt_sha256="pd", image_sha256="id"),
    ])

    summary = compare_documents(ar, [ngram], manifest=manifest)

    assert summary["coverage"]["identity_mismatches"]
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is False


def test_manifest_verified_exact_positive_passes():
    manifest = {
        "requests": [
            {"request_id": "a", "category": "ocr", "prompt_sha256": "p", "image_sha256": "i"},
        ],
        "diagnostic": [
            {"request_id": "d", "category": "diagnostic", "prompt_sha256": "pd", "image_sha256": "id"},
        ],
    }
    ar = _doc([
        _record("a-r0", [1], prompt_sha256="p", image_sha256="i"),
        _record("a-r1", [1], prompt_sha256="p", image_sha256="i"),
        _record("d-r0", [2], "diagnostic", prompt_sha256="pd", image_sha256="id"),
        _record("d-r1", [2], "diagnostic", prompt_sha256="pd", image_sha256="id"),
    ])
    ngram = _doc([
        _record("a-r0", [1], proposed=1, rounds=1, accepted=1, prompt_sha256="p", image_sha256="i"),
        _record("a-r1", [1], proposed=1, rounds=1, accepted=1, prompt_sha256="p", image_sha256="i"),
        _record("d-r0", [2], "diagnostic", proposed=1, rounds=1, accepted=1, prompt_sha256="pd", image_sha256="id"),
        _record("d-r1", [2], "diagnostic", proposed=1, rounds=1, accepted=1, prompt_sha256="pd", image_sha256="id"),
    ])

    summary = compare_documents(ar, [ngram], manifest=manifest)

    assert summary["coverage"]["manifest_verified"] is True
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is True
