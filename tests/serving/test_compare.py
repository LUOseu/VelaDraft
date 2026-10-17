from __future__ import annotations

from vlm_dflash.serving.compare import compare_documents


def _record(request_id, tokens, category="ocr", proposed=0, rounds=0, accepted=0):
    return {
        "request_id": request_id,
        "token_ids": tokens,
        "text": str(tokens),
        "finish_reason": "stop",
        "category": category,
        "raw_counts": {"proposed": proposed, "verify_rounds": rounds, "accepted": accepted},
    }


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
    summary = compare_documents(ar, [ngram])
    assert summary["ar_ar"]["rate"] == 1.0
    assert summary["sd_ar"]["exact_rate"] == 1.0
    assert summary["diagnostic"]["proposed_positive"] is True
    assert summary["completion_levels"]["SD_CORRECTNESS_PASS"] is True


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
