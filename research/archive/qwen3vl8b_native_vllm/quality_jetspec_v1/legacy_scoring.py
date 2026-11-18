"""Deterministic quality scoring for the fixed paper-v2 VQA workloads.

These functions intentionally contain no model or dataset dependencies so the
same code can score saved raw outputs later. The protocol is part of the result:
TextVQA uses the standard VQA soft accuracy, DocVQA uses ANLS with the usual
0.5 threshold, and ChartQA uses numeric relaxed accuracy with a fixed 5%
tolerance and an exact zero rule.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "its": "it's",
    "mightve": "might've",
    "mustnt": "mustn't",
    "shant": "shan't",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "thatd": "that'd",
    "thats": "that's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = set(string.punctuation)


def extract_short_answer(text: str, *, max_lines: int = 1) -> str:
    """Freeze the benchmark answer extraction rule.

    Generation is capped at 128 tokens, but the metric still needs one stable
    answer. Keep the first non-empty line, remove a leading ``Answer:`` label,
    and strip incidental Markdown/quote wrappers. Do not guess an answer from a
    later line and do not apply workload-specific heuristics that would make the
    extraction depend on the expected label.
    """
    lines = [line.strip() for line in str(text).replace("\r\n", "\n").split("\n")]
    nonempty = [line for line in lines if line]
    if not nonempty:
        return ""
    answer = " ".join(nonempty[: max(1, int(max_lines))]).strip()
    answer = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip().strip("`\"'").strip()
    if answer.endswith(".") and not re.fullmatch(r"[+-]?\d+\.", answer):
        answer = answer[:-1].rstrip()
    return answer


def normalize_vqa_answer(text: str) -> str:
    """Canonical VQA answer normalization (articles, punctuation, numbers)."""
    answer = str(text).lower().strip()
    answer = answer.replace("\n", " ").replace("\t", " ")
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)
    answer = re.sub(r"(?<!\d)\.(?!\d)", " ", answer)
    answer = "".join(" " if char in _PUNCTUATION else char for char in answer)
    tokens = []
    for token in answer.split():
        token = _CONTRACTIONS.get(token, token)
        token = _NUMBER_WORDS.get(token, token)
        if token not in _ARTICLES:
            tokens.append(token)
    return " ".join(tokens)


def vqa_accuracy(prediction: str, answers: Sequence[str]) -> float:
    """Standard VQA soft accuracy: ``min(#matches / 3, 1)`` per answer set."""
    if not answers:
        return 0.0
    target = normalize_vqa_answer(prediction)
    matches = sum(normalize_vqa_answer(answer) == target for answer in answers)
    return min(matches / 3.0, 1.0)


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            insert = current[column - 1] + 1
            delete = previous[column] + 1
            replace = previous[column - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def anls_score(prediction: str, answer: str, *, threshold: float = 0.5) -> float:
    """DocVQA ANLS for one reference answer."""
    target = " ".join(str(answer).lower().split())
    pred = " ".join(str(prediction).lower().split())
    if not target:
        return 1.0 if not pred else 0.0
    normalized_distance = _levenshtein(pred, target) / max(len(pred), len(target), 1)
    score = 1.0 - normalized_distance
    return score if score >= threshold else 0.0


def anls(prediction: str, answers: Sequence[str], *, threshold: float = 0.5) -> float:
    if not answers:
        return 0.0
    return max(anls_score(prediction, answer, threshold=threshold) for answer in answers)


def _parse_number(text: str) -> float | None:
    value = str(text).strip().replace(",", "")
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def chartqa_relaxed_score(prediction: str, answer: str, *, tolerance: float = 0.05) -> float:
    """ChartQA relaxed accuracy with a fixed numeric tolerance and zero rule."""
    pred_number = _parse_number(prediction)
    target_number = _parse_number(answer)
    if pred_number is not None and target_number is not None:
        if abs(target_number) < 1e-8:
            return 1.0 if abs(pred_number) < 1e-8 else 0.0
        return 1.0 if abs(pred_number - target_number) / abs(target_number) <= tolerance else 0.0
    pred = " ".join(str(prediction).lower().split())
    target = " ".join(str(answer).lower().split())
    return 1.0 if pred == target else 0.0


def score_workload(workload: str, records: Iterable[dict]) -> dict:
    """Score raw records carrying ``prediction`` and ``answers`` fields."""
    rows = list(records)
    scored = []
    for record in rows:
        prediction = extract_short_answer(record.get("prediction", ""))
        answers = [str(answer) for answer in record.get("answers", [])]
        if workload == "textvqa":
            score = vqa_accuracy(prediction, answers)
        elif workload == "docvqa":
            score = anls(prediction, answers)
        elif workload == "chartqa":
            score = max((chartqa_relaxed_score(prediction, answer) for answer in answers), default=0.0)
        else:
            raise ValueError(f"Unknown quality workload {workload!r}.")
        scored.append({**record, "extracted_answer": prediction, "score": score})
    return {
        "workload": workload,
        "num_records": len(scored),
        "num_scored": len(scored),
        "metric": {"textvqa": "vqa_accuracy", "docvqa": "anls", "chartqa": "relaxed_accuracy"}[workload],
        "score": (sum(row["score"] for row in scored) / len(scored)) if scored else None,
        "by_type": _mean_by_key(scored, "type"),
        "records": scored,
    }


def _mean_by_key(rows: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get(key) is None:
            continue
        grouped.setdefault(str(row[key]), []).append(float(row["score"]))
    return {
        value: {"num_records": len(scores), "score": sum(scores) / len(scores)}
        for value, scores in sorted(grouped.items())
    }


def answer_majority(answers: Iterable[str]) -> str:
    """Small helper for audit/report tables; never used to alter a prediction."""
    normalized = [normalize_vqa_answer(answer) for answer in answers if str(answer).strip()]
    return Counter(normalized).most_common(1)[0][0] if normalized else ""
