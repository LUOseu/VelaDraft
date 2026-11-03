"""MMSpec test: fetch at a pinned commit, verify the official scope, and freeze the session list.

The formal suite is the complete ``dataset/MMSpec/test`` (600 sessions, 723 turns, 100 sessions per
topic). ``mmspec-smoke`` takes the first two test sessions of every topic (multi-turn ones with at
least two turns) and is only a functional check, never a formal table. Dataset answers are not used:
MMSpec ships prompts without references, so the untimed native AR reference supplies the history of
multi-turn sessions and task quality stays ``pending``.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from research.qwen36_a100_seven_eval.settings import CATEGORIES, sha256

EXPECTED = {
    "sessions": 600,
    "turns": 723,
    "turn_histogram": {1: 530, 2: 18, 3: 51, 4: 1},
    "per_topic": 100,
}
SMOKE_PER_TOPIC = 2


def fetch(repo_url: str, commit: str, destination: Path) -> Path:
    """Clone MMSpec (or reuse a clone) and check out the pinned commit; never moves an existing clone."""
    destination = Path(destination).resolve()
    existing = (destination / ".git").exists()
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(destination)], check=True
        )
    head = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("dataset clone has local changes")
    if head != commit:
        if existing:
            raise ValueError(
                "existing dataset clone is not at the pinned commit; use a dedicated clone"
            )
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--quiet", "origin", commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--quiet", "--detach", commit],
            check=True,
        )
    return destination


def load_test(repo: Path) -> list[dict]:
    path = Path(repo) / "dataset/MMSpec/test/mmspec.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_scope(rows: list[dict], expected: dict = EXPECTED) -> dict:
    """Official scope check; raises instead of silently topping up or trimming."""
    histogram = Counter(len(r["turns"]) for r in rows)
    topics = Counter(r["topic"] for r in rows)
    ids = [r["id"] for r in rows]
    problems = []
    if len(rows) != expected["sessions"]:
        problems.append(f"{len(rows)} sessions, expected {expected['sessions']}")
    if sum(histogram[k] * k for k in histogram) != expected["turns"]:
        problems.append(
            f"{sum(len(r['turns']) for r in rows)} turns, expected {expected['turns']}"
        )
    if dict(histogram) != expected["turn_histogram"]:
        problems.append(f"turn histogram {dict(histogram)}")
    if set(topics) != set(CATEGORIES) or set(topics.values()) != {
        expected["per_topic"]
    }:
        problems.append(f"topic counts {dict(topics)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate session ids")
    for r in rows:
        if not r["turns"] or any(
            not isinstance(t, str) or not t.strip() for t in r["turns"]
        ):
            problems.append(f"{r['id']}: empty turn")
    if problems:
        raise ValueError(
            "MMSpec test scope differs from the paper: " + "; ".join(problems)
        )
    return {
        "sessions": len(rows),
        "turns": sum(len(r["turns"]) for r in rows),
        "turn_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "topics": dict(topics),
    }


def smoke_rows(rows: list[dict]) -> list[dict]:
    chosen = []
    for topic in CATEGORIES:
        candidates = [r for r in rows if r["topic"] == topic]
        if topic == "multi-turn conversation":
            candidates = [r for r in candidates if len(r["turns"]) >= 2] or candidates
        chosen += candidates[:SMOKE_PER_TOPIC]
    return chosen


def build_sessions(repo: Path, suite: str) -> tuple[list[dict], dict]:
    """Sessions in the runtime's plan format plus a data manifest (commit, hashes, scope)."""
    repo = Path(repo)
    rows = load_test(repo)
    scope = verify_scope(rows)
    if suite == "mmspec-smoke":
        rows = smoke_rows(rows)
    elif suite != "mmspec-full":
        raise ValueError(f"unknown suite {suite}")
    images = repo / "dataset/MMSpec/test/images"
    sessions = []
    for index, row in enumerate(rows):
        image = images / row["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        sessions.append(
            {
                "id": row["id"],
                "questions": list(row["turns"]),
                "image": str(image),
                "image_sha256": sha256(image),
                "topic": row["topic"],
                "category": row["topic"],
                "source_category": row.get("category"),
                "source_index": index,
            }
        )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "source": "https://github.com/killthefullmoon/MMSpec",
        "commit": commit,
        "suite": suite,
        "formal": suite == "mmspec-full",
        "test_jsonl_sha256": sha256(repo / "dataset/MMSpec/test/mmspec.jsonl"),
        "full_scope": scope,
        "sessions": len(sessions),
        "turns": sum(len(s["questions"]) for s in sessions),
        "image_sha256": {s["id"]: s["image_sha256"] for s in sessions},
        "references": "none shipped with MMSpec test; quality is pending, generations are kept",
    }
    return sessions, manifest
