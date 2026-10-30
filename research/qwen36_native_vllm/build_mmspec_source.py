"""Freeze MMSpec test (600 sessions / 723 turns) and the 12-session smoke as native-eval sources.

Output rows are native-eval ShareGPT: one image, alternating human/gpt turns whose
gpt values are empty placeholders (dataset answers are never used; the untimed
native AR reference supplies the causal history). Original id, image, turns,
topic and category plus the source line index are kept on every row.

  python build_mmspec_source.py <MMSpec clone at 290486861> <testmini jsonl> <out dir>
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from common import sha256

COMMIT = "290486861eabd746075ca776adb66175636c1fc0"
TOPICS = ("general vqa", "text vqa", "image captioning", "chart understanding",
          "complex reasoning pro", "multi-turn conversation")
EXPECTED = dict(sessions=600, turns=723, turn_histogram={1: 530, 2: 18, 3: 51, 4: 1})
SMOKE_PER_TOPIC = 2


def sharegpt(row: dict, index: int) -> dict:
    turns = row["turns"]
    if not turns or any(not isinstance(t, str) or not t.strip() for t in turns):
        raise ValueError(f"{row['id']}: empty turn")
    if any("<image>" in t for t in turns):
        raise ValueError(f"{row['id']}: literal <image> would be stripped by native-eval")
    conversations = []
    for i, turn in enumerate(turns):
        if i:
            conversations.append({"from": "gpt", "value": ""})
        conversations.append({"from": "human", "value": turn})
    return {"id": row["id"], "images": [row["image"]], "conversations": conversations,
            "topic": row["topic"], "category": row["category"], "source_index": index,
            "turns": turns}


def smoke_ids(mini: list[dict]) -> list[str]:
    """First two testmini sessions per topic in file order; the multi-turn topic
    takes its first two sessions with at least two turns so history is exercised."""
    chosen = []
    for topic in TOPICS:
        rows = [r for r in mini if r["topic"] == topic]
        if topic == "multi-turn conversation":
            rows = [r for r in rows if len(r["turns"]) >= 2] or rows
        if len(rows) < SMOKE_PER_TOPIC:
            raise ValueError(f"testmini lacks {SMOKE_PER_TOPIC} sessions for {topic}")
        chosen += [r["id"] for r in rows[:SMOKE_PER_TOPIC]]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("testmini", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError(f"MMSpec clone at {head}, expected {COMMIT}")
    test_file = args.repo / "dataset/MMSpec/test/mmspec.jsonl"
    images = args.repo / "dataset/MMSpec/test/images"
    test = [json.loads(line) for line in test_file.read_text().splitlines() if line.strip()]
    mini = [json.loads(line) for line in args.testmini.read_text().splitlines() if line.strip()]
    histogram = Counter(len(r["turns"]) for r in test)
    counts = Counter(r["topic"] for r in test)
    if (len(test), sum(len(r["turns"]) for r in test), dict(histogram)) != (
            EXPECTED["sessions"], EXPECTED["turns"], EXPECTED["turn_histogram"]):
        raise ValueError(f"Unexpected MMSpec test scope: {len(test)} sessions, {dict(histogram)}")
    if set(counts) != set(TOPICS) or set(counts.values()) != {100}:
        raise ValueError(f"Unexpected topic counts {dict(counts)}")
    by_id = {r["id"]: r for r in test}
    if len(by_id) != len(test):
        raise ValueError("Duplicate session ids")
    for row in mini:
        if by_id.get(row["id"]) != row:
            raise ValueError(f"testmini row {row['id']} differs from the full test row")
    args.out.mkdir(parents=True, exist_ok=True)
    image_hashes = {row["image"]: sha256(images / row["image"]) for row in test}
    full = [sharegpt(row, i) for i, row in enumerate(test)]
    index = {r["id"]: i for i, r in enumerate(test)}
    smoke = [sharegpt(by_id[i], index[i]) for i in smoke_ids(mini)]
    for name, rows in (("mmspec_test.jsonl", full), ("mmspec_smoke12.jsonl", smoke)):
        (args.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "source": "https://github.com/killthefullmoon/MMSpec", "commit": COMMIT,
        "test_jsonl_sha256": sha256(test_file), "testmini_jsonl_sha256": sha256(args.testmini),
        "images_dir": str(images), "image_sha256": image_hashes,
        "sessions": len(full), "turns": sum(len(r["turns"]) for r in test),
        "turn_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "topics": dict(counts),
        "smoke": {"rule": "first two testmini sessions per topic in file order; multi-turn topic "
                          "restricted to sessions with >=2 turns", "ids": [r["id"] for r in smoke],
                  "turns": sum(len(r["turns"]) for r in smoke)},
        "outputs": {name: sha256(args.out / name) for name in ("mmspec_test.jsonl", "mmspec_smoke12.jsonl")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "image_sha256"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
