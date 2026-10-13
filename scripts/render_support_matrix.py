#!/usr/bin/env python3
"""Regenerate SUPPORT_MATRIX.md from the algorithm registry.

The algorithms table (maturity / target scope / capability states) is rendered
from ``AlgorithmMetadata`` -- the same machine-readable data surfaced by
``vlm-dflash doctor --algorithm <name>`` and embedded in exported model cards
-- so the three public surfaces cannot drift apart. The remaining sections are
static template text kept in this file.

Usage: ``python scripts/render_support_matrix.py [--check]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash.algorithms import available_algorithms, get_algorithm  # noqa: E402

HEADER = """\
# Support matrix

Statuses: `stable` / `validated` / `reduced_validation` / `experimental` /
`architecture-only` / `unsupported`. A target column is `validated` when the
canonical protocol has been recorded for that family, and
`reduced_validation` when the full pipeline was recorded at a reduced
protocol (smaller slice / prompt budget). Numbers behind "validated" are in
`BENCHMARKS.md`; protocol details in the linked docs.

This file is generated: the algorithms table below is rendered from
`vlm_dflash/algorithms/<name>.py::<Algorithm>.metadata` (the same data behind
`vlm-dflash doctor --algorithm <name>` and exported model cards). Regenerate
with `python scripts/render_support_matrix.py`.
"""

COLUMNS = [
    "Algorithm",
    "Qwen2.5-VL",
    "Qwen3-VL",
    "Image",
    "Multi-image",
    "Video",
    "Train",
    "Eval",
    "Export",
]

FOOTNOTES = """\
- "Train" is the training maturity: DFlash2 trains with an experimental
  objective reconstruction (upstream objective undisclosed); everything else
  is validated.
- JetSpec eval numbers are validated with its causal in-block mask (its
  parallel-tree production runtime is not ported); every algorithm's runtime
  story is sequential token-exact verification only.
- Eval/export for every algorithm load the exported checkpoint through the
  algorithm's own loader dispatch and produced token-exact output in the
  recorded runs (`RELEASE_VALIDATION.md`).
"""

TAIL = """
## Targets

| Target | Adapter | Regen | Capture | Train | Eval | Export |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | qwen25_vl | validated | validated | validated | validated | validated |
| Qwen3-VL-8B-Instruct | qwen3_vl | validated | validated | validated | validated | validated |

Both are validated on single-image + text with greedy decoding. Qwen3-VL was
validated for the four zoo algorithms at reduced scale (256-row slice,
16-prompt sequential eval, export + reload, all token-exact - see
`RELEASE_VALIDATION.md`); DFlash has the full 2K-protocol run there.
Multi-image, video and extra target families are roadmap items, not support
claims.

## Verification modes

| Mode | Exactness | Use |
| --- | --- | --- |
| sequential | token-exact (gate) | all acceptance/correctness numbers |
| strict_parallel | partial fallback; see probe | diagnostics only |
| parallel | not exact (measured) | experimental throughput |

## Benchmark workloads

| Workload | Status | Prompt set |
| --- | --- | --- |
| caption_general | scaffold | run's own val split (provisional) |
| textvqa | validated | pinned export (lmms-lab/textvqa validation, 128-prompt seed-42 subset; `benchmarks/prompts/textvqa_v1.jsonl`) |
| docvqa | planned | protocol pinned, adapter pending |
| chartqa | planned | protocol pinned, adapter pending |
"""


def _state(supported: bool) -> str:
    return "validated" if supported else "unsupported"


def render() -> str:
    rows = []
    for name in available_algorithms():
        meta = get_algorithm(name).metadata
        caps = meta.capabilities
        targets = meta.targets
        row = [
            name,
            targets.get("qwen25_vl", "unsupported"),
            targets.get("qwen3_vl", "unsupported"),
            _state(bool(caps.get("image"))),
            _state(bool(caps.get("multi_image"))),
            _state(bool(caps.get("video"))),
            meta.maturity.get("training", "unsupported"),
            _state(bool(caps.get("eval"))),
            _state(bool(caps.get("export"))),
        ]
        rows.append(row)

    widths = [max(len(COLUMNS[i]), *(len(row[i]) for row in rows)) for i in range(len(COLUMNS))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    table = [line(COLUMNS), "| " + " | ".join("-" * width for width in widths) + " |"]
    table += [line(row) for row in rows]
    return HEADER + "\n## Algorithms\n\n" + "\n".join(table) + "\n\n" + FOOTNOTES + TAIL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if SUPPORT_MATRIX.md is stale")
    args = parser.parse_args()
    rendered = render()
    target = ROOT / "SUPPORT_MATRIX.md"
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != rendered:
            print("SUPPORT_MATRIX.md is stale; run python scripts/render_support_matrix.py", file=sys.stderr)
            return 1
        print("SUPPORT_MATRIX.md is up to date")
        return 0
    target.write_text(rendered, encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
