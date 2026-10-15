#!/usr/bin/env python3
"""Regenerate docs/TRAINING_SIGNALS.md from the algorithm registry.

Per-method training-signal declarations (CVPR-route phase-1 item 2): what each
objective consumes — hard-label supervision, context layers, teacher final
hidden states, distribution / confidence terms, stage structure and block
convention. Every row is rendered from
``vlm_dflash/algorithms/<name>.py::<Algorithm>.metadata.training_signals`` and
the recipe supports-matrix (``vlm_dflash.objectives.RECIPE_SUPPORT``), so the
document cannot drift from the code (drift test in ``tests/test_algorithms.py``).

Usage: ``python scripts/render_training_signals.py [--check]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash.algorithms import available_algorithms, get_algorithm  # noqa: E402
from vlm_dflash.objectives import RECIPE_SUPPORT  # noqa: E402

HEADER = """\
# Training signals

What each method's training objective consumes, per method: hard-label
supervision, context target layers, whether the target's final hidden states
are required (and under which selectable objective), distribution and
confidence objectives, stage structure, and the block supervision convention.
This is the requirement declaration behind the recipe machinery
(`algorithm.train_recipe`) and the capture gates
(`capture.store_final_hidden`).

Generated from `AlgorithmMetadata.training_signals`
(`vlm_dflash/algorithms/<name>.py`) and the recipe supports-matrix
(`RECIPE_SUPPORT` in `vlm_dflash/objectives.py`). Regenerate with
`python scripts/render_training_signals.py`. Signal text is deliberate: a
requirement stated here must match the trainer's behavior, and any change is
reviewed together with its tests.

In the accepted-recipes list below, ``none`` means the method has no
*additional* selectable recipe on top of its default training objective - it
does not mean the method lacks training. The default objective of each method
is the "Hard label" / "Distribution objective" row of its table.
"""

ROWS = (
    ("Hard label", "hard_label"),
    ("Context layers", "context_layers"),
    ("Teacher final hidden states", "teacher_final_hidden"),
    ("Distribution objective", "distribution_objective"),
    ("Confidence objective", "confidence_objective"),
    ("Stage", "stage"),
    ("Block convention", "block_convention"),
)


def _cell(text: str) -> str:
    """Markdown-table-safe cell (escape pipes; keep it on one line)."""
    return " ".join(str(text).split()).replace("|", r"\|")


def render() -> str:
    names = sorted(available_algorithms())
    parts = [HEADER, "\n## Recipes accepted per algorithm\n\n"]
    for name in names:
        recipes = sorted(
            recipe for recipe, algorithms in RECIPE_SUPPORT.items() if name in algorithms
        )
        listed = ", ".join(f"`{recipe}`" for recipe in recipes) if recipes else "none"
        parts.append(f"- `{name}`: {listed}\n")
    for name in names:
        algorithm = get_algorithm(name)
        signals = algorithm.metadata.training_signals
        assert signals is not None, f"{name} declares no training_signals"
        record = signals.as_dict()
        parts.append(f"\n## `{name}`\n\n| Signal | Requirement |\n| --- | --- |\n")
        for label, key in ROWS:
            parts.append(f"| {label} | {_cell(record[key])} |\n")
        if record.get("notes"):
            parts.append(f"| Notes | {_cell(record['notes'])} |\n")
        parts.append(f"\nDocs: [`docs/algorithms/{name}.md`](algorithms/{name}.md)\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when the checked-in file differs from the rendered output",
    )
    args = parser.parse_args()
    output = ROOT / "docs" / "TRAINING_SIGNALS.md"
    rendered = render()
    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != rendered:
            print(f"{output} is stale; run python scripts/render_training_signals.py", file=sys.stderr)
            return 1
        print(f"{output} is up to date")
        return 0
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
