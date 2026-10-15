#!/usr/bin/env python3
"""Regenerate SUPPORT_MATRIX.md from the algorithm registry.

Two axes are kept separate (review R6):

- **capability** - what the code implements (``implemented`` /
  ``unsupported``); a capability flag is never an evidence claim;
- **evidence state** - what has actually been recorded for a concrete
  family x layout x feature-mode combination (``implemented`` /
  ``cpu_tested`` / ``gpu_smoke`` / ``reduced_validation`` /
  ``canonical_validated`` / ``blocked``), each record carrying a pointer to
  run ids / configs / commits / protocol docs.

The capability table, the per-family derived table and the evidence records
are all rendered from ``AlgorithmMetadata`` -- the same machine-readable data
surfaced by ``vlm-dflash doctor --algorithm <name>`` and embedded in exported
model cards -- so the public surfaces cannot drift apart. The remaining
sections are static template text kept in this file.

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
from vlm_dflash.algorithms.base import EVIDENCE_STATES  # noqa: E402

HEADER = """\
# Support matrix

Two separate axes. **Capability** (`implemented` / `unsupported`) says what
the code implements; it is never an evidence claim. **Evidence state**
(`implemented` / `cpu_tested` / `gpu_smoke` / `reduced_validation` /
`canonical_validated` / `blocked`) says what has actually been recorded for a
family x layout x feature-mode combination, each row traced to run ids /
configs / commits / protocol docs. Numbers behind recorded states live in
`BENCHMARKS.md` / `RELEASE_VALIDATION.md`; a claim without a traceable record
is downgraded.

This file is generated: the capability table, the per-family derived table
and the evidence records are rendered from
`vlm_dflash/algorithms/<name>.py::<Algorithm>.metadata` (the same data behind
`vlm-dflash doctor --algorithm <name>` and exported model cards). Regenerate
with `python scripts/render_support_matrix.py`.
"""

FOOTNOTES = """\
Notes:

- Training maturity (the `maturity.training` field behind `doctor` / model
  cards): DFlash2 trains with an experimental objective reconstruction
  (upstream objective undisclosed); everything else is validated, including
  DFly's ported `upstream_two_stage` recipe, whose canonical 2K run is
  recorded (`R-20261014-11`: AL 1.839 vs controlled_ce 1.682, 128/128
  token-exact - see the evidence records and `docs/upstream/dfly.md`).
- **Multi-image is a capability, not a blanket validation.** The schema-v2
  data path (multiple images per row, images on any user turn) is
  implemented for every algorithm and exercised by the shared CPU suite; the
  only recorded GPU end-to-end multi-image run is DFlash's E2 smoke
  (`R-20261014-08`, 2-image/2-turn rows, 16 train + 4 val, token-exact eval;
  `configs/smoke_multiturn.yaml`) - see the evidence table.
- Batched online capture for multi-image rows was measured (`R-20261014-12`:
  16 -> 4 target forward calls, bit-identical losses); its extraction-wall
  comparison is superseded pending a paired re-measurement under the unified
  timing boundary (review R2, `docs/plan/WP4_MEASUREMENT.md`).
- JetSpec eval numbers are validated with its causal in-block mask (its
  parallel-tree production runtime is not ported); every algorithm's runtime
  story is sequential token-exact verification only.
- Eval/export states come from the recorded runs in the evidence table;
  `canonical_validated` requires the canonical protocol (`BENCHMARKS.md`
  scale), `reduced_validation` a recorded reduced protocol, `gpu_smoke` a
  real GPU run at smoke/demo scale only.
"""

TAIL = """
## Targets

| Target | Adapter | Regen | Capture | Train | Eval | Export |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | qwen25_vl | validated | validated | validated | validated | validated |
| Qwen3-VL-8B-Instruct | qwen3_vl | validated | validated | validated | validated | validated |
| SmolVLM2-2.2B-Instruct | smolvlm | gpu_smoke | gpu_smoke | gpu_smoke | gpu_smoke | - |

Qwen2.5-VL and Qwen3-VL are validated on single-image + text with greedy
decoding. Qwen3-VL was validated for the four zoo algorithms at reduced scale
(256-row slice, 16-prompt sequential eval, export + reload, all token-exact -
see `RELEASE_VALIDATION.md`); DFlash has the full 2K-protocol run there.
SmolVLM2 has one recorded engineering smoke (`R-20261014-17`: single-image,
16 train + 4 val synthetic rows, token-exact 4/4; multi-image and online mode
untested for this family; no quality claim) - see the evidence records.
Video and further target families remain roadmap items, not support claims.

## Verification modes

| Mode | Exactness | Use |
| --- | --- | --- |
| sequential | token-exact (gate) | all acceptance/correctness numbers |
| strict_parallel | partial fallback; see probe | diagnostics only |
| parallel | not exact (measured) | experimental throughput |

## Feature modes

This section is static text in the renderer (feature modes are orthogonal to
the algorithm registry above).

| Feature mode | Status | Evidence |
| --- | --- | --- |
| `offline` (staged capture) | validated | the canonical protocol runs in `BENCHMARKS.md` / `RELEASE_VALIDATION.md` are all offline-path numbers |
| `online` (live target extraction) | experimental | full CPU suite + GPU plumbing smokes on Qwen2.5-VL-3B and Qwen3-VL-8B (`docs/online-training.md`): token-exact eval, provenance, digests, exact-resume gates, per-phase peaks |

Online has no canonical-quality recording yet — its evidence is plumbing
smokes (now including a split-device run; see `docs/online-training.md` GPU
smoke 6). Do not relabel offline benchmark numbers as online.

## Benchmark workloads

| Workload | Status | Prompt set |
| --- | --- | --- |
| caption_general | scaffold | run's own val split (provisional) |
| textvqa | validated | pinned export (lmms-lab/textvqa validation, 128-prompt seed-42 subset; `benchmarks/prompts/textvqa_v1.jsonl`) |
| docvqa | validated | pinned export (lmms-lab/DocVQA DocVQA/validation, 128-prompt seed-42 subset; `benchmarks/prompts/docvqa_v1.jsonl`) |
| chartqa | validated | pinned export (lmms-lab/ChartQA test, 128-prompt seed-42 subset; `benchmarks/prompts/chartqa_v1.jsonl`) |
"""

CAPABILITY_COLUMNS = ["Algorithm", "Image", "Multi-image", "Video", "Eval", "Export"]
FAMILY_COLUMNS = ["Algorithm", "Qwen2.5-VL", "Qwen3-VL", "SmolVLM2"]
EVIDENCE_COLUMNS = ["Algorithm", "Family", "Layout", "Mode", "State", "Evidence"]

FAMILY_ORDER = ["qwen25_vl", "qwen3_vl", "smolvlm"]
FAMILY_LABELS = {"qwen25_vl": "Qwen2.5-VL", "qwen3_vl": "Qwen3-VL", "smolvlm": "SmolVLM2"}

# highest recorded state per family; ranked, with canonical at the top
_STATE_RANK = {
    "implemented": 0,
    "cpu_tested": 1,
    "gpu_smoke": 2,
    "reduced_validation": 3,
    "canonical_validated": 4,
}
_DERIVED_LABEL = {
    "implemented": "implemented",
    "cpu_tested": "cpu_tested",
    "gpu_smoke": "gpu_smoke",
    "reduced_validation": "reduced_validation",
    "canonical_validated": "validated",
    "blocked": "blocked",
}


def capability_state(supported: bool) -> str:
    """Capability cell: implementation-only vocabulary, never 'validated'."""
    return "implemented" if supported else "unsupported"


def family_state(metadata, family: str) -> str:
    """Highest evidence state recorded for one family (capability not enough)."""
    states = [record.state for record in metadata.evidence if record.family == family]
    ranked = [state for state in states if state in _STATE_RANK]
    if ranked:
        return _DERIVED_LABEL[max(ranked, key=lambda state: _STATE_RANK[state])]
    if states:
        return _DERIVED_LABEL["blocked"]
    return "unsupported"


def _table(columns: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    lines = ["| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |")
    return "\n".join(lines)


def render() -> str:
    entries = [(name, get_algorithm(name).metadata) for name in available_algorithms()]

    capability_rows = []
    family_rows = []
    evidence_rows = []
    for name, metadata in entries:
        caps = metadata.capabilities
        capability_rows.append(
            [
                name,
                capability_state(bool(caps.get("image"))),
                capability_state(bool(caps.get("multi_image"))),
                capability_state(bool(caps.get("video"))),
                capability_state(bool(caps.get("eval"))),
                capability_state(bool(caps.get("export"))),
            ]
        )
        family_rows.append([name] + [family_state(metadata, family) for family in FAMILY_ORDER])
        for record in metadata.evidence:
            if record.state not in EVIDENCE_STATES:
                raise ValueError(f"{name}: unknown evidence state {record.state!r}")
            evidence_rows.append(
                [
                    name,
                    FAMILY_LABELS.get(record.family, record.family),
                    record.layout,
                    record.feature_mode,
                    record.state,
                    record.evidence,
                ]
            )

    sections = [
        HEADER,
        "## Capabilities (implementation only - not evidence)\n",
        _table(CAPABILITY_COLUMNS, capability_rows),
        "",
        "## Recorded evidence by family (highest state; details below)\n",
        _table(FAMILY_COLUMNS, family_rows),
        "",
        FOOTNOTES,
        "## Evidence records\n",
        _table(EVIDENCE_COLUMNS, evidence_rows),
        "",
        TAIL,
    ]
    return "\n".join(sections)


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

