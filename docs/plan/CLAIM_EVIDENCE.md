# VelaDraft plan - claim / evidence table

No public claim is made until the evidence column is filled from recorded
runs. Engineering completeness is tracked separately (TASK_LEDGER.md) and does
not by itself constitute paper evidence.

| # | Candidate claim | Status | Evidence | Gap |
| --- | --- | --- | --- | --- |
| C1 | The repo is a reusable, reproducible training infra for VLM block-parallel drafters (7 algorithms, unified train/eval/export) | engineering DONE / paper evidence weak | SUPPORT_MATRIX, RELEASE_VALIDATION, zoo releases | needs the paper-side framing: protocol + cost data |
| C2 | DFly's ported upstream two-stage recipe is selectable and numerically faithful to upstream loss math | CPU DONE + E1 validated (two independent concurrent sets, bit-exact); quality evidence still smoke-scale | `tests/test_objectives.py` (formula oracles, chunk invariance); `R-20261014-01` (stage switch); `R-20261014-03..06` + `R-20261014-07` (select-parity, kill+resume round-trip, export metadata + token-exact reload eval) | canonical 2K run vs controlled_ce (E4) |
| C3 | Multi-image / multi-turn support with a documented position/supervision protocol | engineering DONE + E2 smoke validated (protocol-scale) | Schema v2 + 14 CPU tests (`tests/test_wp2_multiturn.py`: parse/render/regen/mask/span + multi-image MRoPE monotonicity and span separation on the tiny Qwen2.5-VL fixture); E2 smoke `R-20261014-08`: 2-image/2-turn rows end-to-end, capture visual swap check (512/512 tokens over 2 spans, replay bitwise equal), eval token-exact 4/4 both checkpoints, manifest v2; protocol in `WP2_DESIGN.md` | benchmark-scale quality evidence (DocVQA/ChartQA-style, via WP3/E3); batched online capture for multi-image rows (WP4) |
| C4 | DocVQA/ChartQA workloads are runnable and reproducible end-to-end | engineering DONE + E3 validated (single-checkpoint, protocol-scale) | Adapters + pinned exports (`benchmarks/prompts/{docvqa,chartqa}_v1.jsonl`, shard/row hashes in manifests 03/04); frozen-checkpoint benchmark `R-20261014-10`: DocVQA AL 1.201 / ChartQA AL 1.221, both token-exact 128/128 (sequential parity), report `benchmarks/reports/e3_dflash_docvqa_chartqa.json` | other algorithms (DSpark/Domino) not yet measured on these workloads under this plan; no quality claim beyond the frozen checkpoint |
| C5 | Batched online capture lowers capture cost measurably | NOT STARTED | - | WP4 measurement report |
| C6 | A real serving engine runs a zoo DFlash export with token-exact output | NOT STARTED | serving-bridge proposal | WP5 |
| C7 | The stack works on a non-Qwen VLM family | NOT STARTED | - | WP6 |

Rules in force: no "first / SOTA / significant gain / production-ready"
language without the corresponding recorded runs; training-cost and negative
results stay in the log even when the README does not show them.
