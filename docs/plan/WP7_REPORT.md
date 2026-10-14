# WP7 report - reference weights, model cards, paper outline and evidence table

Status: **DONE (CPU/docs only - no GPU charge; budget at the time 4.10/8.00
GPU-h - the later R-18..R-25 jobs bring the current total to 6.25/8.00, see
`RESOURCE_LEDGER.md`).**

## Delivered

- `docs/plan/PAPER_OUTLINE.md`: candidate main line, section plan, the
  per-claim evidence table (engineering status vs paper-evidence status),
  reference-model delivery state, honest positioning and upgrade paths.
- Reference-weights audit (2026-10-14, CPU): `zoo-v0.3.1` and `zoo-v0.3.2`
  release pages live (HTTP 200 on both tags; recipe asset redirects to the
  release CDN); staged `qwen25vl-3b-dfly-recipe.tar.gz` sha256
  `7d72357d...` matches its published `SHA256SUMS`. Earlier clean
  re-download validations (`R-20261014-09`, `R-20261014-13`) already covered
  sha-identical downloads, `load_export` and token-exact reload demos for
  both releases; no new download needed.
- Model-card audit: 9 cards under `docs/zoo/`, exactly the set linked from
  the README model-zoo table; the recipe card carries `objective_profile
  upstream_two_stage`, `code_commit 6ad8500`, target fingerprint, quality
  numbers and scope/limitations.
- Fixed-experiment recap (all E0-E4 landed; details in `RUN_LOG.md`):

| ID | Status | Evidence | Cost |
| --- | --- | --- | --- |
| E0 (CPU suite) | PASS | suite grows with the work; 367 collected / green at WP7 time | 0 |
| E1 (recipe validation) | PASS, twice | `R-20261014-03..06` + duplicate set `R-20261014-07` (bit-exact) | 0.29 + 0.26 GPU-h |
| E2 (multi-image/multi-turn smoke) | PASS (2nd pass) | `R-20261014-08`; pass 1 archived as insufficient evidence | 0.55 GPU-h |
| E3 (DocVQA/ChartQA) | PASS | `R-20261014-10` | 0.15 GPU-h |
| E4 (canonical recipe) | PASS | `R-20261014-11` | 1.45 GPU-h |

## Final capability status (per the agent-prompt closing requirement)

- **Usable now:** the seven-algorithm unified train/eval/export stack;
  DFly default CE and ported `upstream_two_stage` recipe (schema-v2
  multi-image / multi-turn data path); DocVQA/ChartQA benchmark exports;
  batched online capture; zoo releases v0.3.1/v0.3.2 with model cards.
- **CPU-validated only (GPU path unexercised):** batched-online
  record-equality tests; vLLM bridge converter/harness (5 CPU tests);
  SmolVLM2 online mode + pad-id clamp (CPU tests only).
- **Small-scale / smoke-only evidence:** E2/E3/E4 quality numbers are
  protocol- or single-seed scale; the SmolVLM2 family is an engineering
  smoke with no quality claim.
- **BLOCKED / GPU_PENDING:** the vLLM runtime bridge is
  BLOCKED-with-gap (loads and runs; small-margin observations at the traced
  flips; `R-20261014-24/25` showed the engine is not token-reproducible even
  against itself - same-engine 12/32, 0/467 genuine inconsistencies within
  the sampled range - so token-exactness is retired as a criterion and a
  complete serving correctness verification remains outstanding; the
  acceptance-length gap (-17%) is the registered blocker in
  `WP5_REPORT.md`). Nothing in this round is "production
  ready"; the 24 GB preset remains unvalidated on real hardware.
- **Engineering vs paper evidence:** engineering is done for WP0-WP4 and
  WP6 (WP5 partial per the blocking rule); paper evidence currently
  supports a systems / technical report with measured costs and one
  validated recipe - not a "first/SOTA/gain" claim. The upgrade list (seeds,
  parity, benchmark-scale multi-image, more families) is in
  `PAPER_OUTLINE.md`.

## Pointers

- Claims and gaps: `CLAIM_EVIDENCE.md` (C1-C7).
- Runs and costs: `RUN_LOG.md`, `RESOURCE_LEDGER.md`, `BENCHMARKS.md`,
  `docs/training-costs.md`.
- Negative result and runtime queue: `WP5_REPORT.md`.
- Family smoke and its registered gaps: `WP6_REPORT.md`.
