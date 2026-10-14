# Paper outline and evidence table (WP7)

Built 2026-10-14 from the recorded runs of this round (E0-E4 plus the
WP1-WP6 jobs). Every row maps to `RUN_LOG.md` / `CLAIM_EVIDENCE.md` /
`BENCHMARKS.md` entries, not to prose. Engineering status and paper-evidence
status are kept as separate columns by design (plan section 5: implemented /
smoke-passed / popular is not paper evidence).

## Candidate main line

> Reproducible training infrastructure for multimodal block-parallel
> drafters: measured resource costs and reference training recipes.

Alternative framings, strongest evidence first (choose once a venue is
fixed):

1. Systems / technical report: infrastructure, protocols and cost tables.
2. Training-recipe report: the ported DFly two-stage recipe with the matched
   `controlled_ce` comparison.
3. Mixed: infrastructure + recipe + the vLLM bridge as a registered negative
   result / open problem.

## Section plan (draft)

1. Introduction and scope: frozen-VLM block-parallel (L2) drafters; seven
   methods; engineering-first workload (~80/20).
2. System: `DrafterAlgorithm` x `VLMAdapter` unification (train/eval/export);
   schema v2 multi-image / multi-turn; offline store + batched online capture.
3. Protocols: the fixed 2K program (prepare -> regen -> capture -> train ->
   eval), greedy sequential verification, token-exact gates; smoke protocols
   for capability work (`WP2_DESIGN.md`, `WP6_REPORT.md`).
4. Reference recipe: DFly `upstream_two_stage` vs `controlled_ce` at matched
   792 steps; quality plus training cost.
5. Efficiency measurements: batched online capture A/B and extraction
   profiling; per-algorithm training-cost table.
6. Runtime bridge: vLLM 0.22.1 native `dflash` loads and runs but is not
   token-exact against its own target; margin-probe attribution; what numeric
   parity would require.
7. Related work: AngelSpec, SpecForge, speculators/vLLM, MMSpec (read and
   cited, not copied - see plan section 10).
8. Limitations and reproducibility: single node, single seed, bf16 not
   bitwise-deterministic; smoke-scale items named as such.

## Evidence table

| Claim unit | Engineering status | Paper-evidence status | Key runs / artifacts | Cost | Gap to a paper-grade claim |
| --- | --- | --- | --- | --- | --- |
| Reusable unified infra for 7 drafter algorithms | DONE | artifact available (support matrix, zoo, tests) | `SUPPORT_MATRIX.md`, `RELEASE_VALIDATION.md`, zoo releases `zoo-v0.3.1`/`zoo-v0.3.2` | 0 (CPU/prior program) | protocol + cost framing written up in paper text |
| DFly recipe port is numerically faithful, resumable, export-safe | DONE (bit-exact parity, two independent sets) | correctness section ready | E1 `R-20261014-03..06` + duplicate set `R-07`: select-parity / kill+resume bit-identical, metadata chain, token-exact reload | 0.59 GPU-h | none for correctness; upstream cross-check optional |
| DFly recipe beats shared-CE at the 2K protocol | DONE | single-seed matched comparison (AL 1.839 vs 1.682, +9.3%, every slot up, token-exact 128/128) | E4 `R-20261014-11`; `BENCHMARKS.md`, `docs/training-costs.md` | 1.45 GPU-h | multi-seed / scale replication before a headline claim (seed-43 `R-20261014-23` added the trained-vs-init second seed - G4 met; the shared-CE comparison itself stays single-seed) |
| Schema v2 multi-image / multi-turn protocol | DONE (smoke-validated) | protocol description + smoke evidence | E2 `R-20261014-08`: 512/512-token span check over 2 spans, eval token-exact 4/4 both checkpoints; `WP2_DESIGN.md` | 0.55 GPU-h | benchmark-scale multi-image quality evidence |
| Public benchmarks (DocVQA / ChartQA) runnable end-to-end | DONE | single-checkpoint numbers (AL 1.201 / 1.221, token-exact 128/128) | E3 `R-20261014-10`; pinned manifests 03/04 | 0.15 GPU-h | other algorithms / seeds on the same workloads |
| Batched online capture lowers capture cost | DONE (extraction-wall % superseded by R2) | measured on a registered 16-row slice: 4x fewer target forwards (4-step window), prefill-bound profiling (the -18.8% extraction-wall figure is invalid - timer boundary mismatch); paired re-measurement `R-20261014-21` under the unified boundary: extraction wall -6.8%, 128 -> 32 target forwards, peak +0.001 GB, 32-step losses not bit-identical (first epoch only) | `R-20261014-12`, `R-20261014-21`; `WP4_MEASUREMENT.md` | 0.21 + 0.13 GPU-h | 2K-scale and multi-seed online A/B |
| Real serving engine runs the export | BLOCKED-with-gap (partial) | negative result recorded (loads/runs; engine not token-reproducible even against itself - same-engine reruns 12/32, instrumented dump 0/467 genuine inconsistencies within the sampled range (`R-20261014-24/25`); token-exactness retired as a criterion; a complete serving correctness verification remains outstanding; the AL gap is the blocker) | `R-20261014-14..16`, `R-20261014-24/25`; `WP5_REPORT.md` | 0.55 + 0.30 GPU-h | runtime numeric parity; then AL / speed claims |
| Non-Qwen family through the stack | DONE (single-image smoke) | engineering note only | `R-20261014-17`; `WP6_REPORT.md` | 0.53 GPU-h | multi-turn / multi-image support and quality evidence |
| Reference model releases | DONE | artifact available + re-download validated | zoo v0.3.1 (6 archives) / v0.3.2 (recipe checkpoint); `R-20261014-09`, `R-20261014-13` | 0.06 GPU-h (demos) | per-model benchmark tables are recorded for the 2K program |

Total spend for the whole round at outline time: **4.10 of 8.00 GPU-h**
(the later bounded pending list added 2.15 - current total 6.25 of 8.00;
`RESOURCE_LEDGER.md`).

## Reference models and model cards (delivery state)

- `zoo-v0.3.1`: six canonical archives on release (DSpark, JetSpec,
  DFlash2-exp, DFlare, DFly on Qwen2.5-VL-3B; DFlash on Qwen3-VL-8B), all
  `load_export`-verified and re-downloaded sha-identical on the reference
  node (`R-20261014-09`).
- `zoo-v0.3.2`: the DFly `upstream_two_stage` recipe checkpoint, re-exported
  from a git clone (`code_commit 6ad8500`), re-download-validated and
  demo-reloaded token-exact (`R-20261014-13`); staged tarball sha256
  `7d72357d...` matches its published `SHA256SUMS` (re-checked 2026-10-14).
- Model cards: 9 files under `docs/zoo/`, exactly the set linked from the
  README model-zoo table; the recipe card records `objective_profile`,
  `code_commit`, target fingerprint, quality numbers and scope/limitations.

## Positioning (honest current level)

With the evidence above the defensible framing is an open-source systems /
technical report: infrastructure, protocols, measured costs and one
validated reference recipe. The vLLM bridge is a registered negative result,
not a deployment claim. No row currently supports "first" / "SOTA" /
"significant gain" language.

What would upgrade it toward a research submission (needs new budget; not
authorized in this round):

1. Multi-seed replication of the recipe comparison (E4) and of E2/E3.
2. Resolution or precise characterization of the vLLM numeric gap.
3. Benchmark-scale multi-image / multi-turn quality evidence.
4. At least one more target family at benchmark scale (beyond the smoke).

## Queued items if the budget continues

- SmolVLM2: template-aware schema-v2 render/verify, multi-image span
  grouping, online-mode GPU check (`WP6_REPORT.md` gaps).
- vLLM: candidate fixes from `WP5_REPORT.md` section Gaps (verify-path
  precision, draft cache semantics) - one bounded retry each, per the plan
  retry rule.
- E2/E3-style multi-seed rows when a venue decision is made.
