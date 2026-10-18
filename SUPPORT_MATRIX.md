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

## Capabilities (implementation only - not evidence)

| Algorithm | Image       | Multi-image | Video       | Eval        | Export      |
| --------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| dflare    | implemented | implemented | unsupported | implemented | implemented |
| dflash    | implemented | implemented | unsupported | implemented | implemented |
| dflash2   | implemented | implemented | unsupported | implemented | implemented |
| dfly      | implemented | implemented | unsupported | implemented | implemented |
| domino    | implemented | implemented | unsupported | implemented | implemented |
| dspark    | implemented | implemented | unsupported | implemented | implemented |
| jetspec   | implemented | implemented | unsupported | implemented | implemented |

## Recorded evidence by family (highest state; details below)

| Algorithm | Qwen2.5-VL | Qwen3-VL           | SmolVLM2    |
| --------- | ---------- | ------------------ | ----------- |
| dflare    | validated  | reduced_validation | unsupported |
| dflash    | validated  | validated          | gpu_smoke   |
| dflash2   | validated  | reduced_validation | unsupported |
| dfly      | validated  | reduced_validation | unsupported |
| domino    | validated  | reduced_validation | unsupported |
| dspark    | validated  | reduced_validation | unsupported |
| jetspec   | validated  | reduced_validation | unsupported |

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

## Evidence records

| Algorithm | Family     | Layout                 | Mode    | State               | Evidence                                                                                                                                                                                                                                                                                             |
| --------- | ---------- | ---------------------- | ------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dflare    | Qwen2.5-VL | single_image           | offline | canonical_validated | docs/algorithms/dflare.md canonical 2K matched run (AL 1.637, 128/128; configs/dflare/qwen2_5_vl_3b_2k.yaml); zoo-v0.3.1 archive + demo reload                                                                                                                                                       |
| dflare    | Qwen3-VL   | single_image           | offline | reduced_validation  | docs/algorithms/dflare.md Qwen3-VL-8B reduced validation (configs/dflare/qwen3_vl_8b_reduced.yaml)                                                                                                                                                                                                   |
| dflash    | Qwen2.5-VL | single_image           | offline | canonical_validated | RELEASE_VALIDATION.md canonical benchmark (frozen 128p baseline AL 1.6298, 128/128 exact; tests/reference_dflash/manifest.json); zoo-v0.3.0 export reload                                                                                                                                            |
| dflash    | Qwen3-VL   | single_image           | offline | canonical_validated | RELEASE_VALIDATION.md targets (full 2K pipeline AL 1.665, 32/32 exact; zoo-v0.3.1 8B export reload)                                                                                                                                                                                                  |
| dflash    | Qwen2.5-VL | multi_image+multi_turn | offline | gpu_smoke           | R-20261014-08 E2 smoke (2-image/2-turn rows, 16 train + 4 val, token-exact eval; configs/smoke_multiturn.yaml; pre-R3/R4 remediations) + R-20261014-19 post-remediation recheck (fresh rebuild under the R3 last_turn_only protocol; one regenerated turn per row; token-exact 4/4 both checkpoints) |
| dflash    | SmolVLM2   | single_image           | offline | gpu_smoke           | R-20261014-17 (SmolVLM2-2.2B smoke: 16 train + 4 val synthetic rows, token-exact 4/4; single-image only; no quality claim)                                                                                                                                                                           |
| dflash2   | Qwen2.5-VL | single_image           | offline | canonical_validated | RELEASE_VALIDATION.md canonical benchmark (AL 1.954, 128/128; experimental objective reconstruction) + zoo-v0.3.1 archive reload                                                                                                                                                                     |
| dflash2   | Qwen3-VL   | single_image           | offline | reduced_validation  | RELEASE_VALIDATION.md Qwen3-VL-8B reduced validation (256-row slice, 16-prompt eval AL 1.2104, 16/16 exact, export + reload)                                                                                                                                                                         |
| dfly      | Qwen2.5-VL | single_image           | offline | canonical_validated | R-20261014-11 E4 canonical 2K upstream_two_stage recipe (AL 1.839 vs controlled_ce 1.682, 128/128 exact; configs/dfly/qwen2_5_vl_3b_2k_recipe.yaml); zoo-v0.3.2 archive + demo reload                                                                                                                |
| dfly      | Qwen3-VL   | single_image           | offline | reduced_validation  | docs/algorithms/dfly.md Qwen3-VL-8B reduced validation (configs/dfly/qwen3_vl_8b_reduced.yaml)                                                                                                                                                                                                       |
| domino    | Qwen2.5-VL | single_image           | offline | canonical_validated | RELEASE_VALIDATION.md canonical benchmark (AL 1.764, 128/128; TextVQA 1.268) + zoo-v0.3.0/v0.3.1 archive reload                                                                                                                                                                                      |
| domino    | Qwen3-VL   | single_image           | offline | reduced_validation  | RELEASE_VALIDATION.md Qwen3-VL-8B reduced validation (256-row slice, 16-prompt eval AL 1.2278, 16/16 exact, export + reload)                                                                                                                                                                         |
| dspark    | Qwen2.5-VL | single_image           | offline | canonical_validated | RELEASE_VALIDATION.md canonical benchmark (AL 1.735, 128/128; TextVQA 1.253) + zoo-v0.3.1 archive reload                                                                                                                                                                                             |
| dspark    | Qwen3-VL   | single_image           | offline | reduced_validation  | RELEASE_VALIDATION.md Qwen3-VL-8B reduced validation (256-row slice, 16-prompt eval AL 1.2212, 16/16 exact, export + reload)                                                                                                                                                                         |
| jetspec   | Qwen2.5-VL | single_image           | offline | canonical_validated | RELEASE_VALIDATION.md canonical benchmark (AL 1.599, 128/128; causal-mask serving - tree runtime not ported) + zoo-v0.3.1 archive reload                                                                                                                                                             |
| jetspec   | Qwen3-VL   | single_image           | offline | reduced_validation  | RELEASE_VALIDATION.md Qwen3-VL-8B reduced validation (256-row slice, 16-prompt eval AL 1.1185, 16/16 exact, export + reload)                                                                                                                                                                         |


## Targets

| Target | Adapter | Regen | Capture | Train | Eval | Export |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | qwen25_vl | validated | validated | validated | validated | validated |
| Qwen3-VL-8B-Instruct | qwen3_vl | validated | validated | validated | validated | validated |
| SmolVLM2-2.2B-Instruct | smolvlm | gpu_smoke | gpu_smoke | gpu_smoke | gpu_smoke | - |
| Qwen3.6-35B-A3B | qwen3_5_moe | implemented | implemented | implemented | blocked | - |

Qwen2.5-VL and Qwen3-VL are validated on single-image + text with greedy
decoding. Qwen3-VL was validated for the four zoo algorithms at reduced scale
(256-row slice, 16-prompt sequential eval, export + reload, all token-exact -
see `RELEASE_VALIDATION.md`); DFlash has the full 2K-protocol run there.
SmolVLM2 has one recorded engineering smoke (`R-20261014-17`: single-image,
16 train + 4 val synthetic rows, token-exact 4/4; multi-image and online mode
untested for this family; no quality claim) - see the evidence records.
Qwen3.5/3.6-MoE (`qwen3_5_moe`) is implemented and CPU-tested only: the
contract, the partial-rotary draft (64 of 256 head channels, interleaved
MRoPE `[11, 11, 10]`) and the dense draft width are checked against the
transformers implementation (`tests/test_qwen35_moe_target.py`). Spec-decode
eval is blocked: the hybrid Gated DeltaNet layers keep recurrent/conv states
that `DynamicCache.crop` cannot roll back after a partial acceptance.
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
