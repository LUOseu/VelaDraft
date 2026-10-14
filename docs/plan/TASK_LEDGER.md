# VelaDraft plan - task ledger

Statuses: `DONE` / `IN_PROGRESS` / `PENDING` / `GPU_PENDING` / `BLOCKED`.
Baseline `9137de0`. Every "done" line points at code/tests/runs, not prose.

## Capability gap table (verified against the worktree on 2026-10-14)

| Capability | Verified baseline state | This plan |
| --- | --- | --- |
| 7 algorithms unified train/eval/export | `SUPPORT_MATRIX.md` generated from `AlgorithmMetadata` (capability vs evidence separated, review R6); all 7 through Export (DFlash2 training marked experimental; DFly recipe validated with the R-20261014-11 canonical run) | keep interfaces |
| Offline features + online live target capture | `vlm_dflash/target/{capture,online}.py`, `docs/online-training.md` | WP4 batch online capture |
| DFlare/DFly review fixes | merged (`ff406bb` via PR #2 = baseline) | read-only |
| DFly upstream two-stage recipe | ported on `dfly-upstream-recipe` (WP1, not yet on main) | WP1 |
| Image input | single image + text only (`multi_image: unsupported` in every `AlgorithmMetadata`) | WP2 |
| TextVQA benchmark | adapter + pinned 128-prompt manifest validated (`benchmarks/manifests/02_textvqa.json`) | reuse (WP3) |
| DocVQA / ChartQA | manifests `planned`; no adapter | WP3 |
| Serving bridge | `docs/serving-bridge.md` = proposal only (SGLang/vLLM thin adapter scoped); update 2026-10-14: first build landed for vLLM 0.22.1 native `dflash` - loads + measures, token-exact gate NOT passed | WP5 |
| Target family | Qwen2.5-VL-3B + Qwen3-VL-8B only | WP6 adds SmolVLM2-2.2B-Instruct (single-image smoke validated) |
| Docs homepage | capability-first README (AL ranking removed in `8d6c54c`) | keep |

## Work packages

| WP | Deliverable | Status | Evidence |
| --- | --- | --- | --- |
| WP0 | Baseline audit, task ledger, budget, capability list | DONE | this directory; CPU suite green (321 tests) |
| WP1 | DFly staged recipe + reference numeric validation + resume + export metadata | DONE (PR opened for review) | code `66b595a`, docs `c425850`, zoo export entries `6cf8141`, smoke `R-20261014-01`; E1 PASSED twice (`R-20261014-03..06` and the duplicate set `R-20261014-07`, bit-exact select-parity / resume / export metadata) |
| WP2 | Multi-image / multi-turn: schema v2, data, features, supervision, position protocol | DONE (smoke-validated) | code `6fd243c` (schema v2 + 14 tests) + `77813cc` (smoke generator/config) + the capability flip; 349 CPU tests green; E2 smoke PASS `R-20261014-08` (2-image / 2-turn, 16+4 rows: regen bookkeeping, capture visual swap check 512/512 tokens over 2 spans, 32-step train, token-exact 4/4 for trained + init checkpoints, manifest `schema_version: 2` / `num_multi_turn: 16`); design + protocol in `WP2_DESIGN.md`. Smoke-scale only; quality evidence is a later item |
| WP3 | Reuse benchmark harness; DocVQA/ChartQA adapters; training-cost report | DONE | adapters `export_docvqa` / `export_chartqa` + `prepare_benchmark` registration, new CPU tests on synthetic parquet fixtures; pinned exports + prompt manifests (`benchmarks/prompts/{docvqa,chartqa}_v1.jsonl`); E3 `R-20261014-10` PASS on the frozen 3B/2K DFlash (DocVQA AL 1.201 / ChartQA AL 1.221, token-exact 128/128 each; manifests 03/04 `validated`; `BENCHMARKS.md` section); training-cost report `docs/training-costs.md` landed (`fb3c46e`, C-01a13452) with the E4 side-by-side |
| WP4 | Batched online capture + at most one profiled I/O optimization + measurement report | DONE (timing metric corrected by R2) | code `95d6f2e` + `c28ad14` (`prepare=`/`finalize_prepared_records` split, `forward_and_capture_batch`, `fetch_batch`, `OnlineBatchIterator` with DataLoader-order parity + capture-group chunking, digest coupling); 5 new CPU tests (342 green); measurement `R-20261014-12`: 16 -> 4 forwards, losses bit-identical, peak unchanged (the extraction-wall 2.842 -> 2.308 s / -18.8% is superseded - R2 timer-boundary mismatch; unified boundary + fake-clock CPU tests landed, GPU re-measure pending); "at most one I/O optimization" resolved to zero with profiling evidence (render ~17 ms vs forward 129-139 ms/row) - report `WP4_MEASUREMENT.md` |
| WP5 | One fixed runtime (SGLang or vLLM) DFlash real deployment loop | BLOCKED (partial delivered: per the plan's blocking rule the implementable part is done and the gap is registered; not marked done) | vLLM 0.22.1 native `dflash`: converter `vlm_dflash/export/vllm_bridge.py` + CLI `scripts/export_vllm_draft.py` + `scripts/vllm_bridge_check.py` / `probe_divergence_margin.py` + 5 CPU tests; R-14 loads + runs (shared embed/lm_head, aux (2,18,34), n_spec 7) but **token-exact 5/32** (AL 1.4059 vs in-repo 1.6924/1.6866, same weights/prompts); R-15 excludes MRoPE-vs-1D (delta 0.006); R-16 observed small (<=0.125) top1-top2 margins at 4 traced flips (HF-assisted diagnostic; root cause unresolved; v2 frozen-prefix replay probe + causal gate implemented, CPU-tested); gaps registered in `WP5_REPORT.md`; build status in `docs/serving-bridge.md` |
| WP6 | One non-Qwen VLM family, DFlash first | DONE (single-image smoke) | family adapter `vlm_dflash/adapters/smolvlm.py` + contract/capture/eval/regen/visual-check hooks (`ef07b41`, `0f45fa1` + fixes); 14 CPU tests in `tests/test_smolvlm_adapter.py`; `R-20261014-17` PASS: regen 16/16 kept, capture image-token check 1377/1377 + `visual_path_check`, 32-step train, eval token-exact 4/4 on both checkpoints; gaps (schema-v2 multi-turn template drift, multi-image, online mode untested, smoke-scale only) in `docs/plan/WP6_REPORT.md` |
| WP7 | Reference weights, fixed experiments, model cards, paper outline + evidence table | DONE (CPU/docs, no GPU charge) | reference weights: `zoo-v0.3.1` + `zoo-v0.3.2` live, staged recipe tarball sha256 matches `SHA256SUMS` (`7d72357d...`, re-checked 2026-10-14), re-download validations `R-09`/`R-13`; model cards: 9 under `docs/zoo/` = README-linked set; fixed experiments recap E0-E4 (all landed, `RUN_LOG.md`); `docs/plan/PAPER_OUTLINE.md` (per-claim evidence table with engineering vs paper-evidence status, positioning, upgrade paths) + `WP7_REPORT.md` |

## Fixed experiment protocol (E0-E4, reconstructed)

Executed in order; each has a fixed run ID, budget and success criterion in
RUN_LOG.md before launch. Negative results are kept and recorded.

| ID | What | Success criterion | Budget |
| --- | --- | --- | --- |
| E0 | CPU regression suite on the working branch (no GPU) | `pytest tests/ -q` exit 0 | 0 GPU-h |
| E1 | WP1 validation on the real target: (a) `train_recipe: controlled_ce` select-parity vs legacy CE loss trace; (b) 4->8-step resume round-trip across the stage boundary gate; (c) export + reload metadata records `objective_profile`; (d) peak-memory re-measure | (a) same loss trace within bf16 protocol reproducibility; (b) resume log matches straight-through run; (c) export reload token-exact on a tiny prompt set, metadata present; (d) fits 48 GB | 0.3 GPU-h |
| E2 | WP2 multi-image/multi-turn end-to-end smoke (prepare -> capture -> train -> eval) | token-exact eval on the smoke protocol; schema v2 round-trip | 0.5 GPU-h |
| E3 | WP3 DocVQA + ChartQA adapter validation against a frozen DFlash checkpoint | token-exact vs the in-repo verifier on pinned prompt sets; reports under `benchmarks/` | 0.7 GPU-h |
| E4 | DFly 2K canonical recipe run vs matched controlled_ce baseline | DONE PASS (`R-20261014-11`, 1.45 GPU-h): both train + eval; AL **1.839** vs 1.682 (+0.157 / +9.3%), token-exact 128/128; export + reload verified | 1.0 GPU-h + reserve |

(WP4-WP6 measurement/verification jobs are budgeted in RESOURCE_LEDGER.md and
logged individually in RUN_LOG.md; they are not additional protocol rows.)
