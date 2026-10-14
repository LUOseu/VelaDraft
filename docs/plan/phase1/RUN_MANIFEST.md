# Phase-1 run manifest (self-contained)

Inventory of every recorded run and CPU-side result this phase package relies
on. "GPU run" rows are the recorded `R-20261014-*` jobs (full protocol and
result detail: `../RUN_LOG.md`; they were executed in the *previous*
work-package round and are reused here as report evidence, not re-run). The
phase itself (2026-10-14, `cvpr-route`) used **0 GPU-hours**: all fresh work
is CPU tests and docs.

## GPU runs (reused evidence, from `../RUN_LOG.md` / `RESOURCE_LEDGER.md`)

| Run ID | Purpose | Key result | Artifacts |
| --- | --- | --- | --- |
| R-20261014-01 | DFly recipe smoke | sanity drop 0.926, slot0 0.9986 | `recipe-smoke-config/` logs |
| R-20261014-02 | controlled_ce 1-step memory probe | peak 31.4 GB; 0.002 GPU-h | `~/vlm-recipe-smoke/runs/mem_ce_mb16` |
| R-20261014-03..06 | E1 DFly validation (select-parity, resume, export, peak) | bit-exact parity/resume; export metadata chain | E1 archive on GPU host |
| R-20261014-07 | duplicate E1 set (second session) | same conclusions bit-exactly | `e1_evidence.tar.gz` |
| R-20261014-08 | E2 multi-image/multi-turn smoke | token-exact 4/4 both checkpoints; 512/512 image tokens, 2 spans | `e2_final_evidence.tar.gz`; run dir `~/vlm-recipe-smoke/runs/e2/smoke_multiturn` |
| R-20261014-09 / 13 | zoo re-download validations | sha256 match / clean demo reload | `SHA256SUMS`, zoo pages |
| R-20261014-10 | E3 DocVQA/ChartQA vs frozen DFlash | AL 1.201 / 1.221, token-exact 128/128 | `benchmarks/reports/e3_dflash_docvqa_chartqa.json` |
| R-20261014-11 | E4 DFly `upstream_two_stage` canonical | AL **1.839** vs 1.682 (+9.3%), token-exact 128/128, export verified | run dir `~/vlm-dflash-dflare/runs/qwen25vl3b_dfly_2k_recipe/`; `e4_evidence.tar.gz` (sha256 `20f35f28...`) |
| R-20261014-12 | WP4 batched online capture A/B + profiling | 16 -> 4 forwards, losses bit-identical; extraction-wall % superseded by R2 | `~/vlm-recipe-smoke/runs/wp4/{ab1,ab4}`; `wp4_profile.log` |
| R-20261014-14..16 | WP5 vLLM bridge / MRoPE arm / margin probe | loads + runs; token-exact **5/32** (gate NOT passed) | `/tmp/wp5/bridge_check.json`; `wp5_evidence.tar.gz` (sha `2e2687d9...`) |
| R-20261014-17 | WP6 SmolVLM2 smoke | capture 1377/1377 tokens, train 32 steps, eval token-exact 4/4 | `e6_evidence.tar.gz` (sha256 `750db94d...`); `~/vlm-recipe-smoke/runs/e6/smoke_smolvlm` |

## Phase-1 CPU results (this round, `cvpr-route`)

| Item | Evidence | Location |
| --- | --- | --- |
| DSpark faithful objective (`train_recipe: upstream_dspark`) | code + tests; formula-vs-reference, chunk invariance, gradient separation, plan-digest sensitivity | commit `a80f564`; `tests/test_objectives.py`, `tests/test_dspark.py` |
| Per-method training-signal declarations | `TrainingSignals` on `AlgorithmMetadata` + generated doc + drift test | commit `64cef1d`; `docs/TRAINING_SIGNALS.md`, `scripts/render_training_signals.py` |
| Multimodal supervision representation | design note + 9 new tests (mask/input invariance, loss/grad connection, late-image rejection, exact resume, export round-trip) | commit `b5691e5`; `tests/test_multimodal_supervision.py`, `docs/plan/WP2_DESIGN.md` |
| Export records visual conditioning | `visual_conditioning` block + model-card line | commit `b5691e5`; `vlm_dflash/export/hf.py` |
| Full CPU regression | 453 passed / 6 skipped (`pytest -o addopts="" -q`; 443 at the `191da28` package commit, +10 length-bucket profiler tests) | local run 2026-10-14, this worktree |
| Competitive audit | pinned findings for 5 upstream projects | `COMPETITIVE_AUDIT.md` |
| Profiling + cache decision | cost shares and plan comparison on recorded numbers | `PROFILING.md`, `CACHE_DECISION.md` |

## Artifact access notes (self-contained)

- GPU-host archives live under `a6000xuyang:~/recipe-smoke-config/*.tar.gz`
  (see `../REVIEW_REMEDIATION.md` for the extraction command block). SSH was
  unreachable from the remediation host at wrap-up; nothing in this package
  requires re-reading them - every quoted number is already in the repo docs
  listed above.
- Zoo weights: GitHub release assets `zoo-v0.3.1` / `zoo-v0.3.2` with
  `SHA256SUMS`; re-download validations `R-20261014-09` / `R-20261014-13`.
- The phase-1 CPU work needs no GPU-host access to reproduce: `pytest` on this
  commit.
