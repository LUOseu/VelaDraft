# Review remediation - R1-R6 against the acceptance review of `950b394`

Review source: `VelaDraft_Review_950b394.md` (external acceptance review,
2026-10-14). Review baseline `950b394d113ea094517639c8cdeec2d08842f7a0`
(compared against `9137de0`). Work branch: `review-remediation`, based
directly on `950b394` (= `origin/main` at branch time; the review's "keep
user commits" rule applies - nothing on `main` was reset or rewritten).

Scope rules for this round (review section 6): CPU-scope fixes plus
documentation/evidence alignment; no new drafter/VLA/large-scale research
lines; AL is not an acceptance criterion. No new GPU jobs were launched:
the GPU budget stands at **4.10 / 8.00 GPU-h** and the remaining GPU checks
are delivered as ready commands plus the GPU_PENDING list below.

## Inputs

Received: `VelaDraft_01_PLAN.md`, `VelaDraft_05_AGENT_PROMPT.md` (previous
round), `VelaDraft_Review_950b394.md` (this review).

**Missing inputs (recorded, not reconstructed):**
`VelaDraft_02_WORK_PACKAGES.md`, `VelaDraft_03_EXPERIMENTS.md`,
`VelaDraft_04_ACCEPTANCE.md` and `VelaDraft_EXECUTION_CONFIG.yaml` were not
delivered to this round either. Consequences, per the review section 4:

- the E0-E4 protocol in `TASK_LEDGER.md` stays labelled **reconstructed**;
- no original acceptance item is claimed as passed, and the reconstructed
  E-numbering is not redefined to claim original-protocol completion;
- the traceability below maps the review's WP0-WP7 verdicts to the branch
  state and leaves the un-completable items visible.

## Per-item status

Status vocabulary: `DONE` (implemented + CPU test evidence on this branch),
`PARTIAL` (implemented; GPU-side evidence outstanding and registered in the
GPU_PENDING list), `BLOCKED` (waiting on an external condition).

| Item | Severity | Status | Commit | Evidence |
| --- | --- | --- | --- | --- |
| R1 image basename collision | P1 | DONE | `7cc9e5b` | collision-safe media names `{safe(sample_id)}_{sha256[:16]}{suffix}`; reuse only after content-hash verify (loud fail on tamper); intentional duplicates dedupe to one verified file; manifest rows carry `image_sources`/`image_source`; duplicate-id warning; read-only `scripts/audit_prepared_media.py` (exit 0/2: missing file, hash mismatch, content-hash-name mismatch, reference_hash_conflict); `tests/test_prepare_media.py` (7 tests) |
| R2 online timing boundary | P1 | PARTIAL | `7038c56` | `extract()` = `prepare()` + `extract_prepared()` so both paths charge the same render/forward halves; `prepare_seconds`/`forward_seconds` timed with CUDA sync on both sides, `wall_seconds` = their sum; resume skips counted in `fast_forward_items` and consume no wall time; fake-clock test proves identical 3+2 s split for per-sample and batched fetch; WP4 `-18.8%` comparison marked SUPERSEDED (legs timed different boundaries) in `WP4_MEASUREMENT.md` + `RUN_LOG.md` R-12, raw logs kept; supersession notes in online docs/ledgers |
| R3 multi-turn default semantics | P1 | DONE | `c2d98e5` | `regen.multi_turn_mode` with `last_turn_only` as the default (dataset history verbatim; regenerate/supervise the last assistant turn only, single `generate` call) and `all_turns` as explicit opt-in; mode + `assistant_indices` recorded in regen rows, stats and capture manifests; mode is part of the capture/online digest (legacy digests stay byte-stable); switching mode refuses exact-resume; capture loss mask covers only the regenerated final turn; `tests/test_wp2_multiturn.py`, `tests/test_online_mode.py` |
| R4 interleaved layout / unsupported media | P1 | DONE | `47b0075` | schema parses content into ordered `segments` (typed parts and `<image>` splits; `images`/`image_turns` resolve to segment indexes); video/audio/unknown parts raise; assistant image markers raise; v1 rows require a leading placeholder (mid-text -> error with v2 conversion hint; llava/HF readers share the rule); `messages_from_row` renders segment order and `visual_check` passes images in appearance order; `tests/test_multiturn_layout.py` (15 tests incl. the hf data-source reader) |
| R5 vLLM attribution | P2 | DONE (WP5 stays BLOCKED) | `7a44301` | probe v2 replays the **frozen** target-only prefix in one teacher-forced forward (no regeneration; fed ids asserted); per-case causal gate (prompt len vs vLLM record, digest-deterministic preprocessing, prefix on the model's own greedy trajectory); failures carry `exclusion_reason` and are excluded from `summary.causal_cases`; top-k candidates/rank/probe position recorded; same-engine-only evidence marked `missing`; output stamped `hf-assisted-diagnostic`; `tests/test_probe_divergence_margin.py` (7 tests). WP5 report/ledgers now say "small margins observed, root cause unresolved"; token-exact gate stays **BLOCKED at 5/32** (no threshold relaxation) |
| R6 support matrix capability vs evidence | P2 | DONE | `b3730d7` | `AlgorithmMetadata.evidence` (typed `EvidenceRecord`: family x layout x feature_mode x state; states `implemented`/`cpu_tested`/`gpu_smoke`/`reduced_validation`/`canonical_validated`/`blocked`; each record carries a run/config/commit/protocol pointer); renderer splits capability table (implemented/unsupported only) from a derived per-family table and a full evidence-records table; `multi_image=True` can no longer render as validated; DFly recipe line fixed (E4 canonical run recorded); SmolVLM2 row added (single-image smoke); doctor + model cards consume the same `as_dict()` (additive `evidence` key); `tests/test_algorithms.py` (+4 tests incl. drift) |
| Traceability doc (review section 4) | - | DONE | this commit | this file |
| GPU_PENDING list (review section 6-F) | - | DONE | this commit | section below |

### Acceptance-threshold mapping (review section 6)

| Threshold | Where satisfied |
| --- | --- |
| R1: same-basename/different-bytes images not merged; intentional duplicates legal; idempotent rerun + duplicate ids deterministic | `tests/test_prepare_media.py` |
| R2: fake clock - equal work, same boundary; resume skips do not pollute extraction counters; sub-phase vs total defined | `test_online_extraction_boundary_fake_clock`, fast-forward accounting tests in `tests/test_online_mode.py` |
| R3: history answers preserved; no supervision outside the last turn; mode in data + train digest; mode switch not exact-resumable; all-turns only as explicit opt-in | `tests/test_wp2_multiturn.py`, `tests/test_online_mode.py` |
| R4: text/image order preserved; unknown/audio/video explicitly rejected; late images never leak to earlier targets; all four readers share rendering semantics | `tests/test_multiturn_layout.py` |
| R5: saved prefix actually enters the forward; prefix-mismatch records excluded from causal stats; cross-engine results marked auxiliary | `tests/test_probe_divergence_margin.py`; probe `evidence_class` field |
| R6: boolean capability never auto-upgrades the validation level; generated docs match implementation without stale state | `tests/test_algorithms.py` (`test_capability_flags_do_not_imply_validation`, matrix drift test) |

## Compatibility and migration impact

- **R1 (media materialisation):** prepared filenames change to
  `{safe(sample_id)}_{sha256[:16]}{suffix}`; manifests gain
  `image_sources`/`image_source`. Existing stores from older code should be
  re-prepared or checked with `scripts/audit_prepared_media.py` (read-only,
  exit 2 on drift). Stage manifests still guard inputs, so stale stores
  refuse before compute instead of silently mixing naming schemes.
- **R2 (metrics):** summaries gain `prepare_seconds` /
  `forward_seconds` / `fast_forward_items` and `wall_seconds` now means
  "prepare + forward" on every path. `extraction_wall_seconds_total` records
  written under the old mixed boundary (R-20261014-12, `WP4_MEASUREMENT.md`)
  are marked superseded - they must not be quoted as a batching saving.
  Number-parsing consumers see additive keys only.
- **R3 (regen semantics):** default `multi_turn_mode: last_turn_only` changes
  multi-turn regen outputs vs the previous all-turn behaviour; digests
  change, so a mode switch refuses exact-resume by design. Configs that want
  the old behaviour must set `regen.multi_turn_mode: all_turns` explicitly.
  Single-turn data is unaffected.
- **R4 (schema):** content is now parsed into ordered segments; v1 rows with
  a mid-text `<image>` placeholder are rejected with a conversion hint
  (previously silently reordered/lost). llava/HF readers share the rule;
  well-formed v1 (leading placeholder) and all v2 rows keep their previous
  rendering.
- **R5 (probe only):** `scripts/probe_divergence_margin.py` output format
  changes to `vlm-dflash-divergence-margin-v2` (adds gates/evidence marks;
  old artifacts remain valid records of the v1 design). No library runtime
  behaviour changes.
- **R6 (metadata):** `AlgorithmMetadata.as_dict()` gains an additive
  `evidence` list; DFly `maturity.training` moves
  `experimental -> validated` (E4 canonical run recorded). `SUPPORT_MATRIX.md`
  regenerates with new sections; `doctor`/model cards pick the field up
  automatically. Consumers with strict key allow-lists should accept the
  additive key.

## Work-package traceability (review section 4, mapped to this branch)

E0-E4 note: the reconstructed E0-E4 protocol lives in `TASK_LEDGER.md` with
its historical run IDs (`R-20261014-01..13`). The review's demand - keep the
original E numbering reserved, do not redefine the original protocol to
claim completion - is honoured: the mapping below is explicitly the
**reconstructed** one and is not represented as the original acceptance.

Review section 6-E labels per work package (`implemented` /
`CPU-verified` / `GPU-smoke` / `pending` / `blocked`; a work package can carry
several):

| WP | Status labels (review section 6-E) |
| --- | --- |
| WP0 | implemented / CPU-verified; pending: alignment against the original WP definitions (missing input files) |
| WP1 | implemented / CPU-verified; GPU evidence recorded (canonical E4 `R-20261014-11`) |
| WP2 | R1/R3/R4 fixes CPU-verified; recorded GPU-smoke (E2); pending: group/leakage-aware split, post-R3/R4 GPU recheck (GPU_PENDING #1) |
| WP3 | GPU-smoke recorded (pinned DocVQA/ChartQA, E3); pending: fixed caption set, public multi-image/multi-turn workload, full cost ledger |
| WP4 | implemented / CPU-verified; GPU-smoke recorded (timing superseded by R2 -> pending paired re-measurement, GPU_PENDING #2); pending: bucketing + bounded OOM retry |
| WP5 | blocked (token-exact gate 5/32; root cause unresolved; same-engine diagnosis = GPU_PENDING #4) |
| WP6 | GPU-smoke recorded (single-image, `R-20261014-17`); pending: export closure (GPU_PENDING #3), second algorithm, multi-image/online boundaries |
| WP7 | implemented / CPU-verified (docs/releases/matrix); pending: original E0-E4 alignment (missing inputs), public evidence bundle |

| Work package | Branch state | Review verdict | This round | Still open |
| --- | --- | --- | --- | --- |
| WP0 | Task/resource/run/evidence ledgers (`TASK_LEDGER.md`, `COORDINATION.md`, `RUN_LOG.md`, `CLAIM_EVIDENCE.md`, `RESOURCE_LEDGER.md`) | Base delivered, needs alignment with the original definitions | this traceability doc; claim ledger re-checked while narrowing R5/R6 claims | per-item traceability against the *original* WP definitions is impossible while the WP source files are missing - recorded, not fabricated |
| WP1 | DFly two-stage recipe (objectives, stage switch, teacher alignment, resume constraints, export metadata) | Strongest delivered core; real GPU records are report evidence | R6 fixes the stale "no canonical GPU run" text: E4 `R-20261014-11` is now in metadata, matrix and `docs/algorithms/dfly.md`; DFly training maturity -> validated | broader multi-seed recipe evidence (out of scope this round) |
| WP2 | Schema v2 multi-image/multi-turn + DFlash E2 smoke (`R-20261014-08`) | Partial - not acceptable as a whole | R1/R3/R4 implemented with tests (see above); E2 evidence entry annotated as pre-remediation protocol | group/leakage-aware train/val split (review section 4 note) - **not implemented**; kept in the ledger below |
| WP3 | DocVQA/ChartQA pinned prompts + reports (`R-20261014-10`) | Partial | no code change; claims trace to `benchmarks/` + RUN_LOG | caption fixed public prompt set; public multi-image/multi-turn workload; complete cost ledger - kept open |
| WP4 | Batched online capture + order/resume tests + small GPU A/B (`R-20261014-12`) | Implementation valid; the performance conclusion must be re-measured | R2 unified timing boundary + superseded banners (docs + run log) | paired GPU re-timing (GPU_PENDING #2); length/visual-token bucketing and bounded OOM retry remain unimplemented |
| WP5 | Converter + vLLM harness; failure documented | Correctly BLOCKED | R5 probe v2 + attribution narrowed to observations; gate stays 5/32 BLOCKED | same-engine first-divergence diagnosis inside vLLM (GPU_PENDING #4); AL gap (-17%) unexplained |
| WP6 | SmolVLM2 single-image smoke (`R-20261014-17`) | First family milestone; not the full acceptance | status represented as `gpu_smoke` in metadata (R6); no overclaims | export -> fresh-process reload -> eval (GPU_PENDING #3); a second algorithm on the family; multi-image/online boundaries for it |
| WP7 | Model cards, releases (`zoo-v0.3.1`/`v0.3.2`), paper outline | Docs/releases progressed; the original experiment matrix is not closed | R6 rebuilt the support matrix around capability vs evidence; E0-E4 caveat recorded here | alignment with the original E0-E4 definitions needs the missing input files; a compact public evidence bundle for *this* remediation is listed below |

### Unfinished items kept visible (not silently dropped)

1. **Data split isolation (WP2):** `prepare` still splits by row; the
   original image/conversation group split is not implemented, so the same
   image with different captions or slices of one conversation can straddle
   train/val. Not a regression from this round - an open original-scope item.
2. **WP4 bucketing + bounded OOM retry:** not implemented; the current
   narrow length distribution in the smoke data does not demonstrate the
   long tail.
3. **WP6 export closure:** single-image export -> new-process reload -> eval
   evidence is missing; multi-image unimplemented for the family; online
   untested on GPU; multi-turn render drift documented.
4. **Public evidence bundle for this remediation:** the runs involved are
   archived on the training host (`~/recipe-smoke-config/*.tar.gz`); a
   publishable compact bundle (resolved configs, commits, data manifest
   hashes, result JSONs, per-sample parity summaries, environment - private
   paths scrubbed, weights referenced from the checksummed zoo releases)
   has not been published yet. It needs (a) GPU-host access - SSH to
   `a6000xuyang` was unreachable from the remediation host at wrap-up time
   (connection closed, direct and via proxy) - and (b) a user decision on
   the publication target. No GPU compute required.

   ```bash
   # on the GPU host, once reachable:
   mkdir -p /tmp/evidence && cd /tmp/evidence
   for a in e1_evidence e2_final_evidence e3_evidence e4_evidence \
            wp4_ab1 wp4_ab4 wp5_evidence e6_evidence; do
     tar -xzf ~/recipe-smoke-config/${a}.tar.gz -C /tmp/evidence --one-top-level=${a}
   done
   # strip host paths/credentials, keep configs + result JSONs + manifests +
   # parity samples + environment records; then (after user approval):
   #   gh release create remediation-evidence-v1 /tmp/evidence/bundle.tar.gz
   ```

## GPU_PENDING (review section 6-F) - commands only, nothing auto-scheduled

Budget note: 4.10 / 8.00 GPU-h used; these items are **not authorised** by
this remediation and must not be launched without an explicit resource
decision. Order follows the review:

Scheduling status (2026-10-14, `cvpr-route` session): items #1-#3 and #5-#7
are staged as a stop-on-first-failure, restart-idempotent host queue (run IDs
R-18..R-23, `~/recipe-smoke-config/cvpr_queue.sh` on the GPU host) that
starts only when a GPU is idle and hard-stops at 3.3 h cumulative job wall,
inside the remaining 3.90 GPU-h of the first-round cap. #4 stays out of the
queue by design (interactive vLLM session). Results are folded back into
this package when they land; do not re-schedule these items elsewhere.

All commands assume the `review-remediation` checkout on the GPU host and
the env conventions of the target configs (`VLM_DFLASH_TARGET`, plus
`VLM_DFLASH_E2_SOURCE` / `VLM_DFLASH_E2_IMAGES` for E2 data).

1. **Qwen2.5-VL DFlash small multi-image/multi-turn correctness** under the
   post-R3/R4 protocol (regenerated data, new segment layout). Verify the
   R3/R4 semantics on GPU; not a quality run.

   ```bash
   python scripts/make_wp2_smoke_data.py --out data/e2_smoke --train 16 --val 4
   VLM_DFLASH_TARGET=<Qwen2.5-VL-3B dir> \
   VLM_DFLASH_E2_SOURCE=data/e2_smoke/source.jsonl \
   VLM_DFLASH_E2_IMAGES=data/e2_smoke/images \
     python -m vlm_dflash.cli train --config configs/smoke_multiturn.yaml \
     --run-dir runs/e2/smoke_multiturn_postfix
   python -m vlm_dflash.cli eval runs/e2/smoke_multiturn_postfix \
     --config configs/smoke_multiturn.yaml
   ```

   Pass criteria: eval `token_exact` 4/4 on both checkpoints; capture
   `visual_path_check` passed; `regen/regen_stats.json` shows 2 turns/row,
   16/16 kept; index manifest `schema_version: 2`.

2. **Paired single-vs-batched online re-timing** under the unified boundary
   (R2): same fixed rows, warmup pass dropped or recorded explicitly,
   CUDA-synchronised `prepare_seconds`/`forward_seconds`, identical
   checkpoints/IO.

   ```bash
   VLM_DFLASH_TARGET=<Qwen2.5-VL-3B dir> \
   VLM_DFLASH_E2_SOURCE=data/e2_smoke/source.jsonl \
   VLM_DFLASH_E2_IMAGES=data/e2_smoke/images \
     python -m vlm_dflash.cli train --config configs/smoke_multiturn.yaml \
     --mode online --set train.online_batch_size=1 --run-dir runs/wp4/ab1_v2
   # repeat the same command with --set train.online_batch_size=4 --run-dir runs/wp4/ab4_v2
   ```

   Compare `train/train_summary.json` -> `feature_source`:
   `extraction_prepare_seconds_total`, `extraction_forward_seconds_total`,
   `extraction_wall_seconds_total`, `fast_forward_items_total`, plus
   bit-identical step losses between legs. Do not re-use the superseded
   `-18.8%`.

3. **SmolVLM2 single-image export -> new-process reload -> same-condition
   eval**; explicitly label other layouts untested.

   ```bash
   python -m vlm_dflash.cli export runs/e6/smoke_smolvlm --out-dir /tmp/e6_export
   python -m vlm_dflash.cli demo --drafter /tmp/e6_export \
     --target <SmolVLM2 dir> --num-prompts 4          # run in a fresh process
   python -m vlm_dflash.cli eval runs/e6/smoke_smolvlm \
     --config configs/smoke_smolvlm.yaml
   ```

   Record: reload demo token-exact vs target-only greedy, exported metadata
   carries the SmolVLM contract, and the eval numbers match the pre-export
   checkpoint run.

4. **vLLM same-engine first-divergence diagnosis:** instrument vLLM 0.22.1's
   dflash proposer/verify path (or an equivalent in-engine dump) for
   per-step candidate logits, accepted positions and KV/position metadata on
   the same 32 prompts; keep the token-exact gate at 5/32 until closed; no
   large-scale throughput runs before the divergence is understood.

   ```bash
   # baseline reproduction (recorded as R-20261014-14)
   python scripts/vllm_bridge_check.py --target <Qwen2.5-VL-3B> \
     --draft /tmp/wp5/vllm_draft --source runs/qwen25vl3b_2k/data/val.jsonl \
     --images-dir runs/qwen25vl3b_2k/data/images --num-prompts 32 \
     --max-new-tokens 128 --min-pixels 200704 --max-pixels 1003520 \
     --out /tmp/wp5/bridge_check.json
   # then dump per-step verify candidates inside vLLM and re-read with the
   # HF-assisted v2 probe (does not establish root cause by itself):
   python scripts/probe_divergence_margin.py --run-dir /tmp/wp5/run_1d \
     --bridge-json /tmp/wp5/bridge_check.json --max-cases 4 \
     --out /tmp/wp5/divergence_margins_v2.json
   ```

5. **`train_recipe: upstream_dspark` GPU smoke** (new 2026-10-14, `cvpr-route`):
   the recipe's loss math is CPU-pinned (`tests/test_objectives.py`,
   `tests/test_dspark.py`); this run proves the trainer path on GPU - chunked
   teacher-logits reconstruction from the stored final hidden states and
   confidence-head gradient flow. Not a quality run; do not quote AL from it.

   ```bash
   # reuse the capture-v2 JetSpec 2K store ([1,17,33] + final hidden, 1050 rows)
   VLM_DFLASH_TARGET=<Qwen2.5-VL-3B dir> \
     python -m vlm_dflash.cli train \
       --config configs/dspark/qwen2_5_vl_3b_2k_recipe.yaml \
       --features ~/vlm-dflash/runs/qwen25vl3b_jetspec_2k/features \
       --steps 16 --set train.overfit_sanity.enabled=false \
       --run-dir runs/dspark/recipe_smoke
   python -m vlm_dflash.cli export runs/dspark/recipe_smoke --out-dir /tmp/dspark_export  # CPU
   ```

   Pass criteria: `nan_seen=false`; `train_summary.json` ->
   `train_recipe.name == upstream_dspark`; train-log rows carry finite
   `confidence_bce` / `markov_bias_abs_mean`; `train/gradient_coverage.json`
   covers the confidence head; `train/confidence_calibration.json` written;
   the log records the `--features` reuse (regen + capture skipped); export
   metadata carries the recipe. Optional: 4-prompt `demo --drafter` reload
   token-exact. Estimate 0.2-0.4 GPU-h (`docs/plan/phase1/RESOURCE_ESTIMATE.md`).

6. **Length-bucketed online profiling** (route-document §6 P2 gap;
   `docs/plan/phase1/PROFILING.md` §7): per-row render/forward attribution
   across pinned token-length buckets on the real target. The profiler's
   `--length-buckets` mode (added 2026-10-14, CPU-tested) provides it; the
   per-row mode is O(rows) single-sample passes, so pin a `--limit`.

   ```bash
   VLM_DFLASH_TARGET=<Qwen2.5-VL-3B dir> \
     python scripts/profile_online_extraction.py \
       --config configs/smoke_multiturn.yaml --run-dir runs/e2/smoke_multiturn \
       --length-buckets 512,1024 --render-repeats 2 --forward-repeats 3
   ```

   Record: per-bucket median render/forward ms and forward-ms-per-token; the
   batch-group numbers in the same report remain the trainer-facing view.
   Estimate 0.15-0.25 GPU-h.

7. **Second training seed for the 2K recipe** (route-document §6 P4 "seed
   波动" / §6.1 learning-effectiveness gate): rerun the canonical E4 protocol
   (`R-20261014-11`) with `train.seed=43` on the same store and data; train +
   the standard 128-prompt eval including the init control. This is the small
   seed repeat, not the §7 formal matrix. Mirror the recorded E4 invocation
   chain (`~/recipe-smoke-config/e4_train.log` on the GPU host).

   ```bash
   VLM_DFLASH_TARGET=<Qwen2.5-VL-3B dir> \
     python -m vlm_dflash.cli train \
       --config configs/dfly/qwen2_5_vl_3b_2k_recipe_matched.yaml \
       --features ~/vlm-dflash/runs/qwen25vl3b_jetspec_2k/features \
       --set train.seed=43 --set train.overfit_sanity.enabled=false \
       --run-dir runs/dfly/recipe_seed43
   python -m vlm_dflash.cli eval runs/dfly/recipe_seed43 \
     --config configs/dfly/qwen2_5_vl_3b_2k_recipe_matched.yaml
   ```

   Pass criteria: `nan_seen=false`; eval reports both checkpoints token-exact
   and `ckpt_final` AL clearly above `ckpt_init` AL (seed 42: 1.839 vs
   1.003). Estimate 0.9-1.4 GPU-h (E4 train minus sanity + eval legs).

## Delivery checklist (review section 6)

- Per-R commits, change descriptions and test evidence: table above.
  `git log --oneline 950b394..HEAD` lists (oldest first) `7cc9e5b` (R1),
  `47b0075` (R4), `c2d98e5` (R3), `7038c56` (R2), `7a44301` (R5),
  `b3730d7` (R6), `d9a202c` (traceability + GPU_PENDING), `3beafcf`
  (README), `5db73c7` (DFly docs), `f904953` (status labels + GPU commands),
  `7e89240` (hf-reader tests + ledger open items; `git log -1` on
  `review-remediation` gives the exact tip). Baseline `950b394` is
  untouched; `main` history was never rewritten.
- CPU regression: `pytest -o addopts="" -q` on this branch =
  **413 passed, 6 skipped** (411 at first pass + the two final-audit hf-reader
  tests; skips: `tests/test_contract_integration.py`
  real-model integration, requiring `VLM_DFLASH_TEST_MODEL` / a local
  Qwen3-VL directory - GPU-host-only, unchanged from the branch point's 6).
  No test expectations were rewritten to bypass acceptance semantics; the
  new tests were written against the reproduced defects before the fixes.
- README: no new capability claims added in this round; model-zoo links
  unchanged. The matrix/evidence and WP5/WP4 wording fixes live in
  `SUPPORT_MATRIX.md`, `docs/plan/*` and `docs/serving-bridge.md`.
- Negative/superseded results stay in reports (`WP4_MEASUREMENT.md`,
  `WP5_REPORT.md`) with raw logs referenced - not deleted, not re-labelled.

