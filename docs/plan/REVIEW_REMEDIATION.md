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
| R4 interleaved layout / unsupported media | P1 | DONE | `47b0075` | schema parses content into ordered `segments` (typed parts and `<image>` splits; `images`/`image_turns` resolve to segment indexes); video/audio/unknown parts raise; assistant image markers raise; v1 rows require a leading placeholder (mid-text -> error with v2 conversion hint; llava/HF readers share the rule); `messages_from_row` renders segment order and `visual_check` passes images in appearance order; `tests/test_multiturn_layout.py` (13 tests) |
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
   hashes, result JSONs, parity summaries, environment) has not been
   published yet - needs a manual packaging/publishing step (no GPU work).

## GPU_PENDING (review section 6-F) - commands only, nothing auto-scheduled

Budget note: 4.10 / 8.00 GPU-h used; these items are **not authorised** by
this remediation and must not be launched without an explicit resource
decision. Order follows the review:

1. **Qwen2.5-VL DFlash small multi-image/multi-turn correctness** under the
   post-R3/R4 protocol (regenerated data, new segment layout): rerun the E2
   smoke equivalent on `configs/smoke_multiturn.yaml` and check token-exact
   eval + capture span checks. Purpose: verify the R3/R4 semantics on GPU,
   not to chase quality.
2. **Paired single-vs-batched online re-timing** under the unified boundary
   (R2): same fixed rows, warmup pass dropped or recorded, CUDA-synchronised
   `prepare_seconds`/`forward_seconds`, identical checkpoints/IO; report
   `prepare`/`forward` split and peak memory; do not re-use the superseded
   `-18.8%`.
3. **SmolVLM2 single-image export -> new-process reload -> same-condition
   eval**: proves the WP6 family's export closure; explicitly label other
   layouts as untested.
4. **vLLM same-engine first-divergence diagnosis:** instrument vLLM 0.22.1's
   dflash proposer/verify path (or an equivalent in-engine dump) for
   per-step candidate logits, accepted positions and KV/position metadata on
   the same 32 prompts; keep the token-exact gate at 5/32 until closed; no
   large-scale throughput runs before the divergence is understood.

## Delivery checklist (review section 6)

- Per-R commits, change descriptions and test evidence: table above;
  `git log --oneline 950b394..HEAD` lists `7cc9e5b`, `47b0075`, `c2d98e5`,
  `7038c56`, `7a44301`, `b3730d7` plus this doc's commit.
- CPU regression: `pytest -o addopts="" -q` on this branch =
  **411 passed, 6 skipped** (skips: `tests/test_contract_integration.py`
  real-model integration, requiring `VLM_DFLASH_TEST_MODEL` / a local
  Qwen3-VL directory - GPU-host-only, unchanged from the branch point's 6).
  No test expectations were rewritten to bypass acceptance semantics; the
  new tests were written against the reproduced defects before the fixes.
- README: no new capability claims added in this round; model-zoo links
  unchanged. The matrix/evidence and WP5/WP4 wording fixes live in
  `SUPPORT_MATRIX.md`, `docs/plan/*` and `docs/serving-bridge.md`.
- Negative/superseded results stay in reports (`WP4_MEASUREMENT.md`,
  `WP5_REPORT.md`) with raw logs referenced - not deleted, not re-labelled.

