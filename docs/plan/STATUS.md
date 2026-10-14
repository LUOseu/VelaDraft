# VelaDraft plan - execution status (short, current)

Updated: 2026-10-14 (Asia/Shanghai). Plan baseline: `9137de0` (= `origin/main`).
Local dev: Mac M4 (CPU only). GPU: `ssh a6000xuyang` (4x RTX A6000 48GB,
shared server; `/home` is ~100% full, see RESOURCE_LEDGER).

> Provenance note: the plan's companion documents (`VelaDraft_02_WORK_PACKAGES.md`,
> `VelaDraft_03_EXPERIMENTS.md`, `VelaDraft_04_ACCEPTANCE.md`,
> `VelaDraft_EXECUTION_CONFIG.yaml`, `VelaDraft_06_PROGRESS_TEMPLATES.md`) were
> not supplied to the executing agent. This directory reconstructs the task
> ledger, run log, resource ledger, claim-evidence table and the fixed
> experiment protocol (E0-E4) from `VelaDraft_01_PLAN.md` and
> `VelaDraft_05_AGENT_PROMPT.md` only. Reconcile against the originals if they
> surface.

## Now

- **CVPR-route phase-1 round (2026-10-14, branch `cvpr-route`, CPU only - 0
  new GPU-h; budget then 4.10/8.00):** route-document §15 items 1-7 delivered -
  `upstream_dspark` faithful objective (`a80f564`), per-method
  training-signal declarations (`64cef1d`), multimodal-supervision design
  note + invariants (`b5691e5`) - plus the self-contained package
  `docs/plan/phase1/` (report, claim ledger, run manifest, resource
  estimate, profiling, cache decision, competitive audit, GO_NO_GO). CPU
  regression 453 passed / 6 skipped at the package commit (464 / 6 on the
  current branch head, `757f509`). Verdict there: **NO-GO for the formal
  matrix / big-compute request**; the bounded pending list (2.2-3.75 GPU-h,
  incl. the #7 second seed and the CPU-tested length-bucket profiler mode)
  was the recommended next spend. Start with
  `docs/plan/phase1/LOCAL_PILOT_REPORT.md`.
  The bounded pending list ran as an idempotent host queue
  (`~/recipe-smoke-config/cvpr_queue.sh`, run IDs R-18..R-23): #1 `R-19`
  PASS (post-R3/R4 E2 recheck + flex-kernel guard), #2 `R-21` measured
  (extraction wall -6.8% under the unified boundary, 128 -> 32 target
  forwards), #3 `R-22` PASS (export closure: fresh-process reload demo
  token-exact 4/4, same-condition eval reproduces AL 1.182/1.083; demo-config
  adoption fix `6619476` + regression tests), #5 `R-18` PASS
  (`upstream_dspark` smoke, `confidence_bce` 0.2042), #6 `R-20` measured
  (length-bucket render/forward attribution), #7 `R-23` PASS (seed-43
  recipe run: AL 1.830 vs init 1.003, token-exact 128/128 - the two-seed
  G4 condition is now met alongside `R-11`).
  #4 (vLLM divergence) stayed interactive-only as `R-24/R-25`: no
  bridge/acceptance logic error observed within the sampled range (5 prompts
  / 467 spec steps); token-exactness retired as a criterion (bf16 ties +
  engine nondeterminism); WP5 blocked on the -17% AL gap, with a complete
  serving correctness verification outstanding.
- **Review remediation round (2026-10-14, branch `review-remediation` off
  `950b394`): R1-R6 implemented.** R1 collision-safe media + audit script
  (`7cc9e5b`), R4 ordered segments + explicit unsupported-media rejection
  (`47b0075`), R3 `last_turn_only` regen default with `all_turns` opt-in
  (`c2d98e5`), R2 unified online timing boundary + WP4 supersession
  (`7038c56`), R5 frozen-prefix probe v2 + narrowed vLLM attribution
  (`7a44301`), R6 support matrix capability vs evidence (`b3730d7`).
  Full CPU regression 411 passed / 6 skipped; no new GPU jobs (budget then
  4.10/8.00 GPU-h) - GPU re-checks listed in
  `docs/plan/REVIEW_REMEDIATION.md` (GPU_PENDING section). WP5 stays BLOCKED
  (5/32 token-exact at that time, root cause then unresolved; the later
  `R-20261014-24/25` diagnosis retired token-exactness as a criterion and
  narrowed the blocker to the AL gap - see the `cvpr-route` bullet above).

- WP0 done: capability gap table + ledgers below; CPU baseline suite green
  (321 tests collected, all pass; `pytest tests/ -q`).
- WP1 done, pushed: upstream two-stage DFly recipe on branch
  `dfly-upstream-recipe` (`66b595a` code+tests, `c425850` docs, `6cf8141` zoo
  export entries). E1 validation PASSED (R-20261014-03..06): controlled_ce
  select-parity bit-identical, kill+resume bit-identical across the stage
  boundary, changed `--steps` resume refused, export/checkpoint/model-card
  metadata chain verified, reload token-exact. The other concurrent session's
  independent E1 set (R-20261014-07) reached the same conclusions bit-exactly.
  PR opened on the pushed branch for user review.
- Model Zoo `zoo-v0.3.1` published (six canonical archives + `SHA256SUMS`;
  DFly archive rebuilt for the ported-recipe note, `code commit` `6cf8141`);
  README + `docs/zoo` pages + `RELEASE_VALIDATION` updated on this branch.
- WP2 done, smoke-validated: schema v2 multi-image / multi-turn through
  prepare -> regen -> capture -> train -> eval. E2 PASS (`R-20261014-08`,
  two passes 0.55 GPU-h): 2-image / 2-turn protocol, capture visual swap
  check 512/512 tokens over 2 spans, 32-step train (loss 14.5 -> 3.64),
  eval token-exact 4/4 for trained + init checkpoints, manifest
  `schema_version: 2` / `num_multi_turn: 16`. `multi_image: true` in all 7
  `AlgorithmMetadata` + regenerated `SUPPORT_MATRIX.md` footnote (DFlash
  scope); 349 CPU tests green. Smoke-scale only - no quality/speed claim.
  Pass 1 (placeholder eval histories) archived as insufficient evidence.
- WP3 adapters done + E3 PASS (`R-20261014-10`, 0.15 GPU-h): DocVQA +
  ChartQA pinned 128-prompt exports (seed-42 subsets, shard/row hashes
  pinned) against the frozen 3B/2K DFlash checkpoint - DocVQA AL 1.201 /
  ChartQA AL 1.221, both token-exact 128/128 (sequential parity); manifests
  03/04 `validated`; numbers in `BENCHMARKS.md`. Training-cost report
  (`WP3a`, `docs/training-costs.md`) landed with the other session.
- E4 done, PASS (`R-20261014-11`, 1.45 GPU-h): canonical DFly 2K
  `upstream_two_stage` run - AL **1.839** vs the controlled_ce record 1.682
  (+0.157 / +9.3%), every slot up, token-exact 128/128, export + reload
  verified; costs in `docs/training-costs.md`, numbers in `BENCHMARKS.md`.
  Zoo bookkeeping closed out (re-download validation `R-20261014-09`) and
  `zoo-v0.3.2` published with the recipe checkpoint + clean re-download
  validation (`R-20261014-13`).
- WP4 batched online capture done + measured (`R-20261014-12`, 0.21 GPU-h):
  `train.online_batch_size=N` renders per sample, runs one padded target
  forward per capture group (group size `min(N, micro_batch)`), and slices
  records back per row; the trainer walks the same sampler order as the
  per-sample `DataLoader` (parity pinned by tests). A/B on the pinned E2
  slice: 16 -> 4 target forwards, losses bit-identical, peak unchanged (the
  reported extraction-wall 2.842 -> 2.308 s / "-18.8%" is superseded - R2
  timer-boundary mismatch; re-measurement pending); profiling `scripts/profile_online_extraction.py`
  shows prefill-bound extraction (render ~17 ms vs forward 129-139 ms/row),
  so the "at most one further I/O optimization" resolved to zero with
  evidence (report `docs/plan/WP4_MEASUREMENT.md`). 342 CPU tests green.
- WP5 first bridge built + measured (`R-20261014-14..16`, 0.55 GPU-h):
  vLLM 0.22.1 native `dflash` accepts the converted zoo export
  (`scripts/export_vllm_draft.py`; shared embed/lm_head, aux (2,18,34),
  n_spec 7) and runs, but the strict gate fails: **5/32 token-exact**, AL
  1.4059 vs in-repo 1.6924 (MRoPE) / 1.6866 (1-D arm, R-15 - rope convention
  is not the driver), traced flips show small top1-top2 margins (R-16 HF-assisted probe; root cause unresolved per the R5 remediation).
  Registered as BLOCKED-with-gap per the plan rule; report
  `docs/plan/WP5_REPORT.md`, converter/harness code + 5 CPU tests in this
  branch. Lane split recap in COORDINATION.md.
  Diagnosis `R-20261014-24/25` (0.30 GPU-h): no bridge/acceptance logic
  error observed within the sampled range (5 prompts / 467 spec steps;
  same-engine reruns agree only 12/32; 0 genuine inconsistencies, 14 exact
  bf16 ties); the token-exact gate is retired as a criterion. WP5 remains
  BLOCKED on the unexplained AL gap, and a complete serving correctness
  verification remains outstanding.
- WP6 done, engineering gate PASS (`R-20261014-17`, 0.53 GPU-h): first
  non-Qwen family SmolVLM2-2.2B-Instruct through prepare -> regen (16/16) ->
  capture (image-token check **1377/1377**, `visual_path_check` passed) ->
  32-step train -> sequential eval **token-exact 4/4** on trained + init
  checkpoints (AL 1.182 / 1.083); parallel diagnostic 1/2 (0.0-margin flip,
  small-margin flip class cf. R-16). Family adapter + hooks + 14 CPU tests in this
  branch (`ef07b41`, `0f45fa1` + fixes). Gaps registered (schema-v2
  multi-turn template drift, multi-image, online mode untested for the
  family; smoke scale - no quality claim): `docs/plan/WP6_REPORT.md`.
  Next: WP7.
- WP7 done (CPU/docs only, no GPU charge): reference-weights audit (`zoo-v0.3.1`
  / `zoo-v0.3.2` live; staged recipe tarball sha matches `SHA256SUMS`),
  model-card audit (9 cards = the README-linked set), fixed-experiment recap,
  and **`docs/plan/PAPER_OUTLINE.md`** - candidate main line, per-claim
  evidence table with engineering vs paper-evidence status tracked separately,
  honest positioning (systems / technical report level today) and the
  upgrade paths. Report `docs/plan/WP7_REPORT.md`. All work packages now
  have a terminal state (WP5 = BLOCKED-with-gap per the plan rule).

## Budget

8 GPU-h first-round cap; **6.25 GPU-h charged** (E1 0.59 incl. the duplicate
concurrent set, zoo re-download validation 0.04, E2 0.55, E3 0.15, E4 1.45,
WP4 measurement 0.21, WP5 0.55, WP6 0.53, cvpr-route queue R-18..R-23 1.85,
WP5 in-engine diagnosis R-24/25 0.30); **1.75 GPU-h remaining**; allocation
in RESOURCE_LEDGER.md.

## Blockers / risks

- **Concurrent session detected (2026-10-14 ~01:20).** Another live Codex
  session (thread `01a13452-83a9-7ef2-9c26-959c0cc66c3d`, started 2026-10-13
  16:13) is executing the same program in this same repo / server. It finished
  its own E1 set at 01:16:47 (`~/recipe-smoke-config/e1_runs.log`: runs
  `e1_ce_off`, `e1_ce_on`, `e1_recipe_kill`, `e1_recipe_straight`, same
  conclusions as this session's E1), committed `bee335f` + `6cf8141` on this
  branch and pushed `origin/dfly-upstream-recipe`; it holds uncommitted zoo
  model-card / README edits. This session (thread `01a13621`) ran an
  independent E1 set (`runs/e1/e1_ce_select`, `e1_ce_plain`, `e1_recipe`,
  `e1_recipe_resume`, `export_recipe`). Both E1 sets ran on GPU 0
  concurrently (no failures observed; do not repeat). Push happened; PR /
  further GPU jobs / WP2 design work held until the user rules on which
  session continues.
  - Update (02:20): the other session (`01a13452`) published zoo-v0.3.1
    (`7eab1c4` etc.), clean-machine re-download-verified it (`R-20261014-09`)
    and launched E4 (canonical DFly 2K recipe) on GPU2 at 01:26. This session
    finished WP2 + E2 (`R-20261014-08` PASS) on GPU0, then WP3/E3.
  - Update (03:00): E4 landed PASS (`R-20261014-11`: AL **1.839** vs 1.682,
    1.45 GPU-h) together with the WP3a cost report and the zoo bookkeeping
    wrap-up - the other session's lane is complete pending PR #3 review. The
    lane split holds; re-read `COORDINATION.md` before the next big step.
- Shared-server disk: `/home` 100% used and drifting down (130 GB free at
  01:30 -> 61 GB at 02:20 after pruning E2's renewable `ckpt_last`; a
  host-wide writeback stall slowed the E2 eval ~5 min). Keep net-new
  artifacts small; prune this plan's own renewable temp files after each job.
- WP5 blocker (runtime numeric parity): the vLLM 0.22.1 dflash path is not
  token-exact vs its own target-only greedy (5/32) - `R-20261014-24/25`
  showed this is bf16 ties + engine run-to-run nondeterminism (same-engine
  reruns 12/32; instrumented dump 0 genuine inconsistencies in 467 steps),
  not a bridge defect - so token-exactness is retired as a criterion. The
  remaining blocker is the **AL gap: 1.4059 vs in-repo 1.6866/1.6924
  (-17%)**. MRoPE-vs-1D excluded (R-15); mask-input convention verified
  equal. Registry + candidate causes: `docs/plan/WP5_REPORT.md`.
- Recipe training at micro-batch 8 peaked at 46.5 GB on a 48 GB card (~3%
  margin). Do not raise the micro-batch for recipe runs.
- Companion plan documents absent (see note above).

## Pointers

- Task ledger: TASK_LEDGER.md - run log: RUN_LOG.md - budget/disk:
  RESOURCE_LEDGER.md - claim evidence: CLAIM_EVIDENCE.md - paper outline:
  PAPER_OUTLINE.md
