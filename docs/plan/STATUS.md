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
- Next after E1: WP2 (multi-image / multi-turn schema v2).

## Budget

8 GPU-h first-round cap; 0.59 GPU-h used (includes the duplicate concurrent
E1 set); allocation in RESOURCE_LEDGER.md.

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
  - Update: the other session (`01a13452`) finished its zoo-release work —
    `zoo-v0.3.1` published and its README / `docs/zoo` / release-evidence
    commits landed on this branch — and started no further GPU job; E4
    remains on hold.
- Shared-server disk: `/home` 100% used (130 GB free, moving). Keep net-new
  artifacts small; prune this plan's own renewable temp files after each job.
- Recipe training at micro-batch 8 peaked at 46.5 GB on a 48 GB card (~3%
  margin). Do not raise the micro-batch for recipe runs.
- Companion plan documents absent (see note above).

## Pointers

- Task ledger: TASK_LEDGER.md - run log: RUN_LOG.md - budget/disk:
  RESOURCE_LEDGER.md - claim evidence: CLAIM_EVIDENCE.md
