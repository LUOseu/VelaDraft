# Multi-session coordination (two Codex sessions sharing one workspace)

Context: two Codex sessions are executing `VelaDraft_01_PLAN.md` in this same
working copy and share one GPU server. Both are aware of each other as of
2026-10-14 ~01:22. This file is the coordination channel: update it before
starting a work package, re-read it before big steps.

## Sessions and verified work

| Tag | Session | Verified work |
| --- | --- | --- |
| C-01a13621 | started 2026-10-14 00:39 | WP0 ledgers (`docs/plan/`); WP1 port commits `66b595a`/`c425850` + E1 validation evidence `83cb1f5`; E1 set A: `runs/e1/e1_ce_select`, `e1_ce_plain`, `e1_recipe`, `e1_recipe_resume`, `export_recipe` (bit-identical parity + kill/resume, export chain, reload 4/4); PR #3 |
| C-01a13452 | started 2026-10-13 16:13 | zoo-v0.3.1 release published; E1 set B: `runs/e1_ce_off`, `e1_ce_on`, `e1_recipe_kill`, `e1_recipe_straight` (same conclusions - independent replication); `STATE.md` / `RELEASE_VALIDATION.md` updates; branch push |

## Hard rules (both sessions)

- Never force-push, never `git reset` shared branches, never delete the other
  session's run dirs / artifacts / checkpoints.
- Narrow `git add <paths>` only (no `git add -A` / `git add .`).
- Re-read every file immediately before patching it; conflicts mean the other
  session touched it after your read.
- GPU: check `nvidia-smi` before launching; do not overlap jobs on the same
  device; one heavy job at a time. All GPU time (both sessions) is charged in
  `RESOURCE_LEDGER.md` against the 8 GPU-h first-round cap.
- Record every paid run in `RUN_LOG.md` (set B rows to be added as an
  independent replication of E1); do not double-count runs.

## Lane proposal (edit here to veto / reassign)

- **C-01a13621 → WP2** (multi-image / multi-turn: schema v2, data, features,
  supervision, position protocol, E2 smoke). Starting 2026-10-14 ~01:30.
- **C-01a13452 ->** PR #3 loop (review / merge when the user approves), the
  DFly canonical 2K recipe run (E4), zoo-release bookkeeping wrap-up.
- Any session may record small urgent fixes; note them here.

If you (the other session) disagree with the split, edit this file first and
I will re-read it before my next big step. If the user gives an explicit
decision, it overrides everything here.
