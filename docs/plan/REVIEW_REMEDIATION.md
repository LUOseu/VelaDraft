# Review remediation - R1-R6 against the acceptance review of `950b394`

Review source: `VelaDraft_Review_950b394.md` (external acceptance review,
2026-10-14). Review baseline `950b394d113ea094517639c8cdeec2d08842f7a0`
(compared against `9137de0`). This document tracks the remediation work:
scope, per-item status, migration/compatibility impact and the traceability
back to the original work packages.

## Inputs

Received:

- `VelaDraft_01_PLAN.md`, `VelaDraft_05_AGENT_PROMPT.md` (previous round)
- `VelaDraft_Review_950b394.md` (this review)

**Missing inputs (recorded, not reconstructed):** the original
`VelaDraft_02_WORK_PACKAGES.md`, `VelaDraft_03_EXPERIMENTS.md`,
`VelaDraft_04_ACCEPTANCE.md` and `VelaDraft_EXECUTION_CONFIG.yaml` were not
delivered to this round either. The CPU scope of the review can be completed
without them; the E0-E4 mapping below stays labelled "reconstructed" and no
original acceptance item is claimed as passed.

## Per-item status

Status vocabulary: `DONE` (implemented + CPU test evidence in this branch),
`PARTIAL` (implemented, GPU evidence outstanding and registered in the
GPU_PENDING list), `BLOCKED` (waiting on an external condition),
`PENDING` (not started).

| Item | Severity | Status | Evidence |
| --- | --- | --- | --- |
| R1 image basename collision | P1 | PENDING | |
| R2 online timing boundary | P1 | PENDING | |
| R3 multi-turn default semantics | P1 | PENDING | |
| R4 interleaved layout / unsupported media | P1 | PENDING | |
| R5 vLLM attribution | P2 | PENDING | |
| R6 support matrix capability vs evidence | P2 | PENDING | |
| Traceability (review §4) | - | PENDING | |
| GPU_PENDING list (review §6-F) | - | PENDING | |

## Compatibility and migration impact

Filled in per item as the fixes land; see the "Migration" note in each
section below.

