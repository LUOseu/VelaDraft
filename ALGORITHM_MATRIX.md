# ALGORITHM_MATRIX

Machine-checkable progress per algorithm. Statuses: `DONE` / `RUNNING` /
`PLANNED` / `N/A`. An algorithm may only be called **Supported** when every
row up to `Documentation` is `DONE` (GOAL §30); otherwise it stays
`Experimental` or `Architecture only`.

| Stage | dflash | dspark | domino | jetspec | dflash2 |
| --- | --- | --- | --- | --- | --- |
| Upstream audit doc | DONE | DONE | DONE | PLANNED (sources pinned) | PLANNED (sources pinned) |
| Architecture implemented | DONE | DONE | DONE | PLANNED | PLANNED |
| Loss implemented | DONE | DONE | DONE (dual-logit + curriculum) | PLANNED | PLANNED |
| Unit tests | DONE | DONE | DONE (11 tests) | PLANNED | PLANNED |
| Synthetic overfit (CPU) | DONE | DONE | DONE (overfit test) | PLANNED | PLANNED |
| Gradient coverage audit | DONE | DONE | DONE (10% audit point) | PLANNED | PLANNED |
| Real VLM smoke (Qwen2.5-VL-3B) | DONE | DONE (8/8 exact) | DONE (8/8 exact) | PLANNED | PLANNED |
| Visual-path check | DONE | DONE | DONE | PLANNED | PLANNED |
| Checkpoint save/load | DONE | DONE (roundtrip test) | DONE (eval loads ckpts) | PLANNED | PLANNED |
| Eval (canonical metrics) | DONE | DONE (AL 1.735) | DONE (AL 1.764) | PLANNED | PLANNED |
| Export | DONE | DONE (verified) | DONE (verified) | PLANNED | PLANNED |
| Documentation | DONE | DONE | DONE | PLANNED | PLANNED |
| 2K apples-to-apples benchmark | DONE (baseline) | DONE (+ ablation) | DONE (+ ablation) | PLANNED | PLANNED |
| Runtime integration | N/A (future) | N/A | N/A | N/A | N/A |

Notes

- DSpark smoke evidence: COCO 64-row pipeline (overfit sanity passed,
  drafter visual check passed), eval re-run token-exact 8/8 after the
  verify-backend parity fix; the 2K canonical run trains + evals now.
- Domino "Export / Eval / benchmark" rows flip after the smoke + 2K runs;
  its audit is completed in `docs/upstream/domino.md`.
- JetSpec / DFlash2 sources are pinned in `docs/upstream/*.md` at the commits
  recorded there; audits complete before implementation starts (GOAL §2).
