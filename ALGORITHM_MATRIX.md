# ALGORITHM_MATRIX

Machine-checkable progress per algorithm. Statuses: `DONE` / `RUNNING` /
`PLANNED` / `N/A`. An algorithm may only be called **Supported** when every
row up to `Documentation` is `DONE` (GOAL §30); otherwise it stays
`Experimental` or `Architecture only`.

| Stage | dflash | dspark | domino | jetspec | dflash2 |
| --- | --- | --- | --- | --- | --- |
| Upstream audit doc | DONE | DONE | PLANNED (sources pinned) | PLANNED (sources pinned) | PLANNED (sources pinned) |
| Architecture implemented | DONE | DONE | PLANNED | PLANNED | PLANNED |
| Loss implemented | DONE | DONE | PLANNED | PLANNED | PLANNED |
| Unit tests | DONE | DONE | PLANNED | PLANNED | PLANNED |
| Synthetic overfit (CPU) | DONE | DONE | PLANNED | PLANNED | PLANNED |
| Gradient coverage audit | DONE | DONE | PLANNED | PLANNED | PLANNED |
| Real VLM smoke (Qwen2.5-VL-3B) | DONE | PLANNED | PLANNED | PLANNED | PLANNED |
| Visual-path check | DONE | PLANNED | PLANNED | PLANNED | PLANNED |
| Checkpoint save/load | DONE | DONE (roundtrip test) | PLANNED | PLANNED | PLANNED |
| Eval (canonical metrics) | DONE | PLANNED | PLANNED | PLANNED | PLANNED |
| Export | DONE | DONE (loader dispatch) | PLANNED | PLANNED | PLANNED |
| Documentation | DONE | DONE | PLANNED | PLANNED | PLANNED |
| 2K apples-to-apples benchmark | DONE (baseline) | PLANNED | PLANNED | PLANNED | PLANNED |
| Runtime integration | N/A (future) | N/A | N/A | N/A | N/A |

Notes

- DSpark "Real VLM smoke / visual-path / eval / benchmark" rows flip once the
  `configs/dspark/*` pipeline runs complete on the server; results land in
  `BENCHMARKS.md` and `STATE.md`.
- Domino / JetSpec / DFlash2 sources are pinned in `docs/upstream/*.md` at the
  commits recorded there; their audits are completed before implementation
  starts (GOAL §2).
