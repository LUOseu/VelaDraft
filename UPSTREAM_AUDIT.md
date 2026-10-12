# UPSTREAM_AUDIT

Index of the per-algorithm upstream audits. Each linked doc records the paper,
official repository, reference implementation files, inspected commit, license,
backbone, extra modules, training objective, inference procedure, checkpoint
format, what we port, what we change (and why), and the parity level.

| Algorithm | Audit doc | Speculators | NeMo Automodel | Parity level |
| --- | --- | --- | --- | --- |
| DFlash | `docs/upstream/dflash.md` | 55ebbd5b | bb5b34ce | maintainer implementation (via fork port) |
| DSpark | `docs/upstream/dspark.md` | 55ebbd5b | bb5b34ce | maintainer-informed reconstruction (2 documented deviations) |
| Domino | `docs/upstream/domino.md` | — | bb5b34ce | maintainer implementation (ported semantics + same defaults) |
| JetSpec | `docs/upstream/jetspec.md` | — | bb5b34ce | maintainer implementation (training semantics + mask + metrics; runtime not ported) |
| DFlash2 | `docs/upstream/dflash2.md` | 55ebbd5b | — (has no dflash2 at bb5b34ce) | architecture: maintainer / objective: experimental reconstruction (undisclosed) / runtime: inferred |

Pinned commits

- `vllm-project/speculators` @ `55ebbd5b9b49e9bd223f47949c986aa4abfffbc3`
  (main, 2026-10-08; Apache-2.0)
- `NVIDIA-NeMo/Automodel` @ `bb5b34cee5f629aac00d2c61eaefe8e0ee94dace`
  (main; Apache-2.0)
- `jacky566/Automodel_lao` @ `d21fb5aa` — the read-only reference this repo's
  DFlash port was taken from (see `docs/REFERENCE.md`; never imported at
  runtime).

Rule (GOAL §2): no algorithm implementation starts before its audit doc exists
with concrete source file / commit references — not just conclusions.
