# Release checklist — v0.3.0-rc1

Scope freeze in effect: bug fixes, documentation, packaging, release tooling,
benchmark adapters, reduced validation, CI, security/license cleanup only.

- [x] README consistent with measured results and support status
- [x] STATE.md consistent (release freeze header, current status)
- [x] support matrix regenerated from the release regression
- [x] CPU tests (155/155, exit 0)
- [x] GPU regression — 5 algorithms: checkpoint load, eval, export, export reload (all token-exact; frozen baseline bit-identical)
- [x] Qwen3-VL reduced validation — DSpark / Domino / JetSpec / DFlash2 (all token-exact, export + reload)
- [x] TextVQA public benchmark — DFlash / DSpark / Domino (JetSpec / DFlash2 not run: optional item, deferred)
- [x] Model zoo export (6 canonical checkpoints, reload self-check)
- [x] Five-minute demo measured (demo + train-demo)
- [x] Frozen DFlash baseline re-evaluated on the release commit (bit-identical, 128/128 exact)
- [x] Fresh wheel install + CLI help smoke on a clean venv
- [x] CI workflow present (.github/workflows/ci.yml)
- [x] License audit (NOTICE / UPSTREAM_AUDIT / docs/upstream)
- [x] Secret + privacy audit (tokens, keys, absolute personal paths)
- [x] Large-file audit (`find . -type f -size +20M`)
- [x] Claim audit (DFlash2 wording; no lossless-speedup claims; parallel marked experimental)
- [x] RELEASE_VALIDATION.md complete (only the tag-time commit hash was pending at review, filled in at tag time)
- [x] Release commit `5b1c41b` tagged `v0.3.0-rc1`, pushed to private GitHub repo (`LUOseu/vlm-dflash`, visibility: private; CI green)
