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
- [x] Release commit `5b1c41b` tagged `v0.3.0-rc1`, pushed to `LUOseu/VelaDraft` (formerly `vlm-dflash`; visibility: public since 2026-10-13; main `5ca075f`, CI green)

## v0.3.1 (distribution fixes + first-run review, in development)

- [x] Wheel ships packaged resources (presets + tested constraints); fresh-venv
      load verified outside the repo; CI asserts wheel contents + loads presets
      from a repo-less CWD
- [x] Single version source (`vlm_dflash.__version__` -> pyproject dynamic +
      `--version`); test pins metadata == module version
- [x] Identity gate ignores audit-only `target_model_path` (same model in a new
      directory passes; content/revision changes still fail); negative tests
- [x] Input-addressed stage caches for data/regen/capture with refusal on
      changed inputs and `--force` override; tests for all four states
- [x] `train --features <dir>` reuse interface + provenance record; library API
      verified on GPU (one capture, DFlash + DSpark trained from it)
- [x] Algorithm metadata / SUPPORT_MATRIX / doctor / export model cards
      consistent (`reduced_validation` state; matrix generated + drift test)
- [x] Model Zoo downloads published (`zoo-v0.3.0` release: DFlash + Domino
      exports); both re-downloaded from the public URL, sha256-verified and run
      token-exact (demo) on the reference GPU
- [x] `demo --drafter` follows the drafter's recorded algorithm
- [x] README restructured (pretrained demo first, pipeline SVG, Zoo table,
      split benchmark tables, JSON/CSV results), Chinese README, examples
      verified on GPU, serving-bridge proposal doc
- [x] Full CPU test suite green on the reference host; CI updated
