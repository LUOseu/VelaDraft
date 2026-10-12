# Roadmap

Direction: one training stack for multimodal parallel drafters — but only
"support" claims backed by measured evidence. DFlash is frozen as the
reference algorithm; the other four plug in through the registry.

## Algorithm zoo (v0.3, this release)
- [x] DFlash frozen + machine-readable reference
      (`tests/reference_dflash/manifest.json`, post-refactor re-measurements).
- [x] Registry + `--algorithm` plumbing + algorithm tags + `doctor` checks.
- [x] Scale evidence: 3B 2K/5K and 8B 512/2K runs recorded in `BENCHMARKS.md`.
- [x] DSpark, Domino, JetSpec ported + validated on the shared 2K protocol
      (scoreboard + per-algorithm ablations in `BENCHMARKS.md`).
- [x] DFlash2 ported + measured (AL 1.954, token-exact) — **Experimental**:
      upstream publishes the architecture but not the training objective; the
      conv/selector decomposition is bounded by an ablation.
- [~] Benchmark harness (manifests + runner + report; public workloads are
      still smoke-scale).
- [ ] Qwen3-VL reduced-scale validation for DSpark / Domino / DFlash2.

## Release (next)
- Public checkpoints (3B zoo first, 8B DFlash), model cards (export already
  emits `model_card.md`), HF collection, PyPI, 5-minute demo, README front
  page polish.
- Release audit before any push; version/tag hygiene.

## Later (not on the critical path)
- DFlash2 objective isolation (`unary_loss: ce` control) if the number is
  quoted as a method claim; JetSpec production parallel-tree runtime.
- Multi-image, video, vLLM/SGLang interop, latency breakdowns, hardware
  presets validated by measurement.

Rules
- No support checkbox without a run: implemented / tested / validated /
  experimental / planned are distinct states.
- Every algorithm lands with its `docs/algorithms/<name>.md` audit.
- Re-check scope weekly against STATE.md "next highest-value tasks".
