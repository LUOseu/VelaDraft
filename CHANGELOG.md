# Changelog

## v0.2.0 (in development)

Qwen3-VL support, verification modes, scaling evidence, and the algorithm
registry.

Added
- Qwen3-VL-8B-Instruct target adapter (interleaved MRoPE, untied head handled
  end to end: untied targets train against their own `lm_head`).
- Unified CLI: `vlm-dflash {doctor,train,eval,export}` on top of the staged
  scripts; presets `smoke` / `single-gpu-24g` / `single-gpu-48g`.
- Verification modes `sequential` / `parallel` / `strict_parallel` with a
  measured runtime probe: parallel verify is **not bitwise-lossless**
  (documented in `docs/REFERENCE.md`); sequential remains the correctness gate.
- Resume (model/optimizer/scheduler/RNG), self-describing HF-style export,
  `scripts/summarize_runs.py`, and the scaling suite script.
- **Algorithm registry** (`vlm_dflash/algorithms/`): DFlash is now a plugin
  behind a documented interface; checkpoints carry an `algorithm` tag and
  eval/export refuse mismatched checkpoints. No DFlash semantics changed.
- **Benchmark harness skeleton** (`scripts/benchmark.py`,
  `benchmarks/manifests/`): fixed-protocol per-workload reports (JSON + MD).
- Governance docs: `STATE.md`, `SUPPORT_MATRIX.md`, `BENCHMARKS.md`,
  `docs/adding_an_algorithm.md`, `docs/algorithms/dflash.md`,
  `docs/benchmark.md`.
- `pyproject.toml` version aligned with the README (`0.2.0`).

Measured
- Qwen2.5-VL-3B 2K-corpus run: AL 1.630, 128/128 token-exact, per-slot
  acceptance non-zero through slot 5; 5K run in progress.
- Qwen3-VL-8B 512-corpus run: AL 1.322 (sequential verification).

Known limitations
- Single-image only; no video; no serving integration.
- bf16 capture/training is protocol-reproducible, not bit-identical
  (run-to-run AL noise < ~2%).
- Parallel verification is experimental; do not quote it as lossless speedup.

## v0.1.0

First minimal vertical slice: Qwen2.5-VL-3B, single image, target-native
regen, staged fused hidden-state capture, DFlash drafter training (target off
GPU), token-exact acceptance eval, provenance gates, visual-path checks.
