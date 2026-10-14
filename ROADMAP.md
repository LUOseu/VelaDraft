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
- [x] Benchmark harness (manifests + runner + report) + first public workload
      validated: TextVQA (128-prompt seeded subset, pinned prompt manifest,
      token-exact; see `RELEASE_VALIDATION.md`). DocVQA / ChartQA stay pinned
      protocols without adapters.
- [x] Qwen3-VL reduced-scale validation for DSpark / Domino / JetSpec / DFlash2
      (v0.3.0-rc1; results in `RELEASE_VALIDATION.md`).

## Release (next)
- Public checkpoints (3B zoo first, 8B DFlash), model cards (export already
  emits `model_card.md`), HF collection, PyPI, 5-minute demo, README front
  page polish.
- Release audit before any push; version/tag hygiene.

## v0.3.2 (in development)
- [x] Online target feature source (`train --mode online`): live
      teacher-forced extraction sharing the staged trainer; CPU tests + 3B/8B
      plumbing smokes (all five algorithms on 3B) in
      `docs/online-training.md`. Offline stays the default.
- [ ] Online-vs-offline wall-clock/disk curve on a real corpus (only when a
      canonical online run is scheduled).

## DFlare / DFly (AngelSpec ports)
- [x] DFlare: separate-context-KV attention + per-draft-layer softmax fusion
      over explicitly selected captured layers (superset capture +
      `context_layer_ids` subset); audit in `docs/upstream/dflare.md`,
      doc in `docs/algorithms/dflare.md`.
- [x] DFly: DFlash FC base + fusion residual + zero-init TreeFlash
      hidden-states correction (train-time hook; documented sequential-prefix
      serving rule). Audit in `docs/upstream/dfly.md`, doc in
      `docs/algorithms/dfly.md`.
- [x] Canonical 2K runs recorded on Qwen2.5-VL-3B (matched `[1,17,33]`
      context on a shared 5-layer capture superset): DFlare AL 1.637
      (baseline parity), DFly AL 1.682, both token-exact 128/128; the frozen
      DFlash gate reproduces 1.6298 bit-identically on the same branch.
- [x] Layer-rich (`[1,9,17,25,33]`: DFlare 1.670, DFly 1.709), ablations
      (fixed fusion 1.637, shared K/V 1.579, no correction 1.569) and
      Qwen3-VL reduced runs (DFlare 1.185, DFly 1.226) recorded; the
      target-family columns in `SUPPORT_MATRIX.md` are updated (Qwen2.5-VL
      validated, Qwen3-VL reduced validation).
- [x] DFly upstream two-stage objective (LK / D-PACE cold start, then e2e TV):
      ported as the config-selectable `algorithm.train_recipe:
      upstream_two_stage` (exact loss math in `vlm_dflash/objectives.py`;
      CPU-tested for formula parity, stage switching and resume gating, and
      exercised end to end on the tiny fixture). Runs without `train_recipe`
      keep the shared decayed block CE; a canonical GPU recipe run and its AL
      comparison are the next record to land.
- [x] Schema v2 (multi-image / multi-turn): rows with `image_turns` keep the
      full conversation; per-turn greedy regen with incremental `full_ids`
      (asserted against one full-conversation render), multi-span capture
      (`image_spans`, `turn_spans`, multi-segment `loss_mask`). Recorded E2
      smoke `R-20261014-08` (2-image / 2-turn, 16 + 4 rows, token-exact 4/4
      for trained + init checkpoints); DFlash scope note in
      `SUPPORT_MATRIX.md`.

## Later (not on the critical path)
- DFlash2 objective isolation (`unary_loss: ce` control) if the number is
  quoted as a method claim; JetSpec production parallel-tree runtime.
- Video, vLLM/SGLang interop, latency breakdowns, hardware presets validated
  by measurement; multi-image batched online capture (WP4).

Rules
- No support checkbox without a run: implemented / tested / validated /
  experimental / planned are distinct states.
- Every algorithm lands with its `docs/algorithms/<name>.md` audit.
- Re-check scope weekly against STATE.md "next highest-value tasks".
