# Roadmap

Direction: one training stack for multimodal parallel drafters — but only
"support" claims backed by measured evidence. DFlash is frozen as the
reference algorithm; new algorithms plug in through the registry.

## Phase A — freeze + foundations (current)
- [x] Freeze DFlash semantics; document checkpoint contract.
- [x] Algorithm registry + `--algorithm` plumbing + algorithm tags.
- [x] Package/release consistency (version, governance docs).
- [~] Benchmark harness skeleton (manifests + runner + report).
- [ ] Scale evidence: finish 5K (3B) and 2K (8B) runs; record in BENCHMARKS.md.

## Phase B — first new algorithm (DSpark)
- Upstream audit first (`docs/algorithms/dspark.md`): paper, official repo,
  vLLM Speculators / NeMo implementations, license, training objective.
- Port behind the registry; same-corpus, same-seed comparison vs DFlash.
- Kill criteria: no measurable acceptance gain at equal params/latency, or no
  upstream objective to reproduce -> document and stop.

## Phase C — Domino + three-algorithm scoreboard
- Domino port + same-protocol scoreboard (AL, slots, params, train time,
  VRAM, runtime latency).

## Phase D — model zoo + distribution
- Public checkpoints (3B DFlash first, 8B flagship), model cards, HF
  collection, PyPI, 5-minute demo, README redesign.

## Later (not on the critical path)
- JetSpec / DFlash2 (experimental), multi-image, video, vLLM/SGLang
  interop, hardware presets validated by measurement.

Rules
- No support checkbox without a run: implemented / tested / validated /
  experimental / planned are distinct states.
- Every algorithm lands with its `docs/algorithms/<name>.md` audit.
- Re-check scope weekly against STATE.md "next highest-value tasks".
