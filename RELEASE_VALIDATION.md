# Release validation — vlm-dflash v0.3.0-rc1

## v0.3.1 — distribution fixes + first-run review (2026-10-13)

Host: same reference node (one RTX A6000 48 GB used at a time). No algorithm,
protocol or benchmark number changed in this round; all v0.3.0 records below
stand as measured.

| Item | Verification |
| --- | --- |
| Wheel resources | `uv build --wheel` ships `vlm_dflash/presets/*.yaml` + `vlm_dflash/resources/constraints-tested.txt`; fresh venv (`uv venv`, no repo on path) loads all three presets, parses the tested constraints and reports `vlm-dflash --version` 0.3.1 == package metadata |
| Identity portability | Same contract with `target_model_path` moved passes `assert_identity`; changed `model_fingerprint` still raises `IdentityMismatchError` (unit tests); stage digests ignore the path but change with any contract field or upstream stage digest |
| Stage cache safety | data/regen/capture record `<stage>/stage_manifest.json` input digests; changed dataset/config/`--limit` inputs refuse before compute with a `--force` hint; outputs without a recorded digest refuse; `--force` rebuilds (6 unit tests) |
| Feature reuse | `examples/reuse_features_compare.py` on GPU: one regen+capture (12-row COCO pool), then DFlash and DSpark trained from the same store via `train_dflash(..., features_dir=...)` — `provenance.json` records `external=true`, 12 samples, seq 1024, bf16, manifest digest; both runs evaluated token-exact (AL 1.153 / 1.094 at 5 steps, smoke scale). CLI form also verified on GPU: `train --preset smoke` capture, then `train --features <store> --algorithm dspark` logs "regen + capture stages skipped" and trains from the 8-sample store; re-running with a changed `--limit` refuses with the recorded-vs-current digest before any compute |
| Examples | `examples/score_pretrained.py` with the released DFlash export: AL 1.306, token-exact 4/4, per-slot `[0.28, 0.02, 0.01, …]`; `examples/train_from_jsonl.py` (12 COCO rows, 10 steps): loss 7.69 → 7.06, visual check passed, token-exact 4/4 |
| Model Zoo downloads | `zoo-v0.3.0` release published (DFlash 387 MB, Domino 464 MB). Both re-downloaded from the public URL on a clean machine, sha256-identical to the uploaded archive, transferred to the reference GPU node and run in fresh directories: `demo --drafter` (Qwen2.5-VL-3B target) token-exact 4/4; DFlash AL 1.313, Domino AL 1.384 (synthetic sample; algorithm auto-detected from the export) |
| Model Zoo v0.3.1 | `zoo-v0.3.1` release published with six canonical archives — DSpark (487 MB), JetSpec (370 MB), DFlash2-exp (498 MB), DFlare (356 MB), DFly (401 MB) on Qwen2.5-VL-3B and DFlash 8B (955 MB) on Qwen3-VL-8B — plus `SHA256SUMS`; every archive is a `load_export`-verified export (the DFly archive was rebuilt for the ported-recipe note and re-verified with a tar round-trip). Asset sizes verified against the staged tarballs via the release API; clean-machine re-download verification pending (the dev Mac's egress proxy cannot reach `release-assets.githubusercontent.com`) |
| docs metadata | SUPPORT_MATRIX regenerated from `AlgorithmMetadata` (drift test in CI); Qwen3-VL states retagged `reduced_validation` for DSpark/Domino/JetSpec/DFlash2 to match the recorded runs | 
| CPU tests | `pytest tests/ -q` green on the reference host with the v0.3.1 tree (174 collected; was 155 at v0.3.0-rc1) |

Release candidate: `v0.3.0-rc1`
Git commit: `5b1c41b9c5e5498830c0168213db3b3cbac123f6` (release commit; tag `v0.3.0-rc1`)
Date: 2026-10-12/13
Host: single GPU node (one RTX A6000 48 GB used at a time)

## Environment

```text
Python:       3.11.16 (uv venv)
PyTorch:      2.10.0+cu128
Transformers: 5.12.1
Datasets:     4.2.0
CUDA:         12.8 runtime (driver 580.126.09 / CUDA 13.0-capable)
GPU:          NVIDIA RTX A6000 48 GB
CPU-only tests: CUDA_VISIBLE_DEVICES="" (no GPU access)
```

## CPU tests

PASS — `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/ -q`
→ 155/155 collected tests, exit code 0 (pytest 9.1.1). Includes the frozen
DFlash reference manifest (`tests/reference_dflash/`) and the
overfit/trainer/registry/export/CLI unit tests.

## Package tests

PASS — `uv build --wheel --sdist` produces `vlm_dflash-0.3.0-py3-none-any.whl`
+ `.tar.gz`; the wheel installs into a fresh venv
(`uv venv /tmp/vdf_fresh && uv pip install --no-deps <wheel>`) and
`vlm-dflash --help` lists all six commands (`doctor`, `demo`, `train-demo`,
`train`, `eval`, `export`).

## GPU regression

PASS — release-commit regression on the canonical checkpoints (sequential
verification; every row token-exact):

| Check | Result |
| --- | --- |
| Frozen DFlash baseline — full canonical protocol (128 prompts / 128 tokens) | AL **1.6298365840801265**, bit-identical to the frozen reference; 128/128 token-exact; random-init 1.0029844936092909 |
| DSpark — 16-prompt re-run | AL 1.827, 16/16 (full-protocol reference 1.735) |
| Domino — 16-prompt re-run | AL 1.854, 16/16 (reference 1.764) |
| JetSpec — 16-prompt re-run | AL 1.661, 16/16 (reference 1.599) |
| DFlash2 — 16-prompt re-run | AL 2.047, 16/16 (reference 1.954) |
| Model zoo — export + reload self-check | 6/6 canonical exports reload (243.8M / 321.6M / 292.9M / 243.8M / 328.5M / 629.2M params; `logs/export_zoo.log`) |
| TextVQA public benchmark | PASS — see Public benchmark |
| Five-minute demo | PASS — see Five-minute demo |

The 16-prompt re-runs use the first 16 eval prompts, so their AL is a subset
reading (higher than the 128-prompt reference for the same checkpoint) — the
regression gate is token-exactness, not the AL value.

### Qwen3-VL-8B reduced validation (256-row slice, 192 steps, 16-prompt eval)

All four algorithms: prepare / regen / capture / train / visual check / eval /
export / reload — token-exact 16/16, loss decreasing, no NaN (overfit sanity
passed for each).

| Algorithm | final loss (first-10% -> last-10%) | eval AL (init 1.000) | export reload | peak VRAM |
| --- | --- | ---: | --- | ---: |
| DSpark | 6.53 (13.63 -> 6.38) | 1.2212 | 707.0M | 23.7 GB |
| Domino | 5.50 (13.04 -> 5.32) | 1.2278 | 685.1M | 23.6 GB |
| JetSpec | 6.04 (12.29 -> 5.87) | 1.1185 | 629.2M | 28.5 GB |
| DFlash2 | 3.55 (10.90 -> 3.46) | 1.2104 | 733.3M | 31.0 GB |

Shared-capture visual-path check: image-token span located 192/192, replay
bitwise-equal, image swap changes the span (max |Δ| 183.5, all 192 positions)
and the downstream text positions (max |Δ| 135.0, 177 positions, min cosine
0.41) — PASS.

## Algorithms

One row per algorithm × target (numbers are acceptance length, sequential
token-exact verification):

| Algorithm | Qwen2.5-VL-3B | Qwen3-VL-8B | Verdict |
| --- | --- | --- | --- |
| dflash | PASS — frozen 128p baseline bit-identical; zoo export reload | PASS — full 2K pipeline (32p AL 1.665); 8B export reload | PASS |
| dspark | PASS — 2K canonical 1.735 + release re-run 1.827 | PASS — reduced validation 1.2212 | PASS |
| domino | PASS — 2K canonical 1.764 + release re-run 1.854 | PASS — reduced validation 1.2278 | PASS |
| jetspec | PASS — 2K canonical 1.599 + release re-run 1.661 | PASS — reduced validation 1.1185 | PASS |
| dflash2 | PASS — 2K canonical 1.954 + release re-run 2.047 | PASS — reduced validation 1.2104 | PASS (experimental objective reconstruction) |

## Targets

| Target | Status | Evidence |
| --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | PASS | full zoo: 5 algorithms trained, evaluated (128 prompts, token-exact) and exported; canonical 2K corpus |
| Qwen3-VL-8B-Instruct | PASS | DFlash full pipeline (2K, AL 1.665, 32/32 exact, earlier run); reduced validation for DSpark / Domino / JetSpec / DFlash2 (256-row slice, 16-prompt eval, export + reload, all token-exact) |

## Canonical benchmark (frozen)

The frozen reference (`tests/reference_dflash/manifest.json`) and the zoo
scoreboard (Qwen2.5-VL-3B, 2K corpus, 128 prompts / 128 tokens, greedy,
sequential verification — token-exact for every row):

| Algorithm | AL | Token-exact | Notes |
| --- | ---: | --- | --- |
| DFlash (frozen baseline) | 1.630 | 128/128 | re-evaluated bit-identically post-zoo |
| DSpark | 1.735 | 128/128 | |
| Domino | 1.764 | 128/128 | |
| JetSpec | 1.599 | 128/128 | causal-mask serving (tree runtime not ported) |
| DFlash2 | 1.954 | 128/128 | experimental objective reconstruction |

PASS — the frozen DFlash baseline was re-evaluated on the release
commit (128 prompts / 128 tokens, sequential): AL **1.6298365840801265** —
bit-identical to the frozen reference in `tests/reference_dflash/manifest.json`
(random-init 1.0029844936092909); 128/128 token-exact.

## Public benchmark

PASS — TextVQA, pinned export (lmms-lab/textvqa validation split, 128-prompt
seed-42 subset; hashes in `benchmarks/prompts/textvqa_v1.jsonl`), greedy,
`max_new_tokens=32`, sequential verification, Qwen2.5-VL-3B zoo checkpoints:

| Algorithm | AL | token-exact | note |
| --- | ---: | --- | --- |
| DFlash | 1.263 | 128/128 | |
| DSpark | 1.253 | 128/128 | |
| Domino | 1.268 | 128/128 | |

Reports: `benchmarks/reports/textvqa_<run>.{json,md}` (regenerated per host;
numbers promoted to `BENCHMARKS.md`).

## Five-minute demo

PASS — measured on the reference host (RTX A6000 48 GB, Qwen2.5-VL-3B
target, synthetic 8-image sample, 4 prompts / max 64 new tokens; wall = end
to end including model load):

| command | wall | peak VRAM | result |
| --- | ---: | ---: | --- |
| `vlm-dflash demo` (random-init baseline) | 45 s | 8.2 GB | AL 1.000, 4/4 token-exact |
| `vlm-dflash demo --drafter assets/zoo/qwen25vl-3b-dflash` | 42 s | 8.5 GB | AL 1.313, 4/4 token-exact |
| `vlm-dflash train-demo --steps 80` | 3 min 13 s | 8.2 GB | loss 13.0 -> 0.013 (tiny-sample overfit), eval AL 1.138 vs random-init 1.000, 4/4 token-exact, export written |

Peak VRAM is total device usage (polled every 3 s) and fits comfortably in
24 GB. `--drafter` demos align the sample's image pixel bounds with the
drafter's recorded target identity and use a drafter-specific run directory
(`runs/demo_dflash_<drafter>`); the provenance gate stays hard.

## Known limitations

- Single image + text, greedy decoding; no video/multi-image/multi-turn.
- Two target families only (Qwen2.5-VL-3B, Qwen3-VL-8B).
- `sequential` verification is the correctness reference; `parallel` /
  `strict_parallel` are experimental throughput paths (bf16 near-ties can flip
  argmax), never mixed into correctness tables.
- DFlash2: architecture port with an experimental training reconstruction —
  upstream's training objective is not publicly disclosed; not an official
  reproduction.
- JetSpec: production parallel-tree runtime not ported.
- bf16 capture/training are protocol-reproducible, not bitwise deterministic
  (observed run-to-run AL noise < ~2%).
- 24 GB cards: suggested, unvalidated (all measurements are on 48 GB).
