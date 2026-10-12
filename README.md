# vlm-dflash

**A unified training and evaluation stack for multimodal parallel drafters.**

Train, validate and compare multimodal speculative drafters without rebuilding
the VLM plumbing: Qwen-VL processor handling, MRoPE position ids, image-token
spans, staged target hidden-state capture, provenance gates, token-exact
evaluation and self-describing export are implemented once and shared by every
algorithm. Five drafter algorithms plug into that stack; every support claim in
this README is backed by a recorded run (`BENCHMARKS.md`, `docs/`).

## Supported algorithms

| Algorithm | Qwen2.5-VL | Qwen3-VL | Train | Eval | Export | Status |
| --- | --- | --- | --- | --- | --- | --- |
| DFlash | validated | validated | yes | yes | yes | **Stable** (frozen reference) |
| DSpark | validated | reduced validation | yes | yes | yes | **Validated** |
| Domino | validated | reduced validation | yes | yes | yes | **Validated** |
| JetSpec | validated | reduced validation | yes | yes | yes | **Validated** (tree runtime not ported) |
| DFlash2 | experimental | reduced validation | yes | yes | yes | **Experimental** (upstream objective undisclosed) |

```text
              DFlash  (frozen baseline: decayed block CE)
              /  |   \
        DSpark Domino DFlash2
              \  |
             JetSpec
```

All five share the parallel block-drafter backbone. DSpark adds a Markov logit
bias + confidence head; Domino adds a causal GRU correction; JetSpec swaps the
in-block mask to causal and CE for forward-KL distillation; DFlash2 adds local
dynamic convolutions plus a predecessor-conditioned candidate selector.

> **DFlash2 is Experimental.** The architecture is ported from the upstream
> implementation, but the upstream *training objective is not publicly
> disclosed*; this repo implements the maintainer implementation's experimental
> split objective. It is an experimental training reconstruction - not an
> official DFlash2 training reproduction. See `docs/upstream/dflash2.md`.

Switching algorithms never re-opens the multimodal plumbing: checkpoints record
their algorithm, and `train` / `eval` / `export` fail loudly on mismatches.
`vlm-dflash doctor --algorithm <name>` reports registry status and metadata.

## Supported targets

| Target | Status | Notes |
| --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | validated | full zoo, canonical 2K apples-to-apples benchmark |
| Qwen3-VL-8B-Instruct | validated | DFlash full pipeline; remaining algorithms reduced-scale validation |

Single image + text samples, greedy decoding. Everything family-specific
(mask token, MRoPE sections, draft-layer routing, pixel bounds, tied vs untied
`lm_head`) is resolved from the target at runtime into a `contract.json` and
validated against the config defaults.

## Why this exists

Existing speculative-decoding stacks target text-only drafters. The multimodal
gaps that bite you are Qwen-VL's MRoPE position ids, image-token spans, the
processor's pixel bounds, and the target's reserved mask embedding row. This
repo pins the whole loop around those details:

- staged pipeline - `prepare -> target-native regen -> hidden-state capture ->
  drafter training -> eval -> export`; the target model is off the GPU during
  training (the trainer reads captured features only);
- a runtime-resolved model contract and per-stage provenance - target identity
  and processor hashes are recorded, teacher-target mismatches fail loudly;
- a token-exact correctness gate - speculative greedy output must equal the
  target's own greedy output token-for-token (sequential verification);
- one algorithm registry so DFlash / DSpark / Domino / JetSpec / DFlash2 train,
  evaluate, and export through the same commands and the same data.

## Quickstart

```bash
# 0) install (from source until the release lands on PyPI)
python -m venv .venv && . .venv/bin/activate
pip install -e .

# 1) point the configs at your local target model + datasets
export VLM_DFLASH_TARGET=assets/models/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_LLAVA_JSON=data/llava_instruct_150k.json
export VLM_DFLASH_COCO_IMAGES=data/coco/train2017

# 2) check the environment + model contract
vlm-dflash doctor --target "$VLM_DFLASH_TARGET"

# 3) one-shot smoke: the whole pipeline, a few GPU-minutes (COCO captions)
vlm-dflash train --preset smoke --limit 64

# 4) evaluate the trained drafter (token-exact gate + AL + per-slot acceptance)
vlm-dflash eval runs/smoke_coco
vlm-dflash export runs/smoke_coco   # self-describing checkpoint + model card
```

The full long-answer recipe (512-row LLaVA subset, ~1 GPU-hour on an A6000):

```bash
bash scripts/quickstart.sh --config configs/qwen2_5_vl_3b.yaml --limit 512
```

Every stage is also a standalone script (`scripts/prepare_data.py`,
`generate_target_data.py`, `capture_features.py`, `train_dflash.py`,
`eval_dflash.py`) with `--config`, `--limit`, `--set k=v`; `train` additionally
supports `--resume` (model + optimizer + scheduler + RNG state restore) and
`--algorithm <name>` to switch drafters behind the same pipeline.

## Five-minute demo

A local target checkout plus one CUDA GPU is the only requirement - there is no
dataset to download. `demo` generates a tiny license-free sample, runs the
token-exact evaluation and prints AL / token exactness / per-slot acceptance:

```bash
vlm-dflash demo --target "$VLM_DFLASH_TARGET"
# score real weights instead of the random-init acceptance baseline:
vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter assets/zoo/qwen25vl-3b-dflash
```

`train-demo` runs the complete staged loop (prepare -> target-native regen ->
capture -> tiny training -> eval -> export) on the same sample:

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

It finishes with the trained checkpoint scored against the random-init
baseline on held-out prompts. Measured wall-clock/VRAM for both commands on
the reference card are recorded in `RELEASE_VALIDATION.md`.

## Model Zoo

Zoo exports are produced by the release pipeline
(`python scripts/export_zoo.py --out assets/zoo`; see `RELEASE_VALIDATION.md`)
with one self-describing directory per checkpoint - each contains
`model.safetensors`, `config.json`, `draft_config.json`, `algorithm.json`,
`target_contract.json`, `training_manifest.json` and a generated
`model_card.md` (target, algorithm + status, corpus, steps, GPU/VRAM, AL,
per-slot acceptance, token exactness, upstream references, license, known
limitations):

```text
qwen25vl-3b-dflash   qwen25vl-3b-dspark   qwen25vl-3b-domino
qwen25vl-3b-jetspec  qwen25vl-3b-dflash2-exp   qwen3vl-8b-dflash
```

## Benchmark scoreboard (canonical)

Qwen2.5-VL-3B, 2K-row corpus, block size 8, 128 eval prompts / 128 tokens,
greedy, **sequential verification - token-exact for every row** (the same
prompts, corpus, features and seed protocol for all five algorithms):

| Algorithm | AL | slot 1 | slot 3 | slot 6 | Token-exact |
| --- | ---: | ---: | ---: | ---: | --- |
| DFlash (frozen baseline) | 1.630 | .413 | .048 | .001 | 128/128 |
| DSpark | 1.735 | .432 | .075 | .007 | 128/128 |
| Domino | 1.764 | .443 | .078 | .009 | 128/128 |
| JetSpec (causal-mask serving) | 1.599 | .408 | .040 | .001 | 128/128 |
| DFlash2 (experimental objective) | 1.954 | .495 | .118 | .013 | 128/128 |

Random-init drafter AL is ~1.003 across runs (the acceptance baseline);
target-only greedy has no acceptance length and is only an output/speed
reference. Full tables, ablations, the runtime probe and the public-workload
benchmark live in `BENCHMARKS.md`; verification modes are documented in
`docs/verification-modes.md` - `sequential` is the only bitwise-exact path,
`parallel` is an experimental throughput path.

## Installation

```bash
pip install vlm-dflash            # from PyPI after the v0.3.0 release
pip install -e ".[tested]"        # exact validated dependency combo (from source)
```

Tested combination (recorded in `constraints-tested.txt` and
`RELEASE_VALIDATION.md`): Python >= 3.10, `torch==2.10.0`,
`transformers==5.12.1`, `datasets==4.2.0`. Everything was measured on a single
48 GB card (A6000); 24 GB setups are a **suggested, unvalidated** target -
lower `train.micro_batch_size` / `capture.seq_length` if you are
memory-constrained.

## Documentation

| Document | What it covers |
| --- | --- |
| `docs/REFERENCE.md` | ported training semantics, AL definition, head/MRoPE rules |
| `docs/algorithms/<name>.md` | per-algorithm design, config, train/eval commands |
| `docs/upstream/<name>.md` | upstream audit: sources, commits, licenses, parity level |
| `docs/benchmark.md` | the benchmark harness + workload manifests |
| `docs/verification-modes.md` | sequential vs parallel verification, runtime probe |
| `docs/measured-runs.md` | end-to-end measured runs and scaling tables |
| `docs/adding_an_algorithm.md` | what a new drafter plugin must implement |
| `BENCHMARKS.md` / `STATE.md` | every measured number / current project state |
| `RELEASE_VALIDATION.md` | the release validation record for this version |

## Limitations

- Single image + text only; no video, no multi-image, no multi-turn.
- Two target families (Qwen2.5-VL-3B, Qwen3-VL-8B); other families need a
  small adapter (`vlm_dflash/adapters/`).
- No production serving integration - this repo trains, evaluates and exports
  drafters; `sequential` verification is correctness-first, not a speed path.
- Parallel / strict-parallel verification is **not bitwise exact** in bf16
  (see `docs/verification-modes.md`); always compare exactness first.
- DFlash2 trains with an experimental objective reconstruction (upstream
  objective undisclosed); JetSpec's production parallel-tree runtime is not
  ported.
- bf16 capture and training are protocol-reproducible, not bit-identical
  (observed run-to-run AL noise < ~2%).

## License

Apache-2.0. Ported/adapted components carry their upstream attribution in
`NOTICE`; each algorithm's parity level and license are recorded in
`UPSTREAM_AUDIT.md` and `docs/upstream/`.
