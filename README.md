<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>Multimodal parallel drafter training for Qwen-VL.</strong><br />
  Train, compare and export five drafting algorithms through one shared pipeline.
</p>

<p align="center">
  <a href="https://github.com/LUOseu/vlm-dflash/actions/workflows/ci.yml"><img src="https://github.com/LUOseu/vlm-dflash/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/LUOseu/vlm-dflash/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-12263a" alt="Python 3.10+" /></a>
  <a href="https://github.com/LUOseu/vlm-dflash/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-12263a" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#model-zoo">Model Zoo</a> ·
  <a href="#train-on-your-data">Training</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="https://github.com/LUOseu/vlm-dflash/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

VelaDraft brings **DFlash, DSpark, Domino, JetSpec and DFlash2** into a single
multimodal training stack. Generate responses with your target VLM, capture
its features, and reuse compatible captures to train and compare different
drafters.

- **Capture once, reuse across methods.** Shared hidden states, image spans and
  position information keep algorithm comparisons on the same foundation.
- **Train offline or online.** By default the target is off the GPU during
  drafter training and the trainer reads the captured feature store;
  `--mode online` extracts frozen-target features live during training.
- **Keep the pairing explicit.** Runtime contracts and provenance check the
  target, processor, feature store and checkpoint before they are combined.
- **Validate the output.** Sequential evaluation compares speculative greedy
  output with the target's own greedy tokens and reports AL and per-slot
  acceptance.

## Quickstart

Use an existing drafter first, then try the complete training pipeline.
The package and command are named **`vlm-dflash`**.

### Install

```bash
git clone https://github.com/LUOseu/vlm-dflash.git
cd vlm-dflash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"
```

The current supported installation is from source. The tested environment uses
Python ≥3.10, PyTorch 2.10.0 and Transformers 5.12.1 on one RTX A6000 48 GB.
See the [environment guide](docs/env-preflight.md) and
[validation record](RELEASE_VALIDATION.md) for details.

### Run a pretrained drafter

Point to a **local Qwen2.5-VL-3B-Instruct checkout**, then download the released
DFlash drafter:

```bash
export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct

curl -fL -o qwen25vl-3b-dflash.tar.gz \
  https://github.com/LUOseu/vlm-dflash/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz

vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
```

The demo generates a small image sample, prints the answers, and reports
**acceptance length, per-slot acceptance and token agreement**. No training or
external dataset is needed. Download time depends on the network; measured
execution times are in the [validation record](RELEASE_VALIDATION.md).

Omit `--drafter` to run the random-init acceptance baseline (AL ≈1.00).

### Try the whole training loop

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

This runs prepare → response generation → capture → tiny training → eval →
export on generated images. It is a pipeline smoke test; use your own data
and a full recipe for meaningful drafter quality.

## How it works

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="Image-text data goes through target generation and capture into a reusable feature store. An offline trainer selects one of five algorithms, then validates and exports its checkpoint." width="1100" />
</p>

The expensive target-side work is separated from drafter training. Compatible
features can feed multiple algorithms; **the target is reloaded for evaluation**.
Each checkpoint is paired with its target contract and algorithm metadata.
Training can also run against the live frozen target
(`train --mode online`); see [online training](docs/online-training.md).

## Supported algorithms

All five methods use the same train, eval and export commands.

| Algorithm | Qwen2.5-VL-3B | Qwen3-VL-8B | Status | Method |
| --- | --- | --- | --- | --- |
| [DFlash](docs/algorithms/dflash.md) | Validated | Validated | **Stable** | Frozen block-drafter reference |
| [DSpark](docs/algorithms/dspark.md) | Validated | Reduced validation | **Validated** | Markov logit bias + confidence |
| [Domino](docs/algorithms/domino.md) | Validated | Reduced validation | **Validated** | Causal GRU correction |
| [JetSpec](docs/algorithms/jetspec.md) | Validated | Reduced validation | **Validated** | Causal in-block mask + forward-KL |
| [DFlash2](docs/algorithms/dflash2.md) | Validated | Reduced validation | **Experimental** | Local convolutions + candidate selector |

Target names refer to **Qwen2.5-VL-3B-Instruct** and **Qwen3-VL-8B-Instruct**.
Reduced validation covers a 256-row slice and 16-prompt evaluation. Status
reflects the recorded training/evaluation coverage; runtime support is listed
separately in the [support matrix](SUPPORT_MATRIX.md).

DFlash2 uses an experimental reconstruction of the undisclosed upstream
training objective. JetSpec's production tree runtime is not ported.
See the [algorithm matrix](ALGORITHM_MATRIX.md) and
[upstream audits](UPSTREAM_AUDIT.md) for the exact implementation scope.

## Model Zoo

Released checkpoints for **Qwen2.5-VL-3B-Instruct**:

| Drafter | AL¹ | Download | Model card |
| --- | ---: | --- | --- |
| **DFlash** · frozen reference | 1.630 | [387 MB](https://github.com/LUOseu/vlm-dflash/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz) | [Details](docs/zoo/qwen25vl-3b-dflash.md) |
| **Domino** · GRU correction | 1.764 | [464 MB](https://github.com/LUOseu/vlm-dflash/releases/download/zoo-v0.3.0/qwen25vl-3b-domino.tar.gz) | [Details](docs/zoo/qwen25vl-3b-domino.md) |

¹ Canonical 2K protocol below. AL is an acceptance metric, not a wall-clock
speedup. Both released exports were re-verified after download with the
token-exact gate; see the [release evidence](RELEASE_VALIDATION.md).

Each archive includes weights, configuration, algorithm metadata, target
contract, training manifest, model card and file checksums. Other algorithms
can be trained and exported locally; their canonical checkpoints are not
currently hosted.

## Train on your data

Use a JSONL pool of single-image question/answer pairs. Flat records,
OpenAI-style `messages`, and ShareGPT/LLaVA `conversations` are supported.

```json
{"id":"sample-001","image":"000001.jpg","question":"Describe this image in detail.","answer":"A red bicycle is parked beside a brick wall."}
```

A small end-to-end example:

```bash
python examples/train_from_jsonl.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images \
  --limit 32 --steps 60
```

For the full reference recipe, configure the dataset paths and run
`vlm-dflash train --config configs/qwen2_5_vl_3b.yaml`. The CLI also supports
`--algorithm`, `--resume`, and dotted `--set key=value` overrides.
See the [examples](examples/README.md), [data schema](vlm_dflash/data/schema.py)
and [reference guide](docs/REFERENCE.md).

### Reuse a capture across algorithms

```bash
vlm-dflash train \
  --target "$VLM_DFLASH_TARGET" --dataset data/rows.jsonl \
  --algorithm dspark --run-dir runs/dspark_2k \
  --features runs/shared_2k/features --steps 792
```

Use the dataset and compatible target settings associated with the captured
store. Its identity, manifest and saved target head are verified before
training; reuse is recorded in `provenance.json`. A canonical 2K capture is
about **9.4 GiB** on disk.

The [comparison example](examples/reuse_features_compare.py) demonstrates
one capture followed by DFlash and DSpark training and evaluation.

### Train online (live target features)

Skip the feature store and let the frozen target run alongside the trainer:

```bash
vlm-dflash train \
  --target "$VLM_DFLASH_TARGET" --dataset data/rows.jsonl \
  --mode online --run-dir runs/online
```

The target stays resident for the whole run and re-extracts features every
epoch (no multi-GiB store on disk; the target forward cost is paid per epoch
instead of once). `--features` reuse is offline-only. See the
[online training guide](docs/online-training.md).

## Benchmarks

**Qwen2.5-VL-3B-Instruct · 2K training corpus · block size 8 · 128 prompts ·
128 new tokens · greedy · sequential verification · one RTX A6000.**

| Algorithm | AL | Token-exact | Drafter params | Training time² |
| --- | ---: | --- | ---: | ---: |
| DFlash | 1.630 | 128/128 | 243.8M | 9.8 min |
| DSpark | 1.735 | 128/128 | 321.6M | 11.8 min |
| Domino | 1.764 | 128/128 | 292.9M | 12.9 min |
| JetSpec | 1.599 | 128/128 | 243.8M | 20.6 min |
| DFlash2³ | 1.954 | 128/128 | 328.5M | 22.7 min |

² Drafter training after feature capture; excludes data preparation, response
generation and capture. ³ Experimental objective reconstruction.

These are comparisons under the shared protocol, not a general ranking of
upstream methods. Frozen results: [JSON](benchmarks/results/zoo_canonical.json)
/ [CSV](benchmarks/results/zoo_canonical.csv). Per-slot acceptance, VRAM,
ablations and further runs are in [BENCHMARKS.md](BENCHMARKS.md).

On the pinned **TextVQA** workload, DFlash / DSpark / Domino reach AL
1.263 / 1.253 / 1.268, with 128/128 token-exact prompts. This measures
speculative acceptance and agreement; standard-answer VQA accuracy is a
separate metric. See the [benchmark guide](docs/benchmark.md).

**Runtime:** current sequential reference runs measure 0.81–0.87× versus
target-only on the reference card. Parallel verification is experimental and
can diverge in bf16. Full timing and agreement results are recorded together
in [verification modes](docs/verification-modes.md).

## Scope

- **Inputs:** one image + text, one turn, greedy decoding.
- **Targets:** the two Qwen-VL targets listed above; other families need an
  adapter and validation.
- **Hardware:** the measured reference is one 48 GB GPU. The 24 GB preset is
  suggested and unvalidated on an actual 24 GB card.
- **Serving:** production vLLM/SGLang integration is future work; the
  [serving bridge](docs/serving-bridge.md) describes the proposed handoff.
- **Reproducibility:** data and generation protocols are reproducible; bf16
  training is not bitwise deterministic. Observed sub-2% AL differences may
  fall within run-to-run noise.

## Documentation

| Start here | Links |
| --- | --- |
| Use the project | [Environment](docs/env-preflight.md) · [Examples](examples/README.md) · [Online training](docs/online-training.md) · [Measured runs](docs/measured-runs.md) |
| Understand the methods | [Algorithms](docs/algorithms/) · [Reference](docs/REFERENCE.md) · [Upstream audits](UPSTREAM_AUDIT.md) |
| Reproduce the evidence | [Benchmarks](BENCHMARKS.md) · [Harness](docs/benchmark.md) · [Release validation](RELEASE_VALIDATION.md) |
| Extend the stack | [Add an algorithm](docs/adding_an_algorithm.md) · [Support matrix](SUPPORT_MATRIX.md) · [Contributing](CONTRIBUTING.md) |

## Contributing and attribution

Reproduction reports, focused fixes and clear examples help the project.
Read [CONTRIBUTING.md](CONTRIBUTING.md) for the current release policy and
validation requirements. Algorithms plug into `DrafterAlgorithm`; target
families are isolated behind `VLMAdapter`.

Apache-2.0. Adapted components retain their upstream attribution in
[NOTICE](NOTICE), with source commits and implementation scope documented in
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). Citation metadata is in
[CITATION.cff](CITATION.cff).

<sub><strong>Why Vela?</strong> Latin for sails, and the name of the constellation. A lightweight drafter moves forward on the visual context provided by its target.</sub>
