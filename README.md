<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>Multimodal parallel drafter training for Qwen-VL.</strong><br />
  Bring image-text data, train a block drafter against a frozen target, and
  verify its output token-exactly &mdash; train, resume, evaluate, export.
</p>

<p align="center">
  <a href="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml"><img src="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-12263a" alt="Python 3.10+" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-12263a" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#model-zoo">Model Zoo</a> ·
  <a href="#train-on-your-data">Training</a> ·
  <a href="#benchmarks">Benchmarks &amp; history</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

VelaDraft is a unified training toolkit for multimodal block-parallel drafters.
It supports DFlash, DSpark, Domino, JetSpec, DFlash2, DFlare, and DFly against
frozen Qwen-VL targets, with offline feature stores or live target feature
extraction, resumable training, sequential output verification, and checkpoint
export.

DFlare and DFly are integrated on main. DFly currently uses the shared block-CE
training profile; the upstream two-stage training recipe remains a separate
planned capability.

- **One interface for every method.** All seven drafters share the same `train` /
  `eval` / `export` commands, checkpoint format and verification path, so method
  comparisons stay on one foundation.
- **Train offline or online.** By default the target is off the GPU during
  drafter training and the trainer reads the captured feature store;
  `--mode online` extracts frozen-target features live during training.
- **Explicit layer capture and compatible reuse.** A capture records the ordered
  target layers it holds; consumers slice by explicit layer id, so one store can
  feed several methods (and draft depths) whose consumed layers fit inside it,
  and a missing layer fails loud instead of being guessed.
- **Recover, validate, export.** Exact-resume gates compare the resolved training
  plan and the feature source before anything is written; sequential evaluation
  reproduces the target's greedy tokens (`token_exact`) with per-slot
  acceptance; exports are self-describing (weights, config, model card,
  checksums).

## Quickstart

Use an existing drafter first, then try the complete training pipeline.
The package and command are named **`vlm-dflash`**.

### Install

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft
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
  https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz

vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
```

The demo generates a small image sample, prints the answers, and reports
**acceptance length, per-slot acceptance and token agreement**. No training or
external dataset is needed. Download time depends on the network; measured
execution times are in the [validation record](RELEASE_VALIDATION.md).

Omit `--drafter` to run the random-init acceptance baseline.

### Try the whole training loop

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

This runs prepare → response generation → capture → tiny training → eval →
export on generated images. It is a pipeline smoke test; use your own data
and a full recipe for meaningful drafter quality.

## How it works

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="Image-text data goes through target generation and capture into a reusable feature store. An offline or online trainer selects one of seven algorithms (DFlash, DSpark, Domino, JetSpec, DFlash2, DFlare, DFly), then validates and exports its checkpoint." width="1100" />
</p>

The expensive target-side work is separated from drafter training. Compatible
features can feed multiple algorithms; **the target is reloaded for evaluation**.
Each checkpoint is paired with its target contract and algorithm metadata.
Training can also run against the live frozen target
(`train --mode online`); see [online training](docs/online-training.md).

## Supported methods

All seven methods use the same train, eval and export commands.

| Method | Mechanism | Training recipe in this repo |
| --- | --- | --- |
| [DFlash](docs/algorithms/dflash.md) | Shared FC context + block drafting | Reference block CE |
| [DSpark](docs/algorithms/dspark.md) | Markov logit bias + confidence | CE + ported auxiliary objective (see method doc) |
| [Domino](docs/algorithms/domino.md) | Causal GRU correction | Ported curriculum recipe |
| [JetSpec](docs/algorithms/jetspec.md) | Causal in-block mask + teacher distillation | Forward-KL |
| [DFlash2](docs/algorithms/dflash2.md) | Local convolutions + candidate selector | Experimental reconstructed objective |
| [DFlare](docs/algorithms/dflare.md) | Per-draft-layer fusion + dedicated context K/V | Shared block CE |
| [DFly](docs/algorithms/dfly.md) | FC + fusion residual + hidden correction | CE adaptation; upstream two-stage recipe pending |

DFlare and DFly are **integrated on main** and validated in-repo on both
Qwen-VL targets (full pipeline on Qwen2.5-VL-3B, reduced validation on
Qwen3-VL-8B); their release-validation pass is tracked separately in the
[support matrix](SUPPORT_MATRIX.md) and [release validation](RELEASE_VALIDATION.md).

DFly notes: training runs the shared block-CE profile (`objective_profile=controlled_ce`
in the draft config, checkpoint and export); upstream's LK/D-PACE → end-to-end TV
recipe is not ported. At inference the block backbone runs in parallel, while the
hidden-states correction and the LM-head projection run per slot sequentially,
conditioned on the previously proposed tokens — the documented local adaptation
of upstream's train-time-only correction.

DFlash2 uses an experimental reconstruction of the undisclosed upstream training
objective, and JetSpec's production tree runtime is not ported. Per-target status
and capability details are in the [support matrix](SUPPORT_MATRIX.md), the
[algorithm matrix](ALGORITHM_MATRIX.md) and the [upstream audits](UPSTREAM_AUDIT.md).

## Model Zoo

Released checkpoints for **Qwen2.5-VL-3B-Instruct**:

| Drafter | Download | Model card |
| --- | --- | --- |
| **DFlash** · frozen block reference | [387 MB](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz) | [Details](docs/zoo/qwen25vl-3b-dflash.md) |
| **Domino** · GRU correction | [464 MB](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-domino.tar.gz) | [Details](docs/zoo/qwen25vl-3b-domino.md) |

These archives are **reference artifacts for pipeline validation**: they were
trained under the recorded development protocol and are meant for exercising
`train → eval → export`, integration work and output-agreement checks. They
are not tuned or benchmarked as production drafters, and their acceptance
numbers are development measurements — recorded, with the training protocol
and all other runs, in [BENCHMARKS.md](BENCHMARKS.md). Both exports were
re-verified after download with the token-exact gate; see the
[release evidence](RELEASE_VALIDATION.md).

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

### Method-specific configs

The shipped configs are method entry points (also see `configs/dspark`,
`configs/domino`, `configs/jetspec`, `configs/dflash2`); point the target and
dataset paths at your environment first — they are recipes, not runnable
without data preparation.

```bash
# Offline recipes
vlm-dflash train --config configs/dflare/qwen2_5_vl_3b_2k.yaml --run-dir runs/dflare
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly

# Same method, live target features
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --mode online --run-dir runs/dfly_online

vlm-dflash eval runs/dfly
vlm-dflash export runs/dfly
```

### Reuse a capture across runs and methods

```bash
vlm-dflash train \
  --target "$VLM_DFLASH_TARGET" --dataset data/rows.jsonl \
  --algorithm dspark --run-dir runs/dspark_2k \
  --features runs/shared_2k/features --steps 792
```

Reuse is checked, not assumed: the store's target/processor/mask/MRoPE identity
and its ordered captured-layer list must match the run, and each method consumes
the layers it needs by explicit id (`algorithm.<method>.context_layer_ids`, or
the draft routing). A store that lacks a consumed layer is refused with a
recapture hint — never silently re-sliced. Declare the store's layer list in the
run as `drafter.capture_layer_ids`; runs whose draft depth differs from the
capture can reuse it as long as their consumed layers fit. Reuse is recorded in
`provenance.json`. A canonical 2K capture is about **9.4 GiB** on disk.

The [comparison example](examples/reuse_features_compare.py) demonstrates
one capture followed by DFlash and DSpark training and evaluation.

### Resume an interrupted run

```bash
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly --resume
```

`--resume` runs a read-only preflight before anything is written: the feature
source digest and the resolved training plan (seed, batch geometry, optimizer
and schedule, dtype, drafter and method knobs, dataset length) must match the
checkpoint exactly. A refused resume leaves the run's config, checkpoint, log
and provenance byte-identical; a legal one continues from the recorded step and
replays the optimizer/RNG state.

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

Performance measurements — acceptance length, per-slot acceptance, timing,
token-exactness and ablations — live in the development record:
[BENCHMARKS.md](BENCHMARKS.md), [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md)
and [docs/measured-runs.md](docs/measured-runs.md). The README intentionally
does not surface small-scale training numbers or an algorithm ranking: the
recorded runs are plumbing/protocol evidence under one shared recipe, not a
claim about method quality or wall-clock speedup.

What the project does verify at every scale is **correctness**: sequential
speculative decoding must reproduce the target's own greedy tokens exactly
(`token_exact`), together with per-slot acceptance accounting. See
[verification modes](docs/verification-modes.md).

## Scope

- **Inputs:** one image + text, one turn, greedy decoding.
- **Targets:** the two Qwen-VL targets listed above; other families need an
  adapter and validation.
- **Hardware:** the measured reference is one 48 GB GPU. The 24 GB preset is
  suggested and unvalidated on an actual 24 GB card.
- **Serving:** production vLLM/SGLang integration is future work; the
  [serving bridge](docs/serving-bridge.md) describes the proposed handoff.
- **Reproducibility:** data and generation protocols are reproducible; bf16
  training is not bitwise deterministic. Acceptance numbers stay in the
  measurement records (see [Benchmarks](#benchmarks)).

## Documentation

| Start here | Links |
| --- | --- |
| Use the project | [Environment](docs/env-preflight.md) · [Examples](examples/README.md) · [Online training](docs/online-training.md) · [Measured runs](docs/measured-runs.md) |
| Understand the methods | [Algorithms](docs/algorithms/) · [Support matrix](SUPPORT_MATRIX.md) · [Reference](docs/REFERENCE.md) · [Upstream audits](UPSTREAM_AUDIT.md) |
| Reproduce the evidence | [Benchmarks](BENCHMARKS.md) · [Harness](docs/benchmark.md) · [Release validation](RELEASE_VALIDATION.md) |
| Extend the stack | [Add an algorithm](docs/adding_an_algorithm.md) · [Contributing](CONTRIBUTING.md) |

## Contributing and attribution

Reproduction reports, focused fixes and clear examples help the project.
Read [CONTRIBUTING.md](CONTRIBUTING.md) for the current release policy and
validation requirements. Algorithms plug into `DrafterAlgorithm`; target
families are isolated behind `VLMAdapter`.

Apache-2.0. Adapted components retain their upstream attribution in
[NOTICE](NOTICE) (including the AngelSpec-based DFlare/DFly ports, Apache-2.0
with its third-party MIT components), with source commits and implementation
scope documented in [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). Citation metadata
is in [CITATION.cff](CITATION.cff).

<sub><strong>Why Vela?</strong> Latin for sails, and the name of the constellation. A lightweight drafter moves forward on the visual context provided by its target.</sub>
