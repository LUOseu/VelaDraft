# VelaDraft

<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

[![CI status](https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg)](https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml)

[English](README.md) · [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Model zoo](docs/model-zoo.md) · [Benchmarks](BENCHMARKS.md)

VelaDraft trains and evaluates parallel speculative drafters for vision-language models. Capture features from a frozen image-text target once, train compatible drafters on the same records, then compare them against the target's own greedy decoding. The Python package and CLI are called **vlm-dflash**.

## Quick start

The reference CUDA setup uses a single 48 GB GPU and a local Qwen2.5-VL-3B-Instruct target. From a Linux shell with Python 3.10 or later:

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[tested]"

export VLM_DFLASH_TARGET="$PWD/assets/models/Qwen2.5-VL-3B-Instruct"
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$VLM_DFLASH_TARGET"
vlm-dflash doctor --target "$VLM_DFLASH_TARGET"
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

The demo prepares images, regenerates answers, captures features, trains a DFlash drafter, evaluates it and writes an export under `runs/demo_dflash/`. To try trained weights immediately, use the [model zoo](docs/model-zoo.md). It also links all 14 community checkpoints trained by [@khazic](https://github.com/khazic). [Installation and first-run guide](docs/getting-started.md)

## Training and evaluation

| Task | Entry point |
| --- | --- |
| Train on your own images and conversations | [JSONL format](docs/data.md) · [training guide](docs/training.md) · [runnable example](examples/train_from_jsonl.py) |
| Reuse one feature store across methods | [Comparison example](examples/reuse_features_compare.py) · [online extraction and caching](docs/online-training.md) |
| Choose a drafter and training objective | [Method and recipe table](docs/README.md#choose-a-method--recipe) · [model and hardware support](SUPPORT_MATRIX.md) |
| Evaluate a checkpoint | [HF checkpoint diagnostics and native vLLM request timing](docs/benchmark.md) |
| Reproduce a seven-method MMSpec campaign | [Qwen3-VL-8B and Qwen3.6-35B-A3B runners](research/README.md) |
| Run on Ascend 910B3 | [NPU installation, validation and hardware results](docs/npu/README.md) |

The registry includes DFlash, DFlash2, DSpark, Domino, JetSpec, DFlare and DFly. The DFlash2 training objective is an experimental reconstruction. The JetSpec integration uses linear verification. Coverage varies by target, recipe and mode. [Support matrix](SUPPORT_MATRIX.md)

## Engineering results

- **Training cost.** The toolkit supports feature reuse, objective-aware row compaction and chunked vocabulary scoring. On a 512-row, three-epoch workload, pinned cache retention reduced paired training wall time by 3.6% and 6.9% at 0.8 and 1.6 GiB caps, respectively (two pairs per cap). [Training protocol](docs/online-training.md) · [scoring implementation](docs/training-scoring.md)
- **Native evaluation.** The paired vLLM evaluator records request-level AR and speculative wall time, output token IDs, and stop reasons. The target-specific runners pin their exports and runtime settings. [Timing protocol](docs/benchmark.md#native-vllm-e2e) · [recorded results](BENCHMARKS.md)
- **Hardware coverage.** All seven methods have Qwen2.5-VL-3B BF16 offline training, evaluation and export evidence on Ascend 910B3. DFlash also has online and split-device checks. [Hardware records](docs/npu/HARDWARE_RESULTS.md)

## Source and experiment layout

| Location | Contents |
| --- | --- |
| [vlm_dflash/](vlm_dflash/) | Installable package, target adapters, drafter implementations and evaluation |
| [configs/](configs/) · [examples/](examples/) | Training recipes and runnable examples |
| [benchmarks/](benchmarks/) · [tests/](tests/) | Workload manifests, reference records and regression tests |
| [research/](research/README.md) | MMSpec campaign entry points and the runtime modules they load |
| [Research archive](research/archive/README.md) | Earlier experiments, profiling scripts, raw evidence and reproduction notes |

## Contributing and citation

[Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) · [Release validation](RELEASE_VALIDATION.md) · [Upstream attribution](docs/upstream/UPSTREAM_AUDIT.md) · [Apache-2.0](LICENSE)

See [CITATION.cff](CITATION.cff) for citation metadata and [NOTICE](NOTICE) for component attribution.
