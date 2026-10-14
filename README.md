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
  <a href="#-quick-start">Quick Start</a> ·
  <a href="https://github.com/LUOseu/VelaDraft/releases">Model Zoo</a> ·
  <a href="BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/">Documentation</a> ·
  <a href="SUPPORT_MATRIX.md">Support Matrix</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Qwen--VL-1f6feb?style=flat" alt="Qwen-VL" />
  <img src="https://img.shields.io/badge/DFlash-30363d?style=flat" alt="DFlash" />
  <img src="https://img.shields.io/badge/DSpark-30363d?style=flat" alt="DSpark" />
  <img src="https://img.shields.io/badge/Domino-30363d?style=flat" alt="Domino" />
  <img src="https://img.shields.io/badge/JetSpec-30363d?style=flat" alt="JetSpec" />
  <img src="https://img.shields.io/badge/DFlash2-30363d?style=flat" alt="DFlash2" />
  <img src="https://img.shields.io/badge/DFlare-30363d?style=flat" alt="DFlare" />
  <img src="https://img.shields.io/badge/DFly-30363d?style=flat" alt="DFly" />
  <img src="https://img.shields.io/badge/Offline%20%2B%20Online-1f6feb?style=flat" alt="Offline + Online" />
</p>

---

## 📍 Overview

**VelaDraft** is a unified training toolkit for multimodal speculative drafters.

It keeps the VLM-specific plumbing shared while letting the drafter algorithm change.

- **One interface** — train, evaluate, compare, and export multiple drafter families through the same workflow.
- **VLM-first** — processors, image spans, position semantics, target features, and provenance are handled explicitly.
- **Offline or online** — reuse staged target features or extract frozen-target features live during training.
- **Correctness-first** — sequential reference evaluation verifies speculative output against target greedy decoding.

> `online` means live feature extraction from a frozen target during training; it is not on-policy rollout training.

## 🔧 Supported Methods

| Method | Core idea | Status |
|---|---|---|
| **DFlash** | Block-parallel drafting | Validated |
| **DSpark** | Markov correction + confidence | Validated |
| **Domino** | Causal GRU correction | Validated |
| **JetSpec** | Causal block attention + forward KL | Validated* |
| **DFlash2** | Local mixing + candidate selector | Experimental* |
| **DFlare** | Layer-wise target fusion | Validated |
| **DFly** | Fusion residual + hidden correction | Validated |

\* See [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) for method-specific caveats.

## 🚀 Quick Start

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft

python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"

export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct

vlm-dflash train-demo \
  --target "$VLM_DFLASH_TARGET" \
  --steps 80
```

For pretrained checkpoints, full training recipes, online mode, custom data, resume, evaluation, and export:

- 📦 [Model Zoo](https://github.com/LUOseu/VelaDraft/releases)
- 📊 [Benchmarks](BENCHMARKS.md)
- 📖 [Documentation](docs/)
- ✅ [Support Matrix](SUPPORT_MATRIX.md)

## 📃 Citation

```bibtex
@misc{veladraft2026,
  title = {VelaDraft: Multimodal Speculative Drafter Training},
  year  = {2026},
  url   = {https://github.com/LUOseu/VelaDraft}
}
```

---

<p align="center">
  <sub>VelaDraft · multimodal speculative drafter training</sub>
</p>
