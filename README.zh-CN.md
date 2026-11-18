# VelaDraft

<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — 一次特征提取，训练多种草稿模型" width="960" />
</p>

[![CI 状态](https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg)](https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml)

[English](README.md) · [简体中文](README.zh-CN.md) · [文档](docs/README.md) · [模型下载](docs/model-zoo.md) · [基准结果](BENCHMARKS.md)

VelaDraft 用于训练和评测视觉语言模型的并行投机解码草稿模型。冻结的图文目标模型只需提取一次特征，兼容的 drafter 即可用同一份记录训练，再与目标模型自身的贪心解码进行对照。Python 包和命令行工具名为 **vlm-dflash**。

## 快速开始

CUDA 参考配置使用单张 48 GB GPU 和本地 Qwen2.5-VL-3B-Instruct 权重。在安装了 Python 3.10 或更高版本的 Linux 环境中运行下列命令。

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

示例会准备图片、重新生成答案、提取特征、训练并评测 DFlash，最后把导出产物写入 `runs/demo_dflash/`。想直接体验已训练的权重，可以打开[模型目录](docs/model-zoo.md)，其中也收录了 [@khazic](https://github.com/khazic) 训练并分享的全部 14 个社区 checkpoint。[安装与首次运行指南](docs/getting-started.md)

## 训练与评测

| 任务 | 入口 |
| --- | --- |
| 在自己的图文对话上训练 | [JSONL 格式](docs/data.md) · [训练指南](docs/training.md) · [可运行示例](examples/train_from_jsonl.py) |
| 用同一份特征训练不同方法 | [共享特征对照示例](examples/reuse_features_compare.py) · [在线提取与缓存](docs/online-training.md) |
| 选择 drafter 和训练目标 | [方法与配方表](docs/README.md#choose-a-method--recipe) · [模型及硬件支持](SUPPORT_MATRIX.md) |
| 评测 checkpoint | [HF 诊断与原生 vLLM 请求计时](docs/benchmark.md) |
| 复现七种方法的 MMSpec 评测 | [Qwen3-VL-8B 与 Qwen3.6-35B-A3B 入口](research/README.md) |
| 在昇腾 910B3 上运行 | [NPU 安装与硬件验证](docs/npu/README.md) |

已接入 DFlash、DFlash2、DSpark、Domino、JetSpec、DFlare 和 DFly。DFlash2 的训练目标属于实验性重建，JetSpec 使用线性验证路径。各 target、配方和运行模式的验证范围见[支持矩阵](SUPPORT_MATRIX.md)。

## 工程结果

- **训练成本。** 训练路径支持特征复用、按目标函数压缩预测行，以及分块词表评分。在 512 行、3 个 epoch 的对照中，缓存上限为 0.8 和 1.6 GiB 时，固定保留特征分别使训练墙钟时间缩短 3.6% 和 6.9%，每个上限各测两对。[训练协议](docs/online-training.md) · [评分实现](docs/training-scoring.md)
- **原生评测。** vLLM 评测器配对记录自回归与投机解码的请求耗时、输出 token ID 和停止原因。各 target 的运行脚本固定导出产物和运行时配置。[计时协议](docs/benchmark.md#native-vllm-e2e) · [实测记录](BENCHMARKS.md)
- **硬件支持。** 七种 drafter 均有昇腾 910B3 上 Qwen2.5-VL-3B 的 BF16 离线训练、评测和导出记录。DFlash 另有在线训练和双设备放置验证。[硬件记录](docs/npu/HARDWARE_RESULTS.md)

## 代码与实验目录

| 位置 | 内容 |
| --- | --- |
| [vlm_dflash/](vlm_dflash/) | 可安装的包、目标模型适配器、drafter 实现和评测代码 |
| [configs/](configs/) · [examples/](examples/) | 训练配方和可运行示例 |
| [benchmarks/](benchmarks/) · [tests/](tests/) | 工作负载定义、参考记录和回归测试 |
| [research/](research/README.md) | MMSpec 评测入口及其调用的运行时模块 |
| [历史研究归档](research/archive/README.md) | 早期实验、性能诊断、原始记录和复现说明 |

## 贡献与引用

[贡献指南](CONTRIBUTING.md) · [路线图](ROADMAP.md) · [更新记录](CHANGELOG.md) · [发布验证](RELEASE_VALIDATION.md) · [上游归属](docs/upstream/UPSTREAM_AUDIT.md) · [Apache-2.0](LICENSE)

引用信息见 [CITATION.cff](CITATION.cff)，移植组件的归属见 [NOTICE](NOTICE)。
