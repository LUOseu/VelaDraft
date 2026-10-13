<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>面向 Qwen-VL 的多模态并行草稿模型训练工具。</strong><br />
  通过同一条流水线，训练、比较和导出五种 drafter 算法。
</p>

<p align="center">
  <a href="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml"><img src="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-12263a" alt="Python 3.10+" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-12263a" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#模型下载">模型下载</a> ·
  <a href="#训练自己的数据">训练</a> ·
  <a href="#基准结果">基准结果</a> ·
  <a href="#文档">文档</a> ·
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/CONTRIBUTING.md">参与贡献</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

VelaDraft 将 **DFlash、DSpark、Domino、JetSpec 和 DFlash2** 接入同一套多模态训练流程。
先让目标 VLM 生成回答并捕获特征，再复用兼容的特征，训练与比较不同 drafter。

- **一次 capture，多种算法复用。** 共享 hidden states、图像区间和位置编码信息。
- **离线 / 在线训练。** 默认离线：训练阶段目标模型不驻留 GPU，直接读取捕获的
  feature store；`--mode online` 在训练过程中从冻结的目标模型实时提取特征。
- **配对可检查。** 通过运行时 contract 和 provenance 校验目标、processor、features 与 checkpoint。
- **输出可验证。** Sequential 评测逐 token 对照目标模型的贪心输出，并报告 AL 与各位置接受率。

## 快速开始

先体验已训练权重，再运行完整训练流程。安装包与命令名仍为 **`vlm-dflash`**。

### 安装

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"
```

当前支持从源码安装。实测环境为 Python ≥3.10、PyTorch 2.10.0、Transformers 5.12.1，
使用单张 RTX A6000 48 GB。详细环境见[环境指南](docs/env-preflight.md)与[验证记录](RELEASE_VALIDATION.md)。

### 运行已训练的 drafter

准备本地 **Qwen2.5-VL-3B-Instruct** 模型目录，下载已发布的 DFlash 权重：

```bash
export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct

curl -fL -o qwen25vl-3b-dflash.tar.gz \
  https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz

vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
```

Demo 会生成少量示例图片、打印回答，并报告 **接受长度、各位置接受率和 token 一致性**。
无需训练或下载外部数据集；目标模型需要提前准备。下载时间取决于网络，实测执行时间见[验证记录](RELEASE_VALIDATION.md)。

省略 `--drafter` 可运行随机初始化的接受率基线（AL ≈1.00）。

### 体验完整训练流程

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

在生成的图片上运行 prepare → 目标回答生成 → capture → 小规模训练 → eval → export。
这是完整流程的 smoke test；训练有实用价值的 drafter，需要真实数据与完整配方。

## 流程

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="图文数据通过目标模型生成回答与捕获特征。共享特征用于离线训练五种算法之一，再校验输出并导出 checkpoint。" width="1100" />
</p>

将昂贵的目标模型计算与 drafter 训练分开：兼容的捕获特征可供多种算法复用，
**评测时重新加载目标模型**。每个 checkpoint 都记录对应的 target contract 与算法信息。
也可以直接针对在线的冻结目标模型训练（`train --mode online`），见
[在线训练](docs/online-training.md)。

## 支持的算法

五种算法使用相同的 train、eval 和 export 命令。

| 算法 | Qwen2.5-VL-3B | Qwen3-VL-8B | 状态 | 方法 |
| --- | --- | --- | --- | --- |
| [DFlash](docs/algorithms/dflash.md) | 已验证 | 已验证 | **Stable** | 冻结的 block drafter 基线 |
| [DSpark](docs/algorithms/dspark.md) | 已验证 | 小规模验证 | **Validated** | Markov logit bias + confidence |
| [Domino](docs/algorithms/domino.md) | 已验证 | 小规模验证 | **Validated** | 因果 GRU 修正 |
| [JetSpec](docs/algorithms/jetspec.md) | 已验证 | 小规模验证 | **Validated** | 因果块内 mask + forward-KL |
| [DFlash2](docs/algorithms/dflash2.md) | 已验证 | 小规模验证 | **Experimental** | 局部卷积 + candidate selector |

Target 分别为 **Qwen2.5-VL-3B-Instruct** 和 **Qwen3-VL-8B-Instruct**。
小规模验证包含 256 行数据与 16 条评测 prompt。以上状态描述训练与评测覆盖，
运行时支持单独记录在[支持矩阵](SUPPORT_MATRIX.md)中。

DFlash2 的上游训练目标未公开，本项目使用实验性重建；JetSpec 的生产 tree runtime 未移植。
具体实现范围见[算法矩阵](ALGORITHM_MATRIX.md)与[上游审计](UPSTREAM_AUDIT.md)。

## 模型下载

已发布的 **Qwen2.5-VL-3B-Instruct** drafter：

| Drafter | AL¹ | 下载 | Model card |
| --- | ---: | --- | --- |
| **DFlash** · 冻结基线 | 1.630 | [387 MB](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz) | [详情](docs/zoo/qwen25vl-3b-dflash.md) |
| **Domino** · GRU 修正 | 1.764 | [464 MB](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-domino.tar.gz) | [详情](docs/zoo/qwen25vl-3b-domino.md) |

¹ 下方 canonical 2K 协议。AL 是接受指标，不是墙钟加速倍数。
两个 export 均在下载后重新通过 token-exact gate；见[发布证据](RELEASE_VALIDATION.md)。

每个压缩包包括权重、配置、算法 metadata、target contract、training manifest、model card 与文件校验值。
其他算法可在本地训练并导出，当前未托管其 canonical checkpoints。

## 训练自己的数据

使用每行包含一张图片和一对问答的 JSONL 数据池。
支持扁平字段、OpenAI 风格 `messages` 与 ShareGPT/LLaVA 风格 `conversations`。

```json
{"id":"sample-001","image":"000001.jpg","question":"Describe this image in detail.","answer":"A red bicycle is parked beside a brick wall."}
```

小规模端到端示例：

```bash
python examples/train_from_jsonl.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images \
  --limit 32 --steps 60
```

完整参考配方：配置数据路径后运行 `vlm-dflash train --config configs/qwen2_5_vl_3b.yaml`。
CLI 同时支持 `--algorithm`、`--resume` 与 `--set key=value` 覆盖。
详见[示例](examples/README.md)、[数据 schema](vlm_dflash/data/schema.py)与[参考指南](docs/REFERENCE.md)。

### 跨算法复用 capture

```bash
vlm-dflash train \
  --target "$VLM_DFLASH_TARGET" --dataset data/rows.jsonl \
  --algorithm dspark --run-dir runs/dspark_2k \
  --features runs/shared_2k/features --steps 792
```

使用与捕获特征对应的数据和兼容 target 设置。
训练前校验 store 的 identity、manifest 和 target head，复用记录写入 `provenance.json`。
Canonical 2K capture 占用约 **9.4 GiB** 磁盘。

[对比示例](examples/reuse_features_compare.py)展示一次 capture 后分别训练、评测 DFlash 和 DSpark。

### 在线训练（实时提取目标特征）

不落 feature store，让冻结的目标模型与 drafter 一同训练：

```bash
vlm-dflash train \
  --target "$VLM_DFLASH_TARGET" --dataset data/rows.jsonl \
  --mode online --run-dir runs/online
```

目标模型全程驻留显存，每个 epoch 重新提取一次特征（不占磁盘上的多 GiB
feature store，代价是目标前向按 epoch 重复支付）；`--features` 复用仅限
offline。详见[在线训练指南](docs/online-training.md)。

## 基准结果

**Qwen2.5-VL-3B-Instruct · 2K 训练数据 · block size 8 · 128 prompts ·
128 new tokens · 贪心解码 · sequential verification · 单张 RTX A6000。**

| 算法 | AL | Token-exact | Drafter 参数量 | 训练时间² |
| --- | ---: | --- | ---: | ---: |
| DFlash | 1.630 | 128/128 | 243.8M | 9.8 min |
| DSpark | 1.735 | 128/128 | 321.6M | 11.8 min |
| Domino | 1.764 | 128/128 | 292.9M | 12.9 min |
| JetSpec | 1.599 | 128/128 | 243.8M | 20.6 min |
| DFlash2³ | 1.954 | 128/128 | 328.5M | 22.7 min |

² Capture 完成后的 drafter 训练时间，不含数据准备、目标回答生成和 capture。
³ 实验性训练目标重建。

这些是同一协议下的对比，不代表上游方法的通用排名。
冻结结果：[JSON](benchmarks/results/zoo_canonical.json) / [CSV](benchmarks/results/zoo_canonical.csv)。
各位置接受率、显存、ablation 与更多运行记录见 [BENCHMARKS.md](BENCHMARKS.md)。

固定 **TextVQA** workload 上，DFlash / DSpark / Domino 的 AL 为
1.263 / 1.253 / 1.268，均为 128/128 token-exact。
这里评测投机接受行为与输出一致性，标准答案 VQA 准确率是另一个指标；见[benchmark 指南](docs/benchmark.md)。

**运行成本：** 当前 sequential 参考运行相对 target-only 为 0.81–0.87×。
Parallel verification 仍属实验路径，在 bf16 下可能出现输出差异。
时间与一致性结果共同记录在[验证模式说明](docs/verification-modes.md)中。

## 支持范围

- **输入：** 单张图片 + 文本、单轮、贪心解码。
- **Target：** 上述两种 Qwen-VL 模型；其他家族需要 adapter 和验证。
- **硬件：** 实测参考为单张 48 GB GPU；24 GB preset 是建议配置，尚未在实际 24 GB 卡上验证。
- **Serving：** 生产 vLLM/SGLang 集成仍为后续工作；[serving bridge](docs/serving-bridge.md)描述拟议交接方式。
- **复现：** 数据与生成协议可复现，bf16 训练不保证 bitwise deterministic；小于 2% 的 AL 差异可能落在运行噪声内。

## 文档

| 入口 | 链接 |
| --- | --- |
| 使用项目 | [环境](docs/env-preflight.md) · [示例](examples/README.md) · [在线训练](docs/online-training.md) · [实测运行](docs/measured-runs.md) |
| 理解方法 | [算法](docs/algorithms/) · [参考指南](docs/REFERENCE.md) · [上游审计](UPSTREAM_AUDIT.md) |
| 复现实验 | [基准](BENCHMARKS.md) · [Harness](docs/benchmark.md) · [发布验证](RELEASE_VALIDATION.md) |
| 扩展项目 | [添加算法](docs/adding_an_algorithm.md) · [支持矩阵](SUPPORT_MATRIX.md) · [贡献指南](CONTRIBUTING.md) |

## 贡献与归属

欢迎提交复现记录、聚焦的修复与清楚的使用示例。
当前 release policy 与验证要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
算法通过 `DrafterAlgorithm` 接入，目标家族通过 `VLMAdapter` 隔离。

Apache-2.0。移植组件保留上游归属，见 [NOTICE](NOTICE)。
来源 commit 与实现范围见 [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md)；引用信息见 [CITATION.cff](CITATION.cff)。

<sub><strong>为什么叫 Vela？</strong> Vela 意为“船帆”，也是船帆座的名称。
轻量 drafter 借助目标模型提供的视觉上下文前行。</sub>
