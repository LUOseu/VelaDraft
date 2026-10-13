<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>面向 Qwen-VL 的多模态并行草稿模型训练工具。</strong><br />
  带上你的图文数据，针对冻结目标模型训练草稿模型，并逐 token 验证输出——训练、恢复、评测、导出。
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
  <a href="#基准结果">基准与历史记录</a> ·
  <a href="#文档">文档</a> ·
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/CONTRIBUTING.md">参与贡献</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

VelaDraft 是面向多模态 block-parallel drafter 的统一训练工具，支持 DFlash、
DSpark、Domino、JetSpec、DFlash2、DFlare 和 DFly。针对冻结的 Qwen-VL target，
提供离线特征复用、在线特征提取、断点恢复、逐 token 输出验证和 checkpoint 导出。

DFlare 与 DFly 已集成到 main。当前 DFly 使用共享的 block-CE 训练配方，
上游两阶段训练配方作为独立能力继续补全。

- **多方法统一接口。** 七种 drafter 共用同一套 `train` / `eval` / `export`
  命令、checkpoint 格式与验证路径，方法对比建立在同一基底上。
- **离线 / 在线训练。** 默认离线：训练阶段目标模型不驻留 GPU，直接读取捕获的
  feature store；`--mode online` 在训练过程中从冻结的目标模型实时提取特征。
- **显式层捕获与兼容复用。** 每次 capture 记录自身持有的目标层有序列表；消费方
  按显式 layer id 切片，因此同一 store 可供已消费层在其中的多个方法（及不同
  drafter 深度）复用；缺少的层会直接报错，绝不猜测。
- **恢复、验证与导出。** 断点恢复在写入前校验解析后的训练计划与特征来源；
  sequential 评测逐 token 复现目标模型的贪心输出（`token_exact`）并给出各位置
  接受率；导出产物自描述（权重、配置、model card、校验值）。

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

省略 `--drafter` 可运行随机初始化的接受率基线。

### 体验完整训练流程

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

在生成的图片上运行 prepare → 目标回答生成 → capture → 小规模训练 → eval → export。
这是完整流程的 smoke test；训练有实用价值的 drafter，需要真实数据与完整配方。

## 流程

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="图文数据通过目标模型生成回答与捕获特征。离线或在线训练从七种算法（DFlash、DSpark、Domino、JetSpec、DFlash2、DFlare、DFly）中选择其一，再校验输出并导出 checkpoint。" width="1100" />
</p>

将昂贵的目标模型计算与 drafter 训练分开：兼容的捕获特征可供多种算法复用，
**评测时重新加载目标模型**。每个 checkpoint 都记录对应的 target contract 与算法信息。
也可以直接针对在线的冻结目标模型训练（`train --mode online`），见
[在线训练](docs/online-training.md)。

## 支持的算法

七种算法使用相同的 train、eval 和 export 命令。

| 算法 | 核心机制 | 本仓库训练配方 |
| --- | --- | --- |
| [DFlash](docs/algorithms/dflash.md) | 共享 FC context + block drafting | 参考 block CE |
| [DSpark](docs/algorithms/dspark.md) | Markov logit bias + confidence | CE + 已接入的辅助目标（见方法文档） |
| [Domino](docs/algorithms/domino.md) | 因果 GRU correction | 已接入的 curriculum 配方 |
| [JetSpec](docs/algorithms/jetspec.md) | 因果块内 mask + teacher distillation | forward-KL |
| [DFlash2](docs/algorithms/dflash2.md) | 局部卷积 + candidate selector | 实验性重建目标 |
| [DFlare](docs/algorithms/dflare.md) | 逐 draft 层融合 + 独立 context K/V | 共享 block CE |
| [DFly](docs/algorithms/dfly.md) | FC + fusion residual + hidden correction | CE 适配，上游完整配方待接入 |

DFlare 与 DFly 已 **集成到 main**，并在两个 Qwen-VL target 上完成仓库内验证
（Qwen2.5-VL-3B 全流程、Qwen3-VL-8B 小规模验证）；发布验证状态单独记录在
[支持矩阵](SUPPORT_MATRIX.md)与[发布验证](RELEASE_VALIDATION.md)中。

DFly 说明：训练使用共享 block-CE 配方（draft config、checkpoint 与导出中记录
机器可读的 `objective_profile=controlled_ce`）；上游 LK/D-PACE → end-to-end TV
配方未移植。推理时 block backbone 并行执行，hidden-states correction 与 LM-head
投影按 slot 顺序、基于此前实际提议的 token 逐个执行——这是已说明的本地适配。

DFlash2 的上游训练目标未公开，本项目使用实验性重建；JetSpec 的生产 tree runtime
未移植。各 target 的状态与能力细节见[支持矩阵](SUPPORT_MATRIX.md)、
[算法矩阵](ALGORITHM_MATRIX.md)与[上游审计](UPSTREAM_AUDIT.md)。

## 模型下载

已发布的 **Qwen2.5-VL-3B-Instruct** drafter：

| Drafter | 下载 | Model card |
| --- | --- | --- |
| **DFlash** · 冻结块草稿参考实现 | [387 MB](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz) | [详情](docs/zoo/qwen25vl-3b-dflash.md) |
| **Domino** · GRU 修正 | [464 MB](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-domino.tar.gz) | [详情](docs/zoo/qwen25vl-3b-domino.md) |

这些权重是**流水线验证用的参考产物**：按已记录的开发协议训练，用于跑通
`训练 → 评测 → 导出`、集成工作和输出一致性检查，并非面向生产调优或充分训练的 drafter；
它们产生的接受指标属于开发期测量，与训练协议及其余运行一并记录在
[BENCHMARKS.md](BENCHMARKS.md)。两个 export 均在下载后重新通过 token-exact gate；见[发布证据](RELEASE_VALIDATION.md)。

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

### 方法专属 config

仓库内的 config 是各方法的入口（另有 `configs/dspark`、`configs/domino`、
`configs/jetspec`、`configs/dflash2`）；请先把 target 与数据路径配置到你的环境——
它们是配方，不能在没有数据准备的情况下直接执行。

```bash
# 离线配方
vlm-dflash train --config configs/dflare/qwen2_5_vl_3b_2k.yaml --run-dir runs/dflare
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly

# 同一方法，在线提取目标特征
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --mode online --run-dir runs/dfly_online

vlm-dflash eval runs/dfly
vlm-dflash export runs/dfly
```

### 跨运行 / 跨方法复用 capture

```bash
vlm-dflash train \
  --target "$VLM_DFLASH_TARGET" --dataset data/rows.jsonl \
  --algorithm dspark --run-dir runs/dspark_2k \
  --features runs/shared_2k/features --steps 792
```

复用是校验过的，而非默认成立：store 的 target/processor/mask/MRoPE 身份与
其捕获层有序列表必须与本次运行一致；每个方法按显式 layer id 消费所需层
（`algorithm.<method>.context_layer_ids` 或 draft routing）。store 缺少被消费的层时
会直接拒绝并提示重新捕获，绝不静默改切片。请在运行配置中通过
`drafter.capture_layer_ids` 声明 store 的层列表；drafter 深度与捕获不同的运行，
只要其消费层包含在内即可复用。复用记录写入 `provenance.json`。
Canonical 2K capture 占用约 **9.4 GiB** 磁盘。

[对比示例](examples/reuse_features_compare.py)展示一次 capture 后分别训练、评测 DFlash 和 DSpark。

### 从断点恢复

```bash
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly --resume
```

`--resume` 会在任何写入之前执行只读预检：特征来源 digest 与解析后的训练计划
（seed、batch 几何、优化器与 schedule、dtype、drafter 与方法专有参数、数据集长度）
必须与 checkpoint 完全一致。被拒绝的恢复不会改动 run 目录中的 config、checkpoint、
log 与 provenance（字节级不变）；合法的恢复从记录的 step 继续，并重放优化器/RNG 状态。

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

所有性能测量（接受长度、各位置接受率、耗时、token-exact 结果与
ablation）都保存在开发记录中：[BENCHMARKS.md](BENCHMARKS.md)、
[RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) 与 [docs/measured-runs.md](docs/measured-runs.md)。
README 有意不展示小规模训练数字和算法排名：已记录的运行是同一配方下的
流水线/协议证据，不是对方法质量或墙钟加速的声明。

项目在任何规模下都验证的是**正确性**：sequential 投机解码必须逐 token
精确复现目标模型自己的贪心输出（`token_exact`），并提供各位置接受率记账。
详见[验证模式说明](docs/verification-modes.md)。

## 支持范围

- **输入：** 单张图片 + 文本、单轮、贪心解码。
- **Target：** 上述两种 Qwen-VL 模型；其他家族需要 adapter 和验证。
- **硬件：** 实测参考为单张 48 GB GPU；24 GB preset 是建议配置，尚未在实际 24 GB 卡上验证。
- **Serving：** 生产 vLLM/SGLang 集成仍为后续工作；[serving bridge](docs/serving-bridge.md)描述拟议交接方式。
- **复现：** 数据与生成协议可复现，bf16 训练不保证 bitwise deterministic；
  接受指标以测量记录（见[基准结果](#基准结果)）为准。

## 文档

| 入口 | 链接 |
| --- | --- |
| 使用项目 | [环境](docs/env-preflight.md) · [示例](examples/README.md) · [在线训练](docs/online-training.md) · [实测运行](docs/measured-runs.md) |
| 理解方法 | [算法](docs/algorithms/) · [支持矩阵](SUPPORT_MATRIX.md) · [参考指南](docs/REFERENCE.md) · [上游审计](UPSTREAM_AUDIT.md) |
| 复现实验 | [基准](BENCHMARKS.md) · [Harness](docs/benchmark.md) · [发布验证](RELEASE_VALIDATION.md) |
| 扩展项目 | [添加算法](docs/adding_an_algorithm.md) · [贡献指南](CONTRIBUTING.md) |

## 贡献与归属

欢迎提交复现记录、聚焦的修复与清楚的使用示例。
当前 release policy 与验证要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
算法通过 `DrafterAlgorithm` 接入，目标家族通过 `VLMAdapter` 隔离。

Apache-2.0。移植组件保留上游归属，见 [NOTICE](NOTICE)（含基于 AngelSpec 的
DFlare/DFly 移植，Apache-2.0 及其列出的第三方 MIT 组件）。
来源 commit 与实现范围见 [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md)；引用信息见 [CITATION.cff](CITATION.cff)。

<sub><strong>为什么叫 Vela？</strong> Vela 意为“船帆”，也是船帆座的名称。
轻量 drafter 借助目标模型提供的视觉上下文前行。</sub>
