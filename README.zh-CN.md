<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>面向多模态 drafter 训练、系统分析与基准评测的统一工具包。</strong><br />
  VelaDraft 将 DFlash、DSpark 等 parallel block-diffusion drafter 接入同一多模态工作流，
  并通过训练成本 profiling，指导针对这类工作负载的系统优化。
</p>

<p align="center">
  <sub>主要验证对象为 Qwen2.5-VL-3B 与 Qwen3-VL-8B，验证范围因方法、配方和模式而异；
  SmolVLM2 目前仅有 DFlash 单图 smoke。详见<a href="SUPPORT_MATRIX.md">支持矩阵</a>。</sub>
</p>

<p align="center">
  <a href="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml"><img src="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-12263a" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-12263a" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> · <a href="docs/model-zoo.md">模型下载</a> ·
  <a href="BENCHMARKS.md">基准结果</a> · <a href="docs/README.md">文档</a> ·
  <a href="SUPPORT_MATRIX.md">支持矩阵</a> · <a href="CONTRIBUTING.md">参与贡献</a>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a></p>

## 最新动态

- **[2026-10-18]** 新增 **Qwen3.6-35B-A3B target adapter**，接入 partial rotary，提供回答重新生成、特征捕获与 dense drafter 训练路径（[#18](https://github.com/LUOseu/VelaDraft/pull/18)，感谢 [@khazic](https://github.com/khazic)）。验证范围见[支持矩阵](SUPPORT_MATRIX.md#targets)。

## 为什么选择 VelaDraft

Parallel block drafter 的训练同时涉及昂贵的冻结多模态 target 和较小的可训练模型。
重复提取 target 特征、anchor 展开后的 batch，以及全词表评分，都会影响成本；
只看 drafter 大小并不足以解释训练效率。方法比较还需要明确数据、target 身份、
解码和测量协议。

VelaDraft 将三个环节连接起来：

- **共享训练接口。** 图文数据准备、target 答案再生成、特征捕获与不同方法的训练共用基础流程。
- **围绕训练特征做 profiling 和优化。** 先测量时间与内存开销，再减少重复提取和无效评分，并明确数值与资源上的取舍。
- **保留可追溯的评测证据。** 接受长度、任务质量、输出一致性、延迟与训练成本分别记录，结果关联配置、checkpoint 和验证记录。

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="VelaDraft 的三阶段流程：数据与 target 特征准备、选择 drafter 训练、评测与导出。训练系统分析依次定位提取和评分成本、减少重复工作、验证数值与资源取舍。" width="1100" />
</p>

“统一”指共享接口与协议记录，不代表各方法的目标函数、预算或成熟度完全相同。
`VelaDraft` 是项目名；当前包名和 CLI 仍为 `vlm-dflash`。

## 训练系统：从 profiling 到优化

系统优化由工作负载决定，并非每项优化都适用于每个运行。当前实现与记录包括：

| 训练特点 | 系统处理 | 证据与适用边界 |
|---|---|---|
| 冻结 target 在多个 epoch 中重复提取特征 | 复用兼容的 offline store；online 可选有容量上限的主机内存缓存，支持自适应预算与固定保留计划 | 命中缓存时原样返回已存张量；不同 batch padding 下重新提取可能改变 bf16 数值。[模式、计数器与数值说明](docs/online-training.md) |
| Anchor 展开产生零权重 padding block，但这些行仍进入 LM head 和 loss | 在 row-local 评分前压缩无效行；recipe 运行跳过不参与优化的 CE 标量 | 梯度一致性回归测试；row compaction 目前对 DFlare/DFly 自动启用，并非支持所有目标函数。[评分优化](docs/online-training.md#recipe-run-scoring-cost-drafterrecipe_row_compaction) |
| 部分 recipe stage 的分块 TV/KL 运算占比较高 | 可选编译原有运算，失败时回退 eager | 收益依赖硬件与 stage，不适用于所有 loss。[执行优化](docs/online-training.md#execution-optimization-compile-the-tvkl-chunk-arithmetic) |

Profiling 也用于排除收益有限的方向：记录的 A6000 真实样本 sweep 中，增大 teacher
分组使提取时间下降 13.3%，但估算端到端收益上限低于 3%。减少 forward 次数本身不足以
支持继续优先优化这一方向。详见[测量报告](research/system_optimization/REPORT.md)。

这些是有具体条件的工程结果，不是通用加速承诺。例如，在 512 行、3 epoch、每个容量
两组配对的 production-CLI 测试中，固定保留策略在 0.8 / 1.6 GiB 缓存容量下使 wall time
下降 **3.6% / 6.9%**。[完整协议](docs/online-training.md#static-pinned-retention-trainonline_feature_cache_policy-pinned)
将这一结果与 replay 上界分开报告。训练节省不能当作推理加速；缓存策略为可选项，
不能默认视为每组 benchmark 的配置。

## 支持的方法

七种 drafter 共享 `train` / `eval` / `export` 接口和 checkpoint 格式。
下表总结已有证据，不代表在所有 target 上完成验证，也不是大规模实验的最终排名。

| 方法 | 核心机制 | 状态 |
|---|---|---|
| [DFlash](docs/algorithms/dflash.md) | Block-parallel drafting | 已验证 |
| [DSpark](docs/algorithms/dspark.md) | Markov correction + confidence | 已验证 |
| [Domino](docs/algorithms/domino.md) | 因果 GRU correction | 已验证 |
| [JetSpec](docs/algorithms/jetspec.md) | 块内因果注意力 + forward KL | Sequential serving 已验证；parallel tree runtime 尚未移植 |
| [DFlash2](docs/algorithms/dflash2.md) | 局部混合 + candidate selector | 实验性目标重建，上游目标函数未公开 |
| [DFlare](docs/algorithms/dflare.md) | 逐层 target fusion | 已验证 |
| [DFly](docs/algorithms/dfly.md) | Fusion residual + hidden correction | 已验证，含移植的上游两阶段配方 |

[配方索引](docs/README.md#choose-a-method--recipe) 列出默认目标、可选 recipe、teacher 信号与权重；
[支持矩阵](SUPPORT_MATRIX.md) 记录逐方法、target 与运行模式的验证范围。

## 当前范围与限制

- **输入：** 一张或多张图片 + 文本，单轮或多轮，贪心解码。多轮默认 `last_turn_only`，较早的 assistant 回答作为 teacher-forced 上下文，仅重新生成并监督最后一个回答。[数据格式](docs/data.md)
- **硬件：** 已验证参考为单张 48 GB GPU；24 GB preset 仍是建议配置，没有真实 24 GB 卡上的记录。
- **训练模式：** 默认 offline staged features；online 从冻结 target 实时提取；`--features` 复用已有 store。没有 DDP / 多节点启动路径。“online”不是 on-policy rollout。
- **Qwen3.6-35B-A3B：** `qwen3_5_moe` adapter 已实现并通过 CPU 测试，上表不代表该 target 上的方法验证。Dense drafter 必须显式设置 `drafter.intermediate_size`。支持[跨卡 target 加载](docs/online-training.md#devices-what-is-supported-today)及 [chat-template 参数](docs/REFERENCE.md#chat-template-kwargs-render-protocol)，包括 `enable_thinking: false`。端到端训练尚待验证；混合循环缓存回滚实现前，投机 `eval` / `benchmark` / `spec_generate` 被明确阻止。
- **分片评测：** Capture/training 支持 target 分片不代表 serving 已支持。`eval` / `benchmark` 当前拒绝非空的 `target.device_map`。
- **Serving：** 生产 vLLM/SGLang 集成仍为后续工作。[Serving bridge](docs/serving-bridge.md) 是实验性导出/加载工具，正确性与性能验收尚未关闭。
- **复现：** 数据与生成协议可复现；bf16 训练不保证 bitwise deterministic。

## 从这里开始

| 入口 | 用途 |
|---|---|
| [完整流程](docs/getting-started.md) | 安装、准备本地 target、运行 `doctor` 与 demo，再训练自有数据 |
| [模型下载](docs/model-zoo.md) | 直接运行已发布 checkpoint，查看接受长度、逐位置接受率与 token 一致性 |

完成下方安装和 target 准备后，可直接使用已发布 drafter：

```bash
curl -fL -O https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz
vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
```

[Model Zoo](docs/model-zoo.md) 收录九个 archive，包括 Qwen2.5-VL-3B 的不同方法、
DFly 默认与上游配方，以及 Qwen3-VL-8B DFlash；提供下载、验证与加载命令。

## 快速开始

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"  # 从源码安装，目前没有 PyPI release

export VLM_DFLASH_TARGET="$PWD/assets/models/Qwen2.5-VL-3B-Instruct"
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$VLM_DFLASH_TARGET"
vlm-dflash doctor --target "$VLM_DFLASH_TARGET"
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

配置默认使用本地模型，不自动下载。网络受限时可在下载命令前设置
`HF_ENDPOINT=https://hf-mirror.com`。`doctor` 成功时应 exit 0，输出 `[ok]` 与解析后的 target contract。

`train-demo` 用生成图片跑通 prepare → regen → capture → train → eval → export。
历史单张 RTX A6000 48 GB 记录为 3 分 13 秒、loss 13.0 → 0.013、AL 1.138
（随机初始化 1.000）、4/4 token-exact，导出到 `runs/demo_dflash/export/`。
这是流程 smoke，不是任务质量结论。长训练与恢复见[入门指南](docs/getting-started.md)和[训练文档](docs/training.md)。

## 现在可以做的三件事

**1. 训练自己的图文数据。** 每行一个 JSON 对象，`image` 相对 `--images-dir` 解析：

```jsonl
{"id":"row-001","image":"cat.jpg","question":"What is in this image?","answer":"A cat sleeps on a sofa."}
```

```bash
python examples/train_from_jsonl.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images \
  --limit 32 --steps 60
```

`--limit` 指训练行数，验证行另取。多轮格式、划分规则与监督方式见[数据文档](docs/data.md)。

**2. 复用一次 capture 比较方法。** 兼容的 target 身份与所需层必须匹配，不匹配会报错：

```bash
python examples/reuse_features_compare.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images --steps 20
```

也可通过 `vlm-dflash train --features <store>` 复用特征。

**3. 根据资源选择模式。** Offline 读取 staged store；online 保持冻结 target 驻留，可选主机内存缓存：

```bash
# offline（默认）
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly_2k

# online，启用可选缓存
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --mode online \
  --run-dir runs/dfly_online --set train.online_feature_cache_gb=8
```

缓存预算、固定/自适应保留策略、恢复行为和数值条件见[online 训练](docs/online-training.md)。
模式应记录在运行协议中，不能在受控比较中悄悄更换。

## 结果与证据

公开记录包括 smoke、canonical runs 和系统研究；进行中的大规模实验不作为已完成结果。
完整协议见 [BENCHMARKS.md](BENCHMARKS.md)、[RELEASE_VALIDATION.md](RELEASE_VALIDATION.md)
与[实测运行记录](docs/measured-runs.md)。机器可读子集为 `benchmarks/results/zoo_canonical.json`。

- **AL** 是每轮 target 验证接受的 token 数，不是加速倍数。**Token-exact** 表示复现 target 自身的贪心输出，不等于任务准确率。当前尚未报告 TextVQA / DocVQA / ChartQA 对 ground truth 的任务分数。
- Canonical benchmark 使用 **offline** 路径；online smoke 验证流程、一致性、来源与计数，不证明任务质量。
- [支持矩阵](SUPPORT_MATRIX.md) 区分 canonical、reduced、smoke-only 与 unsupported，并关联运行记录。

## 文档

[文档首页](docs/README.md) 按任务组织入口。常用页面：
[模型下载](docs/model-zoo.md) · [数据](docs/data.md) · [训练](docs/training.md) ·
[Online](docs/online-training.md) · [排障](docs/troubleshooting.md) ·
[算法](docs/algorithms/) · [评测](docs/benchmark.md) ·
[环境预检](docs/env-preflight.md) · [实现参考](docs/REFERENCE.md)。

## 贡献与引用

欢迎复现、聚焦的修复和使用示例。范围与发布政策见 [CONTRIBUTING.md](CONTRIBUTING.md)
和 [ROADMAP.md](ROADMAP.md)。算法通过 `DrafterAlgorithm` 接入，target 家族通过 `VLMAdapter` 接入。

采用 Apache-2.0；移植部分保留上游归属，见 [NOTICE](NOTICE) 与 [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md)。

```bibtex
@misc{veladraft2026,
  title = {VelaDraft: Multimodal Speculative Drafter Training},
  year  = {2026},
  url   = {https://github.com/LUOseu/VelaDraft}
}
```

Vela 意为帆，也是一座星座的名字：轻量 drafter 借助 target 提供的视觉上下文前行。
