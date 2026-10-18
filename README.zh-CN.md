<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>用一个工作流训练、评测多模态投机草稿模型。</strong><br />
  VelaDraft 帮助你用自己的图文数据训练、评测和比较多模态投机草稿模型：多种 drafter
  共用数据处理、冻结目标特征、训练与评测流程，让你把精力放在方法和数据上；对
  DFlash、DSpark 这类并行 block-diffusion drafter 提供了完整流程。
</p>

<p align="center">
  <sub>主要验证对象为 Qwen2.5-VL-3B 与 Qwen3-VL-8B，不同算法、配方和运行模式的验证规模不同；
  SmolVLM2 目前有 DFlash 单图 smoke。详见<a href="SUPPORT_MATRIX.md">支持矩阵</a>。</sub>
</p>

<p align="center">
  <a href="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml"><img src="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-12263a" alt="Python 3.10+" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-12263a" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="docs/model-zoo.md">模型下载</a> ·
  <a href="BENCHMARKS.md">基准结果</a> ·
  <a href="docs/README.md">文档</a> ·
  <a href="SUPPORT_MATRIX.md">支持矩阵</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

🎉 最新动态
- [2026-10-18] 🔥 新增 Qwen3.6-35B-A3B 目标模型 adapter，接入 partial
  rotary，支持回答重新生成、特征捕获与 dense drafter 训练路径
  （#18，感谢
  @khazic）。详见支持矩阵。
为什么选择 VelaDraft
VelaDraft 是项目名；当前包名与命令名仍是 vlm-dflash。
- 复用昂贵的目标特征。 兼容的 offline capture 可供多个方法使用；online
  训练可选择内存缓存，减少重复的目标前向提取。
- 用同一流程比较方法。 从自己的图文数据出发，统一运行训练、接受率评测和
  导出；每个记录在案的结果都携带自己的配置与协议。
- 拿到可检查的训练产物。 checkpoint、目标身份、训练记录和评测结果一起保留，
  便于恢复、复现和继续接入工具链。
<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="图文数据通过目标模型生成回答并捕获特征。训练可选 offline feature store 路径，或 online 实时提取（带可选内存缓存）；校验面板比较投机输出与 target-only 贪心解码是否逐 token 一致，随后导出 checkpoint。" width="1100" />
</p>

这里的“统一”是统一接口与协议记录，不代表所有算法拥有相同 objective、同样预算
或相同成熟度。
支持的算法
七种 drafter 共用同一套 train / eval / export 命令、checkpoint 格式与
逐 token 验证路径：
算法	核心机制	验证状态
[DFlash](docs/algorithms/dflash.md)	共享 FC context + block drafting	已验证
[DSpark](docs/algorithms/dspark.md)	Markov logit bias + confidence	已验证
[Domino](docs/algorithms/domino.md)	因果 GRU correction	已验证
[JetSpec](docs/algorithms/jetspec.md)	因果块内 mask + forward KL	已验证（按 sequential 服务；生产 tree runtime 未移植）
[DFlash2](docs/algorithms/dflash2.md)	局部卷积 + candidate selector	实验性重建目标（上游训练目标未公开）
[DFlare](docs/algorithms/dflare.md)	逐 draft 层融合 + 独立 context K/V	已验证
[DFly](docs/algorithms/dfly.md)	FC + fusion residual + hidden correction	已验证，含已移植的上游两阶段配方


配方选择（默认 objective、可选 recipe、必需 teacher 信号、已发布权重）汇总在
[docs/README.md](docs/README.md#choose-a-method--recipe) 一张表中；逐 target 的
证据状态见[支持矩阵](SUPPORT_MATRIX.md)（自动生成）。
范围与限制：
- 输入： 图片（一张或多张）+ 文本，单轮或多轮，贪心解码。多轮监督默认是
  last_turn_only：较早的 assistant 轮次按原样作为 teacher-forced 上下文，只有
  最后一个回答会被重新生成并监督（见 [data.md](docs/data.md)）。
- 硬件： 实测参考为单张 48 GB GPU；24 GB preset 仍是建议配置，尚未在真实
  24 GB 卡上记录运行。训练无 DDP / 多节点启动路径。
- 训练模式： 默认 offline（读 staged feature store）；online 让冻结目标模型
  驻留并实时提取特征；--features 复用已有 store。“online”不是 on-policy
  rollout 训练。
- Qwen3.6-35B-A3B： qwen3_5_moe adapter 已实现并通过 CPU 测试；上表的
  方法验证状态不代表这些方法已在此目标模型上完成验证。dense drafter 必须显式
  设置 drafter.intermediate_size。现已接入[目标模型跨卡加载](docs/online-training.md#devices-what-is-supported-today)
  和 [chat-template 参数](docs/REFERENCE.md#chat-template-kwargs-render-protocol)，
  包括 enable_thinking: false。端到端训练尚待验证；混合循环缓存回滚实现前，
  投机解码的 eval / benchmark / spec_generate 入口会明确报错。
- Serving： 生产 vLLM/SGLang 集成仍为后续工作；
  [serving-bridge.md](docs/serving-bridge.md) 是实验性导出/加载工具，其正确性
  与性能验收尚未完成。
- 复现： 数据与生成协议可复现；bf16 训练不保证 bitwise deterministic。
从这里开始
路径	适合场景
[跑通完整流程](docs/getting-started.md)	安装、准备本地 target、doctor、第一个 demo，然后用自有数据 train
[体验已发布 drafter](docs/model-zoo.md)	直接拿训练好的 checkpoint 跑 demo --drafter：报告接受长度、各位置接受率和 token 一致性，无需训练


curl -fL -O https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz
vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
参考机记录：42 s，AL 1.313，token-exact 4/4（随机初始化基线 demo：45 s，AL
1.000）。两版 DFly 权重都已发布——共享 block CE 默认版与已移植上游两阶段配方版：
Drafter	Target	下载
DFlash	Qwen2.5-VL-3B	387 MB zoo-v0.3.0
DFly（共享 block CE）	Qwen2.5-VL-3B	420 MB zoo-v0.3.1
DFly（上游两阶段配方，AL 1.839）	Qwen2.5-VL-3B	420 MB zoo-v0.3.2
DFlash	Qwen3-VL-8B	1.00 GB zoo-v0.3.1


全部 9 个归档（含 Domino、DSpark、JetSpec、DFlash2、DFlare）、校验方式与加载
命令见 [model-zoo.md](docs/model-zoo.md)。
快速开始
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft

python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"        # 源码安装；目前没有 PyPI 发布

# 目标模型需要提前准备一次（config 使用 local_files_only，不会自动下载）。
export VLM_DFLASH_TARGET="$PWD/assets/models/Qwen2.5-VL-3B-Instruct"
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$VLM_DFLASH_TARGET"
# 受限网络：同一命令前加 HF_ENDPOINT=https://hf-mirror.com

vlm-dflash doctor --target "$VLM_DFLASH_TARGET"
# 成功标志：exit 0、若干 [ok] 行，以及一行解析出的 contract
# （mask=151669 mrope=[16,24,24] layers=[1,17,33] ...）

vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
train-demo 在生成的图片上跑完整流程（prepare → 目标回答生成 → capture →
train → eval → export）。参考机记录（单张 RTX A6000 48 GB，历史数据）：3 min
13 s，loss 13.0 → 0.013，eval AL 1.138（随机初始化 1.000），token-exact 4/4，
导出位于 runs/demo_dflash/export/。这是流水线 smoke，不是质量结论。
长训、日志、停止条件、恢复与硬件要求见
[getting-started.md](docs/getting-started.md) 与
[training.md](docs/training.md)。
今天就能做的三件事
1. 用自己的图文数据训练。 每行一个 JSON 对象，image 相对 --images-dir
解析：
{"id":"row-001","image":"cat.jpg","question":"What is in this image?","answer":"A cat sleeps on a sofa."}
python examples/train_from_jsonl.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images \
  --limit 32 --steps 60
依次执行 prepare → regen → capture → train → eval，并在 run 目录留下
checkpoint 和导出。记录结果（12 行 COCO 数据，10 步）：loss 7.69 → 7.06，
visual check 通过，token-exact 4/4。注意 --limit 是训练行数，验证集在其
之上另行划分。行格式、划分规则与多轮监督细节：
[data.md](docs/data.md)。
2. 一次 capture，比较多个 drafter。 昂贵的目标计算（regen + capture）
只跑一次，之后各 drafter 从同一 store 训练，训练循环中不再加载目标模型：
python examples/reuse_features_compare.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images --steps 20
记录结果（12 行 COCO 池，5 步，smoke 规模）：DFlash AL 1.153 / DSpark
AL 1.094，均 token-exact。复用是校验过的，而非默认成立——store 的 target 身份
与各方法消费的层必须匹配，不匹配会直接报错（[training.md](docs/training.md)）。
CLI 形式为 vlm-dflash train --features <store>。
3. 按资源选择训练方式。 offline 读取已捕获的 feature store（训练时目标不在
GPU 上；2K capture 约占 9.4 GiB 磁盘）；online 让冻结目标模型驻留并实时提取；
多 epoch 的 online 可以选择主机内存缓存（不落盘）：
# offline（默认）
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly_2k

# online，并启用可选内存缓存
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --mode online \
  --run-dir runs/dfly_online --set train.online_feature_cache_gb=8
记录的缓存计数（129 行 / 6 epoch）：645 hits / 129 misses / 129 stores /
0 evictions，峰值 830 MiB（上限 8 GiB），可在 train_summary.json 的
feature_source.feature_cache 查看。命中按位返回存储的张量；缓存不进任何
digest，其训练期数值边界见
[online-training.md](docs/online-training.md)。
结果与证据
所有测量——接受长度、各位置接受率、token-exact 门、训练成本、消融、运行时
探针——都连同协议记录在 [BENCHMARKS.md](BENCHMARKS.md)、
[RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) 与
[measured-runs.md](docs/measured-runs.md) 中。机器可读子集：
benchmarks/results/zoo_canonical.json。
数字的含义需要说清：
- AL 是每轮目标验证接受的 token 数，不是加速倍数。token-exact 表示投机
  输出逐 token 复现目标模型自己的贪心解码，是输出一致性门，不是任务准确率。
  相对 ground truth 的任务分数（TextVQA/DocVQA/ChartQA 的答案准确率）
  尚未报告。
- 基准表里的规范运行来自 offline 路径；online smoke 验证的是流水线
  （token-exact、provenance、计数器），不是质量。
- 覆盖范围是显式的：[支持矩阵](SUPPORT_MATRIX.md) 逐格给出
  canonical-validated / reduced / smoke-only / unsupported，每行可追溯到
  记录的 run id。
文档
[README.md](docs/README.md) 是任务索引：安装与首跑、自有数据、方法与配方
选择、长训、online、评测与排障。常用直达：
[model-zoo.md](docs/model-zoo.md) · [data.md](docs/data.md) ·
[training.md](docs/training.md) · [online-training.md](docs/online-training.md) ·
[troubleshooting.md](docs/troubleshooting.md) · [algorithms/](docs/algorithms/) ·
[benchmark.md](docs/benchmark.md) · [env-preflight.md](docs/env-preflight.md)
（历史主机记录）· [REFERENCE.md](docs/REFERENCE.md)（实现与数值语义）。
贡献与引用
欢迎提交复现记录、聚焦的修复与清楚的使用示例。当前范围与发布政策见
[CONTRIBUTING.md](CONTRIBUTING.md) 与 [ROADMAP.md](ROADMAP.md)。算法通过
DrafterAlgorithm 接入，目标家族通过 VLMAdapter 隔离。
Apache-2.0。移植组件保留上游归属，见 NOTICE 与
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md)。
@misc{veladraft2026,
  title = {VelaDraft: Multimodal Speculative Drafter Training},
  year  = {2026},
  url   = {https://github.com/LUOseu/VelaDraft}
}
<p align="center">
  <sub><strong>为什么叫 Vela？</strong> Vela 意为“船帆”，也是船帆座的名称。
  轻量 drafter 借助目标模型提供的视觉上下文前行。</sub>
</p>
