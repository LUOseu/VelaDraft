# vlm-dflash

**从图文数据出发，训练 Qwen-VL 的多模态并行 drafter（DFlash 家族）并做 token 级精确校验的最简可复现路径。**

![pipeline](docs/assets/pipeline.svg)

纯文本的推测解码栈会在 Qwen-VL 的多模态细节上翻车：processor 像素边界、MRoPE
position ids、图像 token span、保留的 mask embedding 行、tied / untied
`lm_head`。vlm-dflash 把这些环节一次性钉死，让五个 drafter 算法
（DFlash / DSpark / Domino / JetSpec / DFlash2）用同一套命令、同一份捕获特征
完成训练、评测与导出。这里没有新算法、也不是 serving 框架——目标是一个正确、
可检查的多模态训练闭环。

英文主 README：[`README.md`](README.md)；支持矩阵：
[`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md)（由算法注册表的机器可读元数据生成）。

## 1. 直接试用预训练 drafter（无需训练）

**Qwen2.5-VL-3B-Instruct** 的 v0.3.0 冻结导出（已过 token 级精确校验）：

```bash
# 下载（387 MB，时间取决于网络）
curl -L -O https://github.com/LUOseu/vlm-dflash/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar xzf qwen25vl-3b-dflash.tar.gz

# 运行（A6000 上约 2 分钟；需要本地 Qwen2.5-VL-3B 模型目录）
vlm-dflash demo --target <Qwen2.5-VL-3B-Instruct> --drafter qwen25vl-3b-dflash
```

`demo` 会生成一个极小的免版权样本，打印生成结果、校验 speculative 输出与
target 自身 greedy 输出逐 token 一致，并报告接受长度（AL）与 per-slot 接受率。
不带 `--drafter` 时评测的是随机初始化 drafter——那是接受率的基线（AL≈1.00），
用于诊断。AL 是接受性指标，不是加速比；sequential 验证路径以正确性优先。

## 2. 训练自己的 drafter

需要：一张 CUDA GPU（参考硬件为 A6000 48 GB）、本地
Qwen2.5-VL-3B-Instruct、一份图文数据（smoke preset 自带 COCO caption 路径）。

```bash
git clone https://github.com/LUOseu/vlm-dflash && cd vlm-dflash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[tested]"

export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct
vlm-dflash doctor --target "$VLM_DFLASH_TARGET"      # 环境 + 模型 contract 体检
vlm-dflash train --preset smoke --limit 64          # 数据 -> regen -> 捕获 -> 训练
vlm-dflash eval runs/smoke_coco                     # token 精确门 + AL + per-slot
vlm-dflash export runs/smoke_coco                   # 自描述 checkpoint + model card
```

尚未发布 PyPI，从 clone 安装（`pip install -e .`）是唯一受支持的安装方式。
另有一个约 3 分钟、**无需下载数据集**的全链路 demo：

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

## 3. 支持的算法与 target

| 算法 | Qwen2.5-VL | Qwen3-VL | 成熟度 | 说明 |
| --- | --- | --- | --- | --- |
| DFlash | validated | validated | stable | 冻结参考实现；decayed block CE |
| DSpark | validated | 缩减规模验证 | validated | Markov logit bias + 置信度头 |
| Domino | validated | 缩减规模验证 | validated | causal GRU 修正 |
| JetSpec | validated | 缩减规模验证 | validated | forward-KL + 因果 in-block mask；树形 runtime 未移植 |
| DFlash2 | validated | 缩减规模验证 | experimental | 目标函数为重构实现（上游未公开） |

target 列语义与 [`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md) 一致：`validated` 表示
已跑通 canonical 协议；"缩减规模验证" 指完整流程在 256 行切片 / 16 prompt
评测下通过（见 [`RELEASE_VALIDATION.md`](RELEASE_VALIDATION.md)）。所有
family-specific 细节（mask token、MRoPE sections、draft 层路由、像素边界、
tied/untied head）都在运行时从 target 解析成 `contract.json` 并校验。

## 4. Model Zoo

| Checkpoint | 算法 | AL（2K 协议） | 下载 | Model card |
| --- | --- | ---: | --- | --- |
| `qwen25vl-3b-dflash` | DFlash（冻结参考） | 1.6298 | [387 MB](https://github.com/LUOseu/vlm-dflash/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz) | [card](docs/zoo/qwen25vl-3b-dflash.md) |
| `qwen25vl-3b-domino` | Domino | 1.7637 | [464 MB](https://github.com/LUOseu/vlm-dflash/releases/download/zoo-v0.3.0/qwen25vl-3b-domino.tar.gz) | [card](docs/zoo/qwen25vl-3b-domino.md) |

每个压缩包包含权重、`draft_config.json`、`algorithm.json`、
`target_contract.json`、`training_manifest.json`、`checksums.json`（逐文件
sha256）与 `model_card.md`；发布后已重新下载并实测通过 token 精确校验。

## 5. Benchmark（Qwen2.5-VL-3B，2K 语料，128 prompts / 128 tokens，sequential 验证，全部 token-exact）

接受性 / 训练行为（完整表格与消融见 [`BENCHMARKS.md`](BENCHMARKS.md)，机器可读
结果：[`benchmarks/results/zoo_canonical.json`](benchmarks/results/zoo_canonical.json)）：

| 算法 | AL | 随机初始化 | 参数量 | 训练步数 | 峰值显存 | 训练时长 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash | 1.6298 | 1.003 | 243.8M | 792 | 29.5 GB | 9.8 min |
| DSpark | 1.7354 | 1.003 | 321.6M | 792 | 38.1 GB | 11.8 min |
| Domino | 1.7637 | 1.003 | 292.9M | 792 | 38.4 GB | 12.9 min |
| JetSpec | 1.5987 | 1.003 | 243.8M | 792 | 29.7 GB | 20.6 min |
| DFlash2（实验目标函数） | 1.9541 | 1.003 | 328.5M | 792 | 31.1 GB | 22.7 min |

实际推理成本（sequential 模式，**不是加速比**）：3B DFlash 4.14 s/sample
（0.87x）、DSpark 4.10（0.86x）、Domino 4.30（0.84x）、JetSpec 4.07（0.86x）、
DFlash2 4.43（0.82x）。parallel / strict-parallel 模式可到 ~1.0–1.25x，但 bf16
下不是逐 bit 精确，仅作为实验性吞吐诊断（见
[`docs/verification-modes.md`](docs/verification-modes.md)）。TextVQA 上的
数字衡量的是 **接受性与精确性**，不是标准答案准确率。

## 6. 用自己的数据训练

`prepare_data` 接受本地 JSONL（`dataset.source: jsonl`），每行一张图 + 一组
问答，支持 OpenAI `messages` / ShareGPT `conversations` / 扁平
`{question, answer, image}` 三种形状（见
[`vlm_dflash/data/schema.py`](vlm_dflash/data/schema.py)）：

```json
{"id": "ex1", "question": "<image>\nDescribe this image in detail.", "answer": "…", "image": "000000.jpg"}
```

跨算法复用同一份捕获特征（zoo 协议）：

```bash
vlm-dflash train --dataset data/rows.jsonl --target "$VLM_DFLASH_TARGET" \
    --algorithm dspark --run-dir runs/dspark_2k \
    --features runs/shared_2k/features --steps 792
```

训练前会校验特征库的 target identity / manifest / 保存的 target head，复用
条件（目录、样本数、序列长度、dtype、manifest 摘要）写入
`provenance.json`。2K 特征库约 9.4 GiB，每多训练一个算法即可省去一遍
regen + capture。

## 7. 已知限制

- 仅单图 + 文本；不支持 video、多图、多轮对话。
- 两个 target 家族（Qwen2.5-VL-3B、Qwen3-VL-8B）；其他家族需要小型 adapter。
- 没有生产级 serving 集成；sequential 验证以正确性优先（当前不是速度路径），
  serving bridge 是提案文档而非功能
  （[`docs/serving-bridge.md`](docs/serving-bridge.md)）。
- parallel / strict-parallel 在 bf16 下不保证逐 bit 精确。
- DFlash2 为实验性目标函数重构；JetSpec 的生产树形 runtime 未移植。
- 实测硬件为单张 48 GB 卡；24 GB 是建议但未经充分验证的目标，需下调
  `train.micro_batch_size` / `capture.seq_length`。

## License

Apache-2.0；移植/改编组件的上游归属见 [`NOTICE`](NOTICE) 与
[`UPSTREAM_AUDIT.md`](UPSTREAM_AUDIT.md)。
