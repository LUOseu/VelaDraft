# C0 novelty 初筛 — D1 候选机制（代价感知重分组）· 2026-10-15 02:50 CST

- 性质：**第一轮粗查（时间盒 ~30min；用于定位风险评估，不是 novelty 结论）**。来源 = arXiv API + OpenAlex；Semantic Scholar API 本轮全程 429（共享限流）；本会话无 WebSearch 工具 → Megatron-LM / torchtune / HF 等**工程实现文档未核查**（列待办）。下表所有论文均来自检索返回，未人工编造。
- 用途：为 D1 的 related-work 定位与最终"平台 vs 论文"分列提供先导。

## 1. 机制速写（供精读对照；C0 蒸馏，D1 可更正）

1. **同 step 内重分组**：把变长样本的逻辑批重分组为若干物理小组，按块代价最小化（去 invalid/padding/anchor 外块），而非仅长度排序；
2. **损失语义保持**：逐样本权重随组承载（g_i=1/Z_batch；逐组 backward），与 canonical 逻辑批**数值等价**（fp64 oracle exact；bf16 级漂移 ≤7.5e-3）；
3. **效果**：峰值 −57~67%（固定 logical batch）→ 容量 ×2–3（固定 peak）；步时 +13%（离线；sub1 无增量——R4 已封顶）；
4. **设定**：多模态 block-drafter（anchor/block 结构；W3/MIX 真实行）。

## 2. 先验工作聚类（closest prior work）

| 家族 | 代表作（来源/年） | 初判差异 |
|---|---|---|
| ① 序列 packing / padding-free | "Efficient Sequence Packing without Cross-contamination"（arXiv:2107.02027；2021） | 去 padding 为吞吐；防串扰掩码；未见"逐样本权重等价"与"分组代价模型"主张（待核） |
| | 《Beyond Concatenation: A Survey of Sequence Packing for Long-Context LLM Training》（SSRN；2026） | **必读**——若综述已 formalize 权重等价重组，则 delta① 受损 |
| ② bucketing / padding-free 微调 | TokenDrop+BucketSampler（2023） | 按长度桶采样+丢 padding token；采样层机制，非 step 内精确重分组 |
| | NMT minibatch 策略（arXiv:1706.05765；2017）/ Interspeech 2021 batching | 历史先例："按长度组织批量减少 padding"早已是常规操作 |
| ③ 变长数据组织 | Dataset Decomposition（γ variadic；NeurIPS 2024） | 数据侧 concat-and-chunk 替代；非步内内存机制 |
| ④ 自动 microbatching / 内存适配 | Varuna（arXiv:2111.04007；2021） | 自动切 microbatch+算子流水以适配内存；数学不改目标；不针对 padding/invalid slack |
| | Balanced Memory Workload Optimization（TKDE 2024） | 微批内存平衡服务并行调度 |
| ⑤ 代价感知打包 / 负载均衡 | Libra（arXiv:2607.23250；2026） | 注意力代价（ΣL²）感知的定长池化；服务并行负载；单卡 step 内"精确等价"不同 |
| ⑥ 因果负载均衡切分 | Striped Attention（arXiv:2311.09431；2023） | 按因果工作量切分到多设备；并行语义 |
| ⑦ 稀疏变尺寸分组执行 | MegaBlocks（arXiv:2211.15841；2022） | 变尺寸块分组+block-sparse；MoE 路由域 |
| ⑧ VLM / 工程侧 | X-MoE（2025）；PackMamba（2025）；Open-Qwen2VL（2025）等 | "padding-free/高效训练"工程线；block/anchor 语义未见 |

## 3. 初判（供 go/no-go 参考）

- **"去掉 padding 浪费、打包提速不掉点"＝多年、多子领域既存实践**（2017→2026 有连续先例+综述）。单凭此叙述的 novelty = **低**。
- 仍可能站得住的 delta（按可辩护度排序，均需深读确认）：
  1. **单 step 内"精确损失等价重分组"的 formalization**（fp64 证明 + bf16 带宽口径）——若综述/文献未覆盖，可作为语义/系统层的可引贡献；
  2. **容量轴主张**（固定 peak 下 logical batch ×2–3 的"放得下"杠杆，而非吞吐）——文献目标多为吞吐/上下文；
  3. **浪费分解诊断**（invalid-block slack 占残差 96–98% 的逐通道分解）在 block-drafter 设定下或有独立价值；
  4. 多模态 block-drafter 真实设定（与纯文本 packing 的差异必须写实，不能夸大为"新问题"）。
- **结论：工程价值独立成立；论文新颖性未证、空间拥挤。** 任何论文级声明前必须：（a）精读综述 + Libra + 2107.02027 + Varuna（4 篇；建议 D1 或联合作业）；（b）补工程文档核查（Megatron/torchtune/HF；本会话无 WebSearch）；（c）由 D1 在 REPORT 增 related-work 差异表（C0 可起草，D1 审签）。

## 4. 检索留痕（可复跑）

- 实际生效查询 — OpenAlex: "sequence packing large language model training" / "variable length sequences batching memory efficient training" / "automatic microbatching memory" / "equivalence gradient accumulation large batch" / "length bucketing batching padding waste"；arXiv: abs:"padding-free" / abs:"microbatch"+memory / abs:"sequence packing"+training / abs:"gradient accumulation"+equivalence / abs:"ragged"+"batch"。
- 缺口：S2（venue/引用数）→ 429；工程博客/文档；非英语/非 arXiv 文献；引用网络（2107.02027 的 forward citations）——深读阶段补。
