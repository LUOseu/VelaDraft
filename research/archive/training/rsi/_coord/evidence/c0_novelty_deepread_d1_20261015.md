# C0 novelty 深读（增补 #1）— D1 候选机制 · 2026-10-15 03:05 CST

- 性质：`c0_novelty_firstpass_d1_20261015.md` 的增量（第二轮；时间盒 ~45min）。新增：SSRN 综述著录核实、ACL/IJCAI/NeurIPS 系对照、**工程侧核查（上轮缺口，本轮补齐）**。
- 来源：OpenAlex API、arXiv API、GitHub raw（存在性核查）。均来自检索返回；未编造。SSRN 全文：Mac 与服务器抓取均失败（000/超时）→ 维持"内容未读"。
- 读级标注：**[摘]** = 摘要精读；**[著]** = 著录级（标题/出处/DOI 已核）。

## 1. 对照文献（本轮核实）

| 文献 | 出处/年 | 读级 | 与 D1 的关系 |
|---|---|---|---|
| Beyond Concatenation: A Survey of Sequence Packing for Long-Context LLM Training | SSRN EJ 2026；DOI 10.2139/ssrn.7343278 | [著] | **综述级风险项**：内容未读；若已 formalize"等价重分组"，delta① 降级。列为论文路线前置条件项。 |
| Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in SFT | ACL Findings 2025；10.18653/v1/2025.findings-acl.256 | [著] | packing 主流实践再证；"packing 提速"不可当新点。 |
| Beyond Fixed Length: Bucket Pre-training is All You Need | IJCAI 2025；10.24963/ijcai.2025/926 | [著] | bucketing 进入主训练叙事；与 D1 步时收益同族。 |
| Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum | 2024；10.52202/079017-1139 | [著] | 变长课程/数据侧组织。 |
| Training Variable Long Sequences with Data-Centric Parallel | arXiv 2608.07524（2026-08） | [著] | 变长系统层并行；与 D5 相邻。 |
| Near-Policy: … Asynchronous Generation and Selective Packing | arXiv 2605.05940（2026-05） | [著] | 2026 活跃度：选择性 packing 用于蒸馏。 |
| Libra: Taming Attention Workload Skew in Long-Context LLM Training | arXiv 2607.23250 v2（2026-07） | [摘] | ΣL² 负载均衡；目标=DP straggler/管道气泡，**非单卡内存容量** → 与 D1 容量轴正交。 |
| Efficient Sequence Packing without Cross-contamination（SeqPacking） | arXiv 2107.02027（2021） | [摘] | 摘要未见"逐样本权重等价"formalization；delta① 相对其仍可辩。 |
| Varuna | arXiv 2111.04007（2021） | [摘] | 自动 microbatch/内存适配；不针对 invalid-slot slack。 |
| Concertina: Data-Centric Adaptive Pipeline Parallelism | arXiv 2509.21275（2025-09） | [著] | 长上下文流水线；D5 参考面。 |

## 2. 工程侧核查（上轮缺口 → 本轮补齐）

- HuggingFace transformers `docs/source/en/padding_free.md` — 存在（HTTP 200）→ padding-free 官方特性线。
- NVIDIA Megatron-LM `megatron/core/packed_seq_params.py` — 存在（HTTP 200；PackedSeqParams dataclass）→ packed 变长序列官方接口。
- pytorch/torchtune `torchtune/datasets/_packed.py` — 存在（HTTP 200）。
- 结论：packing/padding-free 是三大主栈第一等公民；related-work 必须直面工程实践，不能按"新机制"叙事。

## 3. delta 排序（更新后）

1. **① 单 step 内精确损失等价重分组 formalization**——仍最强可辩项；拦路=SSRN 综述全文未读（未证即未证）。
2. **② 容量轴（同 peak，logical batch ×2–3）**——本轮扫描未见显式同等主张；但 bucket/packing 隐含"省内存→更大有效批"，**差异必须写清**（block/invalid-slot 侧浪费 vs padding 侧）。
3. **③ invalid-block slack 分解（96–98%）**——未见同类；block-drafter 设定特有度最高。
4. 2026 生态：packing/变长训练仍高度活跃（综述 + Near-Policy + Libra + Data-Centric Parallel）→ 论文路线对标压力与相关工作量显著。

## 4. 结论与建议（不改变既有 gate）

- 维持：**工程价值独立成立；论文新颖性未证**。
- 动作：2×2（系统组合）与 d1-006（质量轴）照跑；**综述全文获取 = 论文路线前置条件项**（用户/D1；C0 不阻塞平台线）；D1 的 related-work 差异表可基于本表起草（待 2×2 出数后）。
