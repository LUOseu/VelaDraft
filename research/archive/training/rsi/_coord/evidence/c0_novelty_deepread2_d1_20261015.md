# C0 novelty 深读（增补 #2）— D1 候选机制 · 2026-10-15 04:35 CST

- 性质：第三轮（时间盒 ~40min）。新增：① SSRN 综述 **书目级读取**（全文双机不可达，见 §4）+ Crossref 摘要取得；② 第一轮遗漏的**多模态 packing 线（NaViT / Patch n' Pack）**与 2025–26 新工作（SlimPack / EXACT / Best-fit Packing / Length-Based Training 分析 / GA 等价性讨论）；③ delta① formalization 的**存在性二扫**（arXiv + OpenAlex）。
- 来源：Crossref/doi.org（200）、OpenAlex（200）、arXiv API（200）；Semantic Scholar 仍不可用（404/429）；SSRN 直抓 Mac+服务器均 000（≥4 次）；HF blog 不可达（000，网络侧）。均为检索所得，未编造。

## 1. 综述书目级核对（SSRN《Beyond Concatenation》）

- 著录：DOI `10.2139/ssrn.7343278`；Crossref 200（reference-count 44）；OpenAlex 200（green OA、无 PDF 直链）；正文不可得。
- **摘要（Crossref 已取得）**：自我定位"first survey dedicated to sequence packing for long-context LLM training"（spans pretraining/continued pretraining/SFT/偏好优化）；四类作用 = **Sequence Allocation & Integrity / Context Construction / Workload Scheduling / Shared-Structure Reuse**；明确把 naive concatenation 的失败模式列为 "disrupt document boundaries, mix unrelated contexts, **reweight examples**, and fail to provide meaningful long-range supervision"。
- **44 条参考文献聚类**（清单已全核）：

| 家族 | 代表（书目 #） |
|---|---|
| 长文数据选择/课程 | LongAlign(#1)、LADM(#2)、Prospector(#3)、NExtLong(#17)、EntropyLong(#18)、Lite-Long(#19) |
| packing 机制 | SeqPacking(#20)、Threshold Filtering(#13)、Structured/SPLiCe(#30)、Fewer Truncations/Best-fit(#11)、Seamless Data Packing(#39)、Chunk Flow(#40)、FlexSP(#35)、SlimPack(#25)、Preference Packing(#7)、Optimizing Packing+Shuffling(#6) |
| 负载均衡/并行 | Hydraulis(#22)、WLB-LLM(#36)、Core Attention Disaggregation(#43) |
| 共享结构复用 | Prefix Sharing(#32)、Tree Training(#33)、Schedule-Level Shared-Prefix(#24) |
| 监督分配 | EXACT(#42) |

- 判定：① 综述=**数据组织/长上下文**方向；四 role 均不含"单 step 物理重组的损失语义等价"类条目；② "reweight examples" 以其**失败模式**出现（而非被形式化解决）→ delta① 未被综述直接覆盖（**书目级**结论；全文复核仍列论文前置条件）；③ 书目**未见多模态/视觉 packing**（NaViT 等）→ 综述范围=纯文本。

## 2. 新增对照文献（本轮核实）

| 文献 | 出处 | 读级 | 与 D1 的关系 |
|---|---|---|---|
| Patch n' Pack: NaViT | arXiv 2307.06304（2023） | [摘] | **多模态 packing 规范先例**（训练期序列打包处理任意分辨率/纵横比）——第一轮遗漏，related-work 必引 |
| SlimPack | arXiv 2509.26246（2025-09） | [摘] | 细粒度非对称打包+调度，面向分布式负载均衡；非单卡容量/等价性 |
| EXACT: Effective-Context Exposure Balancing | arXiv 2605.10544（2026） | [摘] | **监督分配重加权**（按有效上下文长度逆频加权）——与"权重保留"相反侧；必须引用并区分 |
| Fewer Truncations / Best-fit Packing | arXiv 2404.10830 · ICML 2024 | [摘] | 长度感知组合优化消除截断（数据完整性线）；无权重等价主张 |
| Rethinking Length-Based Training: Batch Composition and Loss Normalization | arXiv 2609.25890（2026） | [摘] | **最邻近风险项**：证明"批组成/损失归一化选择改变 token 权重与结论"（speech LM；batch-mean vs token-balanced）→ D1 的"权重保留"问题域非新，**可证明的构造**才是 delta |
| Rethinking Gradient Accumulation | Zenodo 10.5281/zenodo.17817933（2025） | [摘] | "GA≈大 batch"叙事的经验反例讨论（等价性话语存在于 GA 轴） |
| Structured Packing (SPLiCe) | AAAI 2025 | [摘] | 检索式相关文档拼装（上下文构建线） |
| Threshold Filtering Packing | NAACL 2025 · arXiv 2408.09327 | [摘] | 相关样本同包（防串扰） |
| Packing + FlashAttention | arXiv 2407.09105（2024） | [摘] | HF SFT trainer 打包工程线（省 padding→提速） |
| Optimizing Packing and Shuffling Strategies | ACL SRW 2026 | [摘] | 打包 vs padding 数据组织权衡研究 |
| TokenDrop + BucketSampler | 2023（SeqPacking 引文） | [著] | 采样层去 padding；非步内重组 |

## 3. delta 复核（二扫结论）

- **① 等价 formalization——仍站（未见证伪）**：arXiv（`"sequence packing"+equivalent` 唯一命中=SeqPacking 自身；`"loss reweighting"+packing`=0 命中）；OpenAlex 多查询无 formalization 命中；综述书目无等价性题名。**但**问题域活跃（EXACT、Length-Based Training、GA 讨论）→ delta 重新表述为"**可证明的逐样本权重保留构造**（fp64 精确 + bf16 带）"，而非"首次指出问题"。
- **② 容量轴——仍站（未见显式同主张）**：两轮扫描未见"同 peak 下 logical batch ×2–3"显式主张；但 NaViT/packing 工程的"省 padding→更大批"叙事普遍 → 差异必须写死（invalid-slot 消除 + 固定 peak 倍数 + "放得下"语义）。
- **③ invalid-slot slack 分解——仍站（无同类）**：未见同型测量（96–98% 残差分解）；EXACT 的"有效上下文分布"最接近但目标不同（监督分配 vs 计算浪费归因）。
- 结论（不改变 gate）：**工程价值独立成立；论文路线="构造①为主 + ②③佐证"**；拥挤度判定维持；SSRN 综述全文仍为**唯一未复核的上位风险**（双机不可达；用户/D1 取全文后 C0 复核即可收口该风险项）。

## 4. 检索留痕（可复跑）

- 成功：Crossref `/works/10.2139/ssrn.7343278`（200；44 refs）；OpenAlex（200×多查询：title.search×8、search×6、cites:W4303647954）；arXiv API（200×多查询：`ti:"Fewer Truncations"`、`all:"SlimPack"`、id_list 2307.06304、等价性查询×3）。
- 失败（环境限制，非文献不存在——OpenAlex/Crossref 双证存在）：SSRN `papers.ssrn.com`（Mac+服务器 000×≥4，04:15–04:30）；S2 API（survey 404 不收录 / 检索 429）；HF blog（000）。
- 更正：seqpacking OpenAlex 引文仅 6 条（覆盖率低，仅作存在性参考）。
