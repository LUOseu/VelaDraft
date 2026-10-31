# VelaDraft：面向 ML Systems 的训练执行方案

## Problem Anchor
- Bottom-line problem: reduce the total cost of training diffusion block drafters within VelaDraft while preserving the specified learning objective and validating attained quality.
- Must-solve bottleneck: anchor expansion repeats frozen teacher/output-head work and materializes vocabulary-sized intermediates; existing feature caching does not remove this scoring cost.
- Non-goals: change serving semantics, claim a new diffusion architecture, or equate microbenchmark speed with end-to-end savings.
- Constraints: existing main 9404b93; shared four-A6000 host; preserve and restore user tasks if paused; opt-in behavior; CPU and CUDA correctness before long runs.
- Success condition: reproducible objective/gradient checks and lower measured training resource cost, followed by matched-quality multi-seed evaluation before paper claims.

## 定位与本轮决定

目标 venue 是 ML Systems / 系统会议。论文应回答：**在相同训练目标和质量要求下，如何依据 block 展开、监督复用、词表宽度与显存约束选择执行方式，降低总训练成本？**

本轮完成三个可对照的执行选项：有效 slot 的 checkpoint CE、冻结头的 hidden-gradient 预计算，以及 teacher 投影的 chunk 内去重。所有选项默认关闭；保持完整词表、原 anchor 表和原 loss 权重。

这些组件本身不是足够的新颖性：chunked/fused CE、冻结头蒸馏和一般去重都有已有工作。本轮独立研究审查给出 **REVISE，6.9/10**。因此不把当前 patch 宣称为投稿完成的贡献，也不通过命名强化新颖性。

## 从实测得到的设计约束

完整数据由 [summary.md](summary.md) 与原始 JSON 提供。共享 GPU 上的速度区间很宽，不能用于论文主表。随后空闲 GPU 2 的受监控复测在限定 padded 合成形状上测到 1.396×（成对区间 1.350–1.411×）；仍缺真实数据、组件归因与收敛验证。

- teacher 投影去重在高重叠样本上减少投影行数，但第一轮完整 update 未建立稳定加速。保留为可否证的实验组件，默认关闭。
- checkpoint CE 明确降低显存，但反向重算可能抵消计算节省。
- 冻结头可把 backward 所需的状态从 vocabulary 维度移到 hidden 维度；CPU 和 BF16 的 loss / 全参数梯度检查通过，但低精度舍入不等同于逐位一致或长期质量一致。
- 长短样本混合时，跳过 padding 和 clean-anchor 的机会更大。必须报告无 padding 的负例，不能把人工构造的 padding 收益外推到真实 workload。

结论是继续做**受训练目标与资源约束的执行选择**，而非默认把所有优化叠加。自动选择策略尚未实现。

## 有潜力的主贡献（待验证）

用一个明确的监督依赖表示区分三类状态：

1. `(sample, anchor, slot)` 对应的 student 状态不能因预测同一 token 而合并。
2. `(sample, predecessor position)` 对应的冻结 teacher 状态可以复用，但 full-distribution consumer 的展开/归一化代价需要计算在内。
3. 冻结输出头下的 CE / KL 与 TV / hybrid / confidence 目标需要不同的反向状态；纯 KL 的已有充分统计量方法不能直接替代 TV 等目标。

在该表示上建立有显存约束的简单执行策略，输出 dense、compact、checkpoint、precompute 或 consumer-side teacher reuse 的选择及原因。先做少量可解释候选，不引入通用编译器或学习型调度器。若实际数据表明某一路径不值得保留，删除该候选。

当前尚未实现的核心候选是：让 TV/hybrid consumer 直接消费 unique teacher rows 与 inverse map，避免先生成完整重复 teacher logits 后再计算 loss。只有 profiler 证明该重复分布计算占比足够大，才投入实现。对同样的机制继续查新。

## reviewer 会问什么，以及怎样回答

| 质疑 | 必需证据 |
|---|---|
| “就是 Liger/CCE 加 caching” | 同目标下比较现有 fused loss；明确已有组件与新调度/数据流机制的差异；消融只留下会产生独立收益的部分 |
| “只优化了一个 kernel” | forward、backward、FP32 optimizer、数据读取、capture/regen、eval 分阶段计费；报告完整 update 与总 GPU-hours |
| “靠缩小 batch / 少训练 / 牺牲质量” | 相同样本、anchor、优化器、更新次数的 execution 对照，以及等质量 time-to-threshold；单列资源限制下可行 batch 的收益 |
| “刻意制造 padding 才有收益” | 从实际 feature store 采集长度与 anchor 利用率分布；按分位数加权报告；无 padding 与低重叠负例必须保留 |
| “BF16 改了训练轨迹” | 同代码重复与数学等价重排的数值底噪；FP64/FP32 全梯度；至少三种子、独立验证集、质量阈值与 censored failures |
| “已有 block parallelism” | 区分长上下文 attention/通信与监督 scoring；有重叠时直接比较，不能声称首次利用 block 独立性 |

## 最小实验路线与停止条件

**E0：真实 workload / strong baseline。** 从现有 Qwen2.5-VL-3B、Qwen3-VL-8B feature stores 读取确定的样本与 anchor 表，记录有效 slot 比例、teacher R/U、shape 和各阶段成本。固定 main 9404b93 为历史对照，同时 pin 当前 SpecForge 和已有 VelaDraft compile/compaction 开关。若目标子系统不足以解释潜在总收益，缩小或更换论文主机制。

**E1：组件与选择策略。** 至少分开 dense、仅 row compaction、checkpoint CE、precomputed CE、适配的 Liger/CCE、teacher local reuse；KL 场景增加已有充分统计量/fused-KL baseline，TV/hybrid 不以 top-k KL 替代目标。校准集选择 chunk / 路由阈值后冻结，在不同真实 shape 分层和另一 target 上评估。独占 GPU 做随机交错成对测量，保存所有样本；报告 memory/time frontier 和回退率。若收益不能超过强 baseline 和测量底噪，保留平台功能，不作为主贡献。

**E2：等质量成本。** E1 成立后才扩展到至少两 target、多个相关 drafter / recipe、三种子。预注册验证 AL / 接受率及任务质量阈值；测试集不参与调参。报告达到阈值的训练 GPU-hours、未达到者、峰值显存、总 pipeline 成本以及一份 capture 被多个方法复用时的摊销。AL 不等于推理速度；native serving 和任务质量分别测量。

费用按实际占用的 GPU 数量与时间累积，不把共享 GPU 的 wall time直接称为独占 GPU 成本。三张用户卡上的长任务没有因本轮实验暂停；正式独占窗口需要单独安排，避免暂停正在计时的评测而污染其结果。

## 相关工作与声明边界

独立审查的完整文献与差异表见 [round-1-review.md](refine-logs/round-1-review.md)。关键比较对象包括 [SpecForge](https://arxiv.org/abs/2603.18567)、[Liger](https://arxiv.org/abs/2410.10989)、[CCE](https://github.com/apple/ml-cross-entropy)、[Block Parallelism](https://arxiv.org/abs/2609.19242) 和 [fused chunked KL](https://arxiv.org/abs/2608.03796)。[KL cache technical note](https://jonathanc.net/blog/kl-cache-trick) 属于作者技术说明，不能标为已审稿论文。

当前交付是可复现的平台改造与首轮否证实验。没有宣称新的 diffusion 模型、普遍加速、端到端训练成本已降低、长期质量无损或投稿就绪。
