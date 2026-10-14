# 集成与接口 INTEGRATION v0.1.0

## 1. 冻结基线（已核实 2026-10-14 23:00–23:20）

- **BASE_SHA = `7eab1369266010137702de8c714053729cde5834`**（origin/main，= `research/online-systems` 7ada005 + PR #6 merge）。D1/D2/D3/D5 本地 worktree 均已在此 SHA 建立（只读核实）；D4 worktree 亦已建立（`vlm-dflash-d4` @`7eab136`，23:12 起用，C0 23:15 确认；23:41 快照复核）。
- B0（历史核对基线）= `e15256d7c4bd10a490a7ac821eb12a0749a96fdd`（PR #4；docs/plan 等引用它，仅为线索，不代表当前 HEAD）。
- 现有未提交工作：逐一核查六处 worktree（vlm-dflash、veladraft-systems、vlm-dflash-recipe、vlm-dflash-reviewfix、vlm-dflash-dflare、/private/tmp/vlm-dflash-prev）——全部 clean；无需要保护的未提交改动。
- 本地 CPU 验证：`tests/test_online_mode.py` **66/66 passed**；全套 CPU suite **478 collected → 472 passed / 6 skipped / 0 failed**（exit 0，7.2s；torch 2.10 CPU，py3.12 venv，在 veladraft-c0 以 cwd 导入；日志 `research/rsi/_coord/evidence/pytest_full_20261014.log`）。

### 1.1 上轮 review 指出的问题——现状核查（均已看代码/文档，不得凭空删除）

| 问题 | BASE 现状 | 证据 |
|---|---|---|
| 实验环境开关 FLEX_FORCE_MAIN / SKIP_RESUME_CKPT / TRAIN_SHUFFLE | 代码中已不存在；flex 修复改为**无条件代码级 pin**（比环境开关更稳） | `rg` 全仓无命中；`vlm_dflash/dflash/draft_model.py` flex_attention 分支 `setdefault("kernel_options", {"FORCE_USE_FLEX_ATTENTION": True})`（注释说明 torch≥2.10 flex_decoding 在 sm_86 的共享内存/NoValidChoices 问题） |
| 上述开关的历史运行脚本 | `research/system_optimization/server/*.sh` 仍 export 这些变量 → 在 BASE 上是**无效导出**；保留为历史记录，不删除；未来 run 不得依赖 | grep 命中全部位于 .sh/.md 记录文件 |
| cache-on 冷恢复语义 | 已明确：resume 一律冷缓存；功能可恢复；shuffling loader 下不保证与未中断 cached run bit-exact；drift 属 bf16 rounding 类 | `docs/online-training.md` §Host-RAM record cache；`vlm_dflash/target/online.py` 注释 |
| partial-hit / hit-miss 数值描述 | 已精确：hit=bitwise 复用；miss（冷/evict/超容量）=按参考路径重算；records shared not copied；新增 eviction→recompute value-exact 测试 | online.py 注释与测试；REPORT §4.3–4.6（amended gate：loss 相等 + ulp 界 + wdiff，不要求 shuffle 下 sha parity） |

结论：四项无需新的 cleanup 提交；**不删除历史脚本**。若在 a6000 上遇到 flex 崩溃，以 BASE 的代码级 pin 为准排查（reference：research/system_optimization/REPORT.md）。

## 2. 基线分层（不移动 B_shared）

- **B0**：历史工程对照（e15256d），用于量化工程改善，不作为唯一论文基线。
- **B_shared = BASE（7eab136）**：含已验收固定预算 LRU（`train.online_feature_cache_gb`）与必要常规修复；提供两个冻结配置——**cache_off**（默认 0）与 **cache_on(cap)**（记录容量值，如 8/16 GiB）。
- **路线强基线**：每条路线先在自己的 REPORT 登记强基线（配置 + commit + 前后数字）：D2 的 chunked/fused loss；D3 的 warm/tuned FlexAttention（对比对象不能是 Python eager 弱版本）；D4 的多 shard LRU/mmap/预取；D5 的 staged offline。常规新优化不得静默并入 B_shared——另记 commit + 前后结果，由 C0 决定是否进入下一版 B_shared。
- **比较纪律**：D1/D2/D3 主隔离实验固定同一份特征（同 store 或同 extraction recipe）；online 联合实验统一缓存模式；禁止“一组 warm 全量缓存、一组 cold miss 却只报机制收益”。缓存模式必须写入每次 run 的 EXPERIMENTS 记录。

## 3. 接口 rsi-iface-v0.1.0（复用优先；缺口才加可选字段）

已有可复用接口（BASE 核实）：

- sample key：`sample_id`（feature record / online row 均有）；跨模块统一用它做键组件。
- context length：`len(input_ids)`；collate 按 batch 最大长度 padding（`vlm_dflash/dflash/features.py::collate_features`）。
- anchor / block ID：`anchor_positions` [B,N] + `block_size b`；监督映射 `label_indices` [B,N,b]（`trainer_core._compute_labels`）。
- teacher source position：slot k 读 `label_indices[...,k]-1` 处的 hidden（`objectives.TeacherLogitsStream` 约定；flat row 为 block-major）。
- query inverse map（D2 去重）：`unique(label_indices[...,k]-1)` + inverse index 由 D2 派生；如需落 schema，提案新增 **optional** `teacher_position_ids` / `inverse_index`，不得改既有字段语义。
- loss weight：`loss_mask`（float，逐位置）+ recipe knobs（由 `training_plan_digest` 覆盖）。
- MRoPE：`position3` [3,S] 必须保留（三轴）；packing 后的相对位置与 document id 分开管理。
- document/turn 边界：`turn_spans` / `n_turns` / `image_spans`（capture-v2）；v1 与 online 记录缺这些字段——需要时显式处理，不得默认存在。
- feature store：`index.jsonl` + `shard_*.pt`；record keys：`input_ids / loss_mask / hidden / position3 / image_span(s) / sample_id / final_hidden(capture-v2) / turn_spans / n_turns`；`manifest.json` 描述 layer set；digest：`offline_feature_digest` / `online_source_digest` / `training_plan_digest`（v1 版式）。
- 消费接口：`FeatureStore.get(index)` / `FeatureDataset.__getitem__` / `collate_features`（record → tensor 的唯一入口）。
- 缓存/字节/命中接口：`train.online_feature_cache_gb`（GiB，0=off）；stats `{hits, misses, stores, evictions, retained/peak bytes}`；`_record_nbytes` 计字节；hit=bitwise；miss=重算；生命周期=OnlineFeatureSource 内、train/overfit-sanity/calibration 共享、不落盘、resume 冷启动。
- 成本计数：extraction 分阶段 `prepare_seconds/forward_seconds`（`_sync_target` 边界）；run summary 的 `feature_source` 块（缓存统计、forward 次数）。

变更流程：D 角色写提案（`research/rsi/_coord/proposals/<role>-NNN.md` 或 `handoff.md`）→ C0 接收并在 C0 分支集成 → 接口版本 +0.x → REGISTRY 通知受影响角色。默认只加可选字段；先复用已有字段。

## 4. 波次放行

**Wave 1（已放行；预算见 RESOURCE_SCHEDULE）**：D4 收尾/容量探针；D1 两轴 probe；D2 N/U 与 teacher 成本；D3 attention/mask profile；D5 静态流水线/trace 可行性。

**Wave 2 门槛（需 C0 复核证据）**：

- D1：真实混合 workload 整步 ≥15% 提升，或峰值显存 ≥25% 下降且扩大可训练 batch/context；质量一致。
- D2：真实 ρ≥2 且 teacher/head/loss 占比较高；整步 ≥15% 或显存 ≥25%；parity 全覆盖（含非法组合跳过记录）。
- D3：attention+mask 占整步 ≥20% 才做完整 kernel；否则交付否定/边界证据。
- D4：相对多 shard LRU/mmap/预取的强基线仍有额外、可解释收益才升级为候选贡献；否则作为平台基础设施。
- D5：存在静态强基线且整流程 ≥15% 时间或 GPU-hours 改善、由多模态/anchor 成本解释；G=2 起步（按实际可用卡数，不假设 4 卡）。

优先给 D1/D2 研究资源，D4 提供实用 backend；按真实证据调整排序。

## 5. 组合与复核（只做相关组合）

- D4×D1 2×2：{cache_off, cache_on} × {bucket_off, bucket_on}，同一逻辑 batch 与 anchor 表。
- D1×D2：仅在软目标覆盖重合（同一监督位置集合）时才组合。
- D4×D5：重测 producer 负载（缓存改变 capture/extraction 到达曲线）。
- D1×D3：重测形状分布（bucketing 改变 attention 形状直方图）。
- 报告正/负交互；不得乘 speedup；重叠阶段时间不能相加；修改逻辑 batch/anchor/目标权重的候选要退回纠正，不得混作“等价优化”。

## 6. 结论与合并

- 每方向 GO/HOLD/STOP/BLOCKED + 证据（run_id、原始路径）+ 相对强基线判断 + 区分“平台工程收益”与“可能支撑论文贡献”。
- GO：最小可审查分支/patch + 验证命令 + merge-readiness 说明；**不自动 push/merge main**，由用户 review 后决定。
- STOP：保留证据与再启动条件。

## 7. 晋级复核 checklist（wave-2 放行前 C0 逐项取证）

1. 收益：相对**该路线自己登记的强基线**的配对结果（同机 A/B ≥3 交替或一个足够长的稳定窗口），原始日志可复算；microbenchmark 数字不得当训练结论。
2. 语义：逻辑 batch / anchor 表 / loss 权重 / LR 计划未被改动；若确实重排或重组，须证明保留各原 microbatch 的归一化权重（如 decay loss 的分母 Z），并附 oracle parity（loss/分项 loss/梯度/一次 update）。
3. 数值：parity 声明与四层证据一致（数学等价 / 容差内 / 质量一致 / bit-exact 分清）；bf16 容差为预设值。
4. 特征与缓存：隔离实验固定同一份特征；online 实验缓存模式固定或 {off,on} 成对；partial-hit 描述精确（hit 行 vs 重算行分开）。
5. 计量：train wall、pipeline wall、GPU-hours（含等待）、显存/RAM/存储峰值、有效监督/block 吞吐齐全；计时边界与 sync 位置明确；首次加载与 warm 分开；重叠阶段不简单相加、各层收益不乘倍率。
6. 代码侵入：改动最小并落在角色目录/新模块；共享入口改动须有 C0 接收的提案与测试；不破坏既有 online suite（BASE 66/66）。
7. 声明审计：抽查 2–3 个关键数字与原始日志/JSON 一致；限制与反证条件写明；负结果保留。
8. 复现：EXPERIMENTS 字段齐全、依赖 SHA 固定、复现命令可执行；外部引用标明“引用、未独立复核”。
