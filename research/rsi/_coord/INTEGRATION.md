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

### 4.1 Wave-1 台账（滚动；更新 2026-10-15 02:05）

| 路线 | wave-1 证据 | 门槛核验 | 当前判定 |
|---|---|---|---|
| D1 | 浪费审计 + oracle（C0 复跑 ✓）；store 全 ✓（W1 191/W2 144/W3 192/MIX 335）；**wave-2 完成：6/6 rc=0，grid W3 1.129/1.136/1.141、MIX 1.154/1.137/1.107（均值 13.5%/13.3%，win 580/600；peak −57%；parity b0/50/100 ✓；0.67/1.5 GPU-h）**；round-2 R3：W3 k4 +19.0% / MIX k4 +24.7%（8 批 cell，改口径）；EXPERIMENTS 24 条 | 整步门 ≥15%：wave-2 13.5%/13.3%（未达）；显存门 ≥25% ✓（−57%）→ OR 通过 | **GO（修订口径，`d1-002-c0`）**：capacity demo（§2 单列）收尾 → 关闭 L-W2-D1 → 集成 d1-001 最小集 + D4×D1 2×2 |
| D2 | rho=6.83；**12 项 CPU 测试（C0 复跑 12/12 ✓）**；pin 2×2 fwd -6.8%（loss 四腿 bitwise）；**dfly recipe 2×2 完成：fwd −12.9ms、bwd −19.5ms（12 步 loss 四腿逐值相同，C0 直读 ✓）**；phase dedup4 归因完成（editable install 遮蔽=空对照；未钉卡 event 伪影已修） | ρ≥2 ✓；**收益 −3.0%（Jetspec）/ −5.6%（DFly recipe）≪15%；显存无降** | **STOP（D2 自判，C0 采纳；d2-001-c0）**→DEFERRED（lane-local opt-in；再开条件=REPORT §8） |
| D3 | trace + Amdahl：attention+mask=21.16ms/595.9ms=**3.55%**（cap +3.7% 步 / +1.4% online），**C0 核数 ✓** | ≥20% ✗（远未达） | **STOP**（kernel 轨；re-open 条件已登记） |
| D4 | W1 五腿 counters 逐值一致（**C0 5/5 ✓**）；W2 CPU 压力轨迹（零越界）；**W2 降级 B 收口（F1–F4 PASS；两腿 counters 645/129/129/0/0 相同；adjustments=0；lane `a8e762c`）** | 无提速声称（安全阀）；d4-001 集成 `1f8963e`（sha1 4 文件再核一致；13+4+70/70 ✓） | **merge-ready（平台工程级，`d4-003-c0`）**：静态 RAM 场景 / opt-in 默认关；压力响应未验证（环境限制）；不自动 merge main |
| D5 | trace 分解 + sim v3/v4；regen 真测 1.66×/2.33×；静态分裂 sim 1.70×；d5-001 步 2 + 两卡 split 接线（lane 85/85 C0 复跑 ✓）；**G2 链：Tier0 canon ✓、staged_b8 ✓（762 步/kept 1012）、sync_b8 ✓（762 步 final_loss=4.03125；visual ✓）、sync_b32 ✓（768 步 final_loss=4.0625）；s3_b8/s3_b32 ✓（1716s/1074s，vs sync_b8 −21% / vs sync_b32 −12.6%；s3_b32 store parity 待复验）** | cost-aware ✗（≪15%）；静态 split 实测出数（vs sync_b8 −21% / vs staged_b8 −25%） | HOLD（split=wave-2 条件项；S3 收尾进行中） |

未决：D1——wave-2 GO+capacity ✓、集成 `4e1674e` ✓、e2e smoke ✓、**R4 负分支关闭**（`d1-005-c0`）→ **D4×D1 2×2 预注册/放行**（`d4xd1-001-c0`，L-W3-D4xD1）；**初跑 03:02–03:17 因 parity 捕获缺陷整体作废（自检停链；全披露 addendum §5.5）→ 裁定 `d4xd1-002-c0`（update 轴读法 + 重启 GO；8 腿重启集 = 唯一正式证据）**；**held-out 质量 `d1-006`/L-W3-D1Q 已接收（≤2.4，staged）**；D4 收口完成（merge-ready；无待办）；D5——Tier0/staged_b8/sync_b8/sync_b32/s3_b8/s3_b32 ✓（s3_b32 store parity 待复验）；Tier C 待 C0；预算实耗（详见 §8）：D1 0.92/1.5 + wave-2 0.90（单列）+ R4 0.108/0.2、D2 0.5/1.0、D3 0.22/1.0、D4 0.37/0.5、D5 进行中（staged_b8 ≈1.27 + sync_b8 ≈1.21 + sync_b32 ≈0.68 + s3_b8 0.95 + s3_b32 0.59 分配口径，待 harvest 复核）。 **Δ04:15：D4×D1 2×2 收官（`d4xd1-003-c0`——W3 F1-as-registered FAIL/组合数值相容性声明不成立·正确性阻断；MIX 全过；交互 bit-0；`d4xd1-004-c0` 放行）；D5 staged_b32 待开跑（L-W3-D5B 已批）。**

## 5. 组合与复核（只做相关组合）

- D4×D1 2×2：{cache_off, cache_on} × {bucket_off, bucket_on}，同一逻辑 batch 与 anchor 表。**预注册/租约已发布：`d4xd1-001-c0`（L-W3-D4xD1，≤1.5 GPU-h，8 legs online）。** **收官 `d4xd1-003-c0`：W3 F1-as-registered FAIL（组合数值相容性声明不成立·正确性阻断）/ MIX cell 全过；交互 = 0（bit）；隔离对照 `d4xd1-004-c0`（L-W3-D4xD1B）。**
- D1×D2：仅在软目标覆盖重合（同一监督位置集合）时才组合。
- D4×D5：重测 producer 负载（缓存改变 capture/extraction 到达曲线）。
- D1×D3：重测形状分布（bucketing 改变 attention 形状直方图）。
- 报告正/负交互；不得乘 speedup；重叠阶段时间不能相加；修改逻辑 batch/anchor/目标权重的候选要退回纠正，不得混作“等价优化”。

## 6. 结论与合并

- 每方向 GO/HOLD/STOP/BLOCKED + 证据（run_id、原始路径）+ 相对强基线判断 + 区分“平台工程收益”与“可能支撑论文贡献”。
- GO：最小可审查分支/patch + 验证命令 + merge-readiness 说明；**不自动 push/merge main**，由用户 review 后决定。
- STOP：保留证据与再启动条件。

### 收口结论快照（2026-10-15 05:50，C0；滚动更新）

**方向判定（区分平台工程 / 论文口径）**

- **D1｜GO（工程口径）＋ 论文两轴 BLOCKED**。工程：wave-1/2 步时 13.5%/13.3%（grid 配对均值 1.135/1.133）、580/600 配对胜、peak −57%、容量 b24 OOM→14.41 GiB 可跑 5/5、parity b0/50/100 ✓（C0 重算）。论文：①d1-006 质量轴 **FAIL**（两 seed×两指标出带；band_AL=1.815e-3 / band_acc=2.792e-4；收据 `evidence/c0_d1q_receipt_verdict_20261015.md`）→ 长训练 GO 阻断项成立、「质量中性」不成立；②在线 `cache×regroup` 数值阻断（`d4xd1-003/005`）维持暂禁；③R4：k4 残余 4.7–5.2%/6.1–6.4%，sub1 边界跑未过门 → R4 收口为 boundary result（full removal 真实但被额外 microbatch 固定成本吃掉；packed-block kernel dormant，D3 已关）。
- **D2｜STOP**：pin 2×2 最好 −6.8%、常见 −3.0%/−5.6%（<15% 筛选门）；−5.5% fwd 未复现（+0.5%）；质量全平（loss 四腿 bitwise、CPU oracle 12/12）。再开条件见 D2 REPORT §8。
- **D3｜STOP（CLOSED）**：attention+mask = 3.55% 步时（≪20% 门）；k-split 自身 +0.8%。再开条件：W2/long-answer、B>8、S≈2.5–4k。
- **D4｜merge-ready（平台工程）**：`1f8963e` 集成（sha1 对账 + 复跑）；W2 F1–F4 PASS；opt-in 默认关；0.37/0.5 GPU-h。无独立论文声称。
- **D5｜HOLD（staged_b32 已批未跑）**：S3 静态分裂 b8 −25.3%（剔除 overfit 记账偏差后 −32.9%）/ b32 −12.6% vs sync（store parity partial 待 staged_b32 关闭）；成本感知线 STOP（sim 增量 <15%，真实 sync/staged=0.947 证伪 sim 0.86）；pinned_b8=HOLD、Tier C 暂缓。

**最小可审查集成集（GO 候选）**

- 分支 `research/rsi-c0`（最新见 git log；代码面 vs `7eab136` = 3 提交：`1f8963e` D4 adaptive budget / `4e1674e` D1 regroup（opt-in、默认 byte-identical）/ `192554a` 测试补充；10 文件 +1603/−24）。
- 验证：本地套件 **515 collected / 509 passed / 6 skipped / 0 failed**（7.19s；`evidence/pytest_rsi_c0_head_20261015.{log,xml}`）；D4 最小集 sha1 对账（§5）；D1 opt-in 默认关闭的 byte-identical 由 `4e1674e` 测试覆盖。
- merge-readiness：可合并（**不自动 merge main**）；合并前建议复跑上述套件 + 抽查 d1/d4 最小集文件。

**未决（他人执行项）**

- ~~D1 harvest 回填~~ **已收讫（`1b0328e` 等）并逐项对账一致（05:52；收据 `evidence/c0_d1q_harvest_receipt_20261015.md`；budget 1.1536/2.4 关闭）**；
- D5 `staged_b32`（L-W3-D5B ≤0.75 GPU-h）开工 → C0 复核（`research/rsi/d5/runner/split_verify.py`）；
- 转发文本：`_coord/messages/20261015_d1q_recovery.md`（D1 终裁通知 + D5 提醒）。

## 7. 晋级复核 checklist（wave-2 放行前 C0 逐项取证）

1. 收益：相对**该路线自己登记的强基线**的配对结果（同机 A/B ≥3 交替或一个足够长的稳定窗口），原始日志可复算；microbenchmark 数字不得当训练结论。
2. 语义：逻辑 batch / anchor 表 / loss 权重 / LR 计划未被改动；若确实重排或重组，须证明保留各原 microbatch 的归一化权重（如 decay loss 的分母 Z），并附 oracle parity（loss/分项 loss/梯度/一次 update）。
3. 数值：parity 声明与四层证据一致（数学等价 / 容差内 / 质量一致 / bit-exact 分清）；bf16 容差为预设值。
4. 特征与缓存：隔离实验固定同一份特征；online 实验缓存模式固定或 {off,on} 成对；partial-hit 描述精确（hit 行 vs 重算行分开）。
5. 计量：train wall、pipeline wall、GPU-hours（含等待）、显存/RAM/存储峰值、有效监督/block 吞吐齐全；计时边界与 sync 位置明确；首次加载与 warm 分开；重叠阶段不简单相加、各层收益不乘倍率。
6. 代码侵入：改动最小并落在角色目录/新模块；共享入口改动须有 C0 接收的提案与测试；不破坏既有 online suite（BASE 66/66）。
7. 声明审计：抽查 2–3 个关键数字与原始日志/JSON 一致；限制与反证条件写明；负结果保留。
8. 复现：EXPERIMENTS 字段齐全、依赖 SHA 固定、复现命令可执行；外部引用标明“引用、未独立复核”。

## 8. Wave-1 收口（滚动草稿 · 更新 2026-10-15 03:36）

> wave-1 尚未关闭（D1 wave-2 已完成并关闭租约、D4×D1 2×2 重启中；D5 S3 追赶链 s3_b8/s3_b32 已出数、store parity 待复验；D4 已收口）。本节是 C0 的滚动台账：给每条**当前判定 / 证据路径 / 解除条件**；最终 GO/HOLD/STOP/BLOCKED 在关闭时冻结。平台工程价值与论文贡献潜力分列，不互相折算。

| 路线 | 当前判定 | 关键证据 | 平台 / 论文 | 解除或关闭条件 |
|---|---|---|---|---|
| D1 | **GO（修订口径；`d1-002-c0`+`d1-003-c0`）— wave-2 全收，L-W2-D1 已关闭** | wave-2 配对（C0 重算）：grid W3 1.129/1.136/1.141、MIX 1.154/1.137/1.107（均值 13.5%/13.3%；win 580/600；peak −57%；parity b0/50/100 ✓）；**capacity demo ✓ C0 逐值重算**（W3 b16 43.59→14.32 GiB；b24 canonical OOM 5/5 → cost 14.41 GiB 5/5；MIX b16 27.74→9.83 GiB）；用量 ≈0.90/1.5 GPU-h；round-2 的 +19.0%/+24.7% 为 8 批 cell（不再作主数字） | 平台：布局重排引擎（跨 seed 稳定复现；容量杠杆=同卡逻辑 batch ×2–3）；论文：候选保持（+13–14% 步 + −57% 峰值），**novelty 深读 #1/#2：delta①/②/③ 二扫未见证伪（表述=可证明权重保留构造为主）；综述全文=论文前置条件**（`evidence/c0_novelty_deepread2_d1_20261015.md`） | **已完成**：L-W2-D1 关闭；集成落地 `4e1674e`（opt-in、默认 byte-identical、509/6 全绿）；R4 §12 评估已收（k4 残余 4.7–5.2%/6.1–6.4%）；cross-check `3e29244` = PASS（建议 a 已采纳 `192554a`）；**L-W3-D1R4 已关闭：负分支**（`d1-005-c0`；02:33：sub1 W3 1.1296 / MIX 1.0199 < 1.135——速度维度封顶，D3 kernel 不重开，0.108/0.2 GPU-h）；**集成 e2e smoke PASS**（02:27–02:30 GPU0：产品路径 regroup 20 步 rc=0，peak 25.34→9.98 GB；`evidence/c0_e2e_regroup_smoke_20261015.md`）；**D4×D1 2×2 预注册/租约已发布并放行**（`d4xd1-001-c0`，L-W3-D4xD1）；初跑 03:02–03:17 因 parity 捕获缺陷整体作废（自检停链、全披露 addendum §5.5）；**裁定 `d4xd1-002-c0`（重启 GO；重启集 = 唯一正式证据）**→ **2×2 收官（`d4xd1-003-c0`）：W3 cell F1-as-registered FAIL（1.2475e-2@11，4 步超门）→ 组合数值相容性声明不成立·正确性阻断；MIX cell 全过；交互 bit-0；隔离对照 `d4xd1-004-c0` 放行（L-W3-D4xD1B）**；D1 harvest 回填已收讫（`e7ea0cf`+`9990a0f`；与 C0 重算逐项一致）；**d4xd1-004 已收口（`d4xd1-005-c0`：F1B FAIL→阻断加深）**；`d1-006`（L-W3-D1Q）待跑 |
| D2 | **STOP（自判+采纳；机制 DEFERRED；已全收口）** | `research/rsi/d2/` @ lane `bfb3635`（+`d6e3c9a` 记录提交；REPORT/EXPERIMENTS/handoff 全量含失败与归因段；“12/12”笔误已更正；预算 0.5/1.0 GPU-h 已记）；pin 2×2（−6.8%，loss bitwise）+ dfly recipe 2×2（−12.9/−19.5ms，12 步 loss 逐值相同） | 论文：否（−3.0%/−5.6% ≪15%；Amdahl 上限 ~5–6%）；平台：opt-in 旋钮（lane 存档，不集成） | **已关闭**；再开=REPORT §8 条件 |
| D3 | STOP（已接受） | REPORT `b88cdf2`：attention+mask=3.55%（C0 核数 ✓）；/tmp 已清 | 论文：否 | 再开条件：W2 长答 ~15% 外推 / b16 / S≥2.5–4k |
| D4 | **merge-ready（平台工程级；`d4-003-c0`；不动 main）** | W1 五腿 counter-exact（C0 5/5 ✓）；**d4-001 集成 `1f8963e`（sha1 核对 ✓；13+4+70/70 ✓）**；**W2-B 收口**（F1–F4 PASS；两腿 counters 相同、adjustments=0；lane `a8e762c`；0.062 GPU-h） | 平台：adaptive 安全阀（opt-in、默认关、静态 RAM 场景）；论文：低（无提速声称；压力响应未验证） | 无待办；推进 main 由消费方/用户决定（按 INTEGRATION §7 checklist，不自动 merge）；**D4×D1 2×2 中以只读接口参与（线不重开；`d4xd1-001-c0`）——已收官：floor probe bit-exact 0.0（`d4xd1-003-c0`）** |
| D5 | HOLD（条件项；S3 出数） | REPORT `85cbf8b` + sim v4 `23875e8` + d5-001 步 2 + lane 85/85（C0 ✓）；**Tier0 ✓；staged_b8 ✓（762 步，kept 1012，≈1.27 GPU-h 分配口径）；sync_b8 ✓（wall 2174.5s，vs staged −5.3%，远低于 sim −14%；762 步 final_loss=4.03125）；sync_b32 ✓（01:53:29→02:13:58；768 步 final_loss=4.0625）；s3_b8 ✓（1716s；parity 全 ✓；vs sync_b8 −21%、vs staged_b8 −25%）；s3_b32 ✓（1074s；regen parity ✓、store 待复验；vs sync_b32 −12.6%）**；S3 修复已合入（`2cc57c6`/`daf6b1d`） | 平台：runner 语义级可驱动真 trainer（本 build 抽取与训练串行）；论文：低（H2 负） | **余项**：staged_b32 store-parity 复验 + §10 汇总（GPU-h 时间戳重算）+ 三态 parity 收口；Tier C 倾向暂缓（S3 出数后定） |

跨路线观察：D5 v4 的 keep-oracle 变体需要“生成前已知 keep”的预测器——若 D1 双轴组织能提供此类信号，仅作为 **D1×D5 条件组合的讨论项**（未批准；须双方有共同监督位置才立项）。

**共同基线与资产**
- BASE `7eab136`（复核无漂移）；协议/接口 v0.1.0；B0=`e15256d`。
- 共享资产：D1 W1 store（`e044f488…`）+ W2 store（`5980f77b…`）+ **W3 store（`95ef12e4…`，192 行多轮）+ MIX store（335 行 = W1+W2 交织）**；D5 CPU LivePipeline（`b60d1eb`，lane）；D4 fixed-LRU（已随 PR #6 合入）。
- 集成分支：**`research/rsi-c0` 本身**；d4-001 集成 commit `1f8963e`（13/13 + 70/70）与 **d1-001 集成 commit `4e1674e`**（opt-in `train.regroup`；默认路径 byte-identical；全套 **509 pass/6 skip/0 fail**）——均不自动 merge main。

**预算实耗（GPU-h）**：D1 wave-1 **0.92/1.5**；**wave-2 单列 0.90/1.5（关闭：in-run 0.73 + capacity demo ≈0.17；GPU3 单卡）**；**R4 单列 0.108/0.2（关闭 02:33，负分支）**；D2 0.5/1.0（已盘点）；D3 0.22/1.0；D4 **0.37/0.5**（W1 0.31 + W2-B 0.062）；D5：staged_b8 ≈1.27 + sync_b8 ≈1.21 + sync_b32 ≈0.68 + s3_b8 0.95 + s3_b32 0.59（分配 2 卡口径：2×wall；含 0.16 事故计入；待 harvest 复核）。**D4×D1 2×2：L-W3-D4xD1 收官——初跑 0.26（void）+ 重启链 0.472 + 修复 smoke ≈0.03 ≈ 0.76/1.5（余额 ≈0.74）；新租约 L-W3-D4xD1B ≤0.10（`d4xd1-004-c0`）；L-W3-D1Q 接收（≤2.4，staged，待跑）**。

**事故台账**：见 RESOURCE_SCHEDULE §3 / REGISTRY §6（#1–#5，均含时间线、处置与租约修正）。

**最终交付格式**（关闭冻结时按此输出，每方向）：判定 + 证据 run_id/路径 + 相对强基线配对结论 + 平台/论文分列；GO 附最小 patch/分支 + 验证命令 + merge-readiness；STOP 附再启动条件。


### 终裁 `d4xd1-005-c0`（L-W3-D4xD1B 隔离臂；2026-10-15 04:24，C0）

- 腿 rc=0 @04:22:43（101 步；GPU3；≈0.081/0.10 GPU-h 关闭）。C0 独立重算 ≡ D1 `d4xd1_004_f1b.json`（未提交 @读取时）：**F1B FAIL（2.2727e-2@11；17/100 超门；中位 4.32e-3；门 7.5e-3）**。
- **判决（预注册 FAIL 分支）**：在线分组超带、阻断加深——`cache×regroup` 维持暂禁；组合声明不成立不变；机制排查须新提案（含同路径伴随对照）。
- 描述性：损失路径单因子翻转亦超门（1.4080e-2@15）；identity/anchors 通过（104 calls 三腿一致；6.13e-8）。
- 编排：D1 harvest 回填（2×2+004）→ `d1-006`（L-W3-D1Q ≤2.4 staged）现可启动；D5 `L-W3-D5B`（staged_b32 ≤0.75）已批未跑。
- 全文 `proposals/d4xd1-005-c0.md`；证据 `evidence/c0_d4xd1_004_recompute_20261015.json`。


### 收讫 D1 harvest 回填 + 整合分支测试刷新（2026-10-15 04:32，C0）

- 收讫 `e7ea0cf`（2×2 harvest，04:16:30：EXPERIMENTS 54 + REPORT §13.1–13.7）与 `9990a0f`（004 harvest，04:26:49：EXPERIMENTS 55 + REPORT §13.8 + evidence/run-copy/构建脚本）——两者与 C0 重算（`d4xd1-003-c0`/`d4xd1-005-c0`）逐项一致，无整改；F1B FAIL 如实记录。
- C0 复算 D1 grid-ulp 披露（非判决项）：fal 5.00/1.00/2.00 ulp / 17>1ulp @12；tru 2.79/0.42/0.93 / 8 —— 与 REPORT §13.8 一致。
- 预算对账：D1 计算口径 0.0747 GPU-h vs C0 窗口口径 0.081（含 staging 首跑）—— 均 ≤0.10；租约按 0.081 关闭。
- 引用差（无整改）：`9990a0f` 与 005（04:24 发布）平行提交、未引用 005；方向一致（阻断加深）。
- 整合分支测试刷新 @HEAD：515 collected / **509 passed / 6 skipped / 0 failed**（exit 0，7.19s，py3.12 + torch 2.10 CPU）；证据 `evidence/pytest_rsi_c0_head_20261015.{log,xml}`。代码面（vs `7eab136`，除 `_coord`）= 3 提交（`1f8963e` D4 adaptive budget / `4e1674e` D1 regroup / `192554a` 测试补充），10 文件 +1603/−24 —— GO 候选最小可审查集。
- 排期（04:32 实测）：`d1-006`（L-W3-D1Q）、D5 `staged_b32`（L-W3-D5B）均**未启动**（4 卡全空；/home 20G）；D5 复核入口就位：`research/rsi/d5/runner/split_verify.py`（执行按 `d5-002-c0`③②，需 `runner` 包路径）。


### C0 ack `96709e6`（d1-006 执行 addendum；L-W3-D1Q 已开跑）（2026-10-15 04:33，C0）

- **ack（对齐租约；无整改）**：band-first（Stage-1 canonical×{42,43} 建带 → Stage-2 regroup×{42,43} 配对）、N=200、`eval_dflash.py` 64-prompt/固定 val、GPU3、预算 ≤2.4 —— 与接收条款一致。披露接受：①val 16→64 为**确定性扩展**（train 192 行不动、不重跑 prepare〔rng.sample 会重抽训练行〕；规则+src_index+sha256 落 `val64_record.json`；四腿共享同一 `val.jsonl`）；②`max_new_tokens=128`（shipped 默认，非 config 的 32；四腿一致，配对比较不受影响）；③offline 特征（与 2×2/004 online 臂不同类，不与 F1 门混用）；④smoke-gate 失败即停不烧租约——按此执行。
- **执行实况（04:31–04:33）**：`d1q_val64` 构建完成（val 64 行 / 48 新图 / 记录 schema `d1q_val64_v1`）；`d1q_smoke` 04:32:19 开跑（guard 通过；`train.regroup enabled: mode=grid sub_batch=2 fp32_weights=True` ✓；3 步；visual check passed；**exit=0 @04:33:06**）→ Stage-1 canonical×{42,43} 排队（GPU3）。
- 待办：四腿（train+eval）落地后 C0 独立重算（band/配对判决；raw=`evidence/d1q_runs/<leg>/{train_summary.json,train_log.jsonl,eval/*.json}`）。D5 `staged_b32` 仍未启动（04:33）。

### d1-006 终裁（L-W3-D1Q，质量轴）：FAIL（2026-10-15 05:47，C0）

- 四腿 c42/c43/r42/r43 全收敛（200 步，loss_decreased=true，无 NaN）+ token_exact=True；band_AL=0.0018152、band_acc=0.00027917（canonical 42/43 ckpt_final）。
- 配对（逐 seed）：s42 dAL 3.26e-3（1.80×）/dacc 4.72e-4（1.69×）；s43 dAL 8.81e-3（4.86×）/dacc 1.29e-3（4.63×）→ **regroup 质量出带（FAIL，两 seed 同向）**。
- 结论：**长训练 GO 阻断项成立**（addendum 预注册映射）；「质量中性/等价」声明不成立；工程测速/容量结果保留。与 `d4xd1-003/005`（cache×regroup 数值阻断）同向。
- 收据 `evidence/c0_d1q_receipt_verdict_20261015.md`；收口影响：D1 quality-neutrality=未通过；D1+ 在线组合维持暂禁；D5 `staged_b32` 仍未启动。
