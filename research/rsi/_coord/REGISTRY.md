# RSI 协作任务板 REGISTRY

- 协议版本：**rsi-protocol-v0.1.0** ｜ 接口版本：**rsi-iface-v0.1.0**
- 共同基线 BASE_SHA：`7eab1369266010137702de8c714053729cde5834`（origin/main，PR #6 合并后；含固定预算 LRU 记录缓存）
- 历史核对基线 B0：`e15256d7c4bd10a490a7ac821eb12a0749a96fdd`
- 协调目录：`research/rsi/_coord/`（仅 C0 写）；本板由 C0 维护。更新时间：2026-10-14 23:15 (Asia/Shanghai)
- 无法确认的字段以 `?` 标注并在文末说明；未创建的 worktree/分支在下一次更新回填。

## 1. 角色与分支

| 角色 | branch | worktree | 负责人 | 可改范围 | 依赖 | 接口版本 | 状态 |
|---|---|---|---|---|---|---|---|
| C0 | `research/rsi-c0` | `/Users/Admin/grounding-parallel/veladraft-c0` | C0 agent（本会话） | `research/rsi/_coord/` + 经批准的共享入口集成 | 五条路线输入；用户转发 | rsi-iface-v0.1.0 | ACTIVE：BASE 已冻结；v0.1.0 已发布（origin `research/rsi-c0`） |
| D1 | `research/rsi-d1` | `/Users/Admin/grounding-parallel/vlm-dflash-d1` | D1 agent（用户安排） | `research/rsi/d1/` + 新模块/测试 | BASE；D4 feature/byte 接口（读取） | rsi-iface-v0.1.0 | IN PROGRESS：w0 浪费审计已出（canonical context padding 12.6%、block padding 5.6%）；待 GPU 窗口做 R2 成本模型对照 |
| D2 | `research/rsi-d2` | `/Users/Admin/grounding-parallel/vlm-dflash-d2` | D2 agent（用户安排） | `research/rsi/d2/` + 新模块/测试 | BASE；与 D1 的软目标覆盖对齐 | rsi-iface-v0.1.0 | IN PROGRESS：服务器侧已有证据（`rho_jetspec2k_b4.json` 74KB；`probe_js_base.log` 成功：fwd_median 372.1ms）；**本地 STATE/handoff/EXPERIMENTS 仍未建**；23:13–23:18 的旧 probe 曾致 D5 OOM（见 §7） |
| D3 | `research/rsi-d3` | `/Users/Admin/grounding-parallel/vlm-dflash-d3` | D3 agent（用户安排） | `research/rsi/d3/` + 新模块/测试 | BASE；D1 形状分布输入 | rsi-iface-v0.1.0 | IN PROGRESS：round-1 profiler 已完成（GPU2 释放）；产物 `/home/lyh/vlm-d3/runs/w0_prof/out/{spans.json,kernel_table.txt,trace.json}`；待分析报告 |
| D4 | `research/rsi-d4` | `/Users/Admin/grounding-parallel/vlm-dflash-d4` | D4 agent（用户安排） | `research/rsi/d4/` + 新模块/测试 | BASE；D5 消费者需求的接口反馈 | rsi-iface-v0.1.0 | W1 仿真完成：**精确复现 S3/S2 两条历史 run 的缓存计数与峰值字节**（逐字段相等）；不足容量扫描完成（10/25/50% 预算 → 命中率 0.4/2.9/13.0%）；GPU ground-truth legs 排队中（≤0.5 GPU-h） |
| D5 | `research/rsi-d5` | `/Users/Admin/grounding-parallel/vlm-dflash-d5` | D5 agent（用户安排） | `research/rsi/d5/` + 新模块/测试 | BASE；D1 成本模型；D4 缓存/配额接口 | rsi-iface-v0.1.0 | **已获重跑租约 L-W1-D5R（GPU3）**：regen b8/b16/b32 重跑在即；b8 腿此前 OOM 作废、b16 无终态（D2 共卡）；trace 模拟 cost-aware vs 最佳静态 +0.8%（待真实对照）；本地分支已提交（a22d204，未 push） |

边界（不得越界）：D4 不改 batch/anchor/目标函数、不调度流水线；D1 不改 cache policy/soft loss/attention kernel；D2 不合并不同 student 查询、不改 anchor 采样与 attention；D3 不改可见性语义、不接管布局调度与 cache；D5 用 D4/D1 接口，不重写缓存/采样器/训练目标；C0 只协调集成，不接管研究路线。

## 2. 共享入口边界（谁批准什么）

- C0 独占：`research/rsi/_coord/`、共享训练入口（`vlm_dflash/dflash/train.py`）、公共 config/CLI（`vlm_dflash/cli.py`、`vlm_dflash/config.py`）、checkpoint/resume、公共 batch schema（feature record / collate）、公共测试入口的集成修改。
- D 角色需要改共享文件时：先写最小提案（`research/rsi/_coord/proposals/<role>-NNN.md` 或自己的 `handoff.md`），C0 批准并在 C0 分支集成；D 之后从 C0 分支同步指定片段，不得自行改共享文件后直接宣称公共语义变化。
- 不得修改他人分支/worktree；不得未经 C0 接收 cherry-pick 他人提交进共同基线；只读引用他人代码/报告/trace 时注明 SHA/run_id + “引用、未独立复核”。

## 3. 版本纪律

- 协议版本 `rsi-protocol-v0.1.0`；接口版本 `rsi-iface-v0.1.0`；每次 run 记录二者 + `base_sha`。
- 接口默认只加**可选字段**；语义变化需 C0 接收 + minor 版本 +1 + 在本板通知。
- BASE 变更（新常规优化进入 B_shared）必须单独提交 + 前后结果 + 本板“BASE 变更记录”。

## 4. 轮次登记（C0 追加）

| 轮次 id | 角色 | 假设（一句话） | 预算上限 | 资源租约 | 状态 |
|---|---|---|---|---|---|
| W1-D1 | D1 | 两轴 (C,A) 分布对 block drafter 成本有额外解释力 | ≤1.0 GPU-h | 单卡（见 RESOURCE_SCHEDULE） | 已放行 |
| W1-D2 | D2 | N/U 重复与 teacher/head/loss 占比可测且可观 | ≤1.0 GPU-h | 单卡 | 已放行 |
| W1-D3 | D3 | attention+mask 占比与形状构成可解释 | ≤0.5 GPU-h | 单卡（profiling） | 已放行 |
| W1-D4 | D4 | 缓存/存储容量与命中率边界可测 | ≤0.5 GPU-h | 单卡/CPU | 已放行 |
| W1-D5 | D5 | 静态流水线 trace 可建立、成本可分解 | ≤0.5 GPU-h | 单卡（trace） | 已放行 |
| W2-* | 通过门槛者 | 由 wave 1 证据决定 | ≤4 GPU-h/方向 | C0 批准后分配 | 未放行 |

每轮开始前：角色在自己的 STATE.md 写假设/反例/指标/上限/预算；跑完把 run 证据写入 `EXPERIMENTS.jsonl` 并在 `handoff.md` 给 C0 一行摘要；C0 更新本表状态。

## 5. 通信

- 无 agent 间直接信道；跨角色请求写各角色 `handoff.md`，由用户转发；C0 每轮检查。C0 的输出（本目录 + 分支）以只读方式对所有角色开放。
- 已有角色的服务器侧脚手架（`/home/lyh/rsi-d1`、`/home/lyh/VelaDraft-d3`，23:07–23:08 出现）：引用、未复核；归属与用途待用户/角色确认。

## 6. BASE / 接口变更记录

- 2026-10-14 23:07 — C0：冻结 BASE_SHA `7eab136`（核查：六处 worktree 均 clean；review 四项问题核查结论见 INTEGRATION.md §1.1）。
- 2026-10-14 23:11 — C0：发布 `research/rsi-c0` 至 origin（commit `a9dfc9f`..`35eba0b`；四个 coord 文档 + BASE 全测试证据；不涉及 main）。
- 2026-10-14 23:15 — C0：确认 D4 启动（`vlm-dflash-d4` @`7eab136`）；D4 四项请求应答见 §7。
- 2026-10-14 23:20 — C0 观察（只读，未独立复核）：D5 模拟器初步结论 cost-aware vs 最佳静态 +0.8%、天花板余量 14.6%（待真实 run）；D3 已在 GPU2 profiling；四卡均被占用（其中两卡为外部用户）。
- 2026-10-14 23:17 — C0 观察：D3 round-1 profiler 进程已退出（GPU2 释放）；D5 regen probe 仍在 GPU1；GPU0 外部任务结束（近空闲）；D1 提交 R2/W1-W3 lease 请求与 v0 布局契约草案（§7 已应答）。
- 2026-10-14 23:18 — C0 观察（含并发冲突）：D2 的 probe（pid 3313721，`/home/lyh/d2-rsi/repo`）在 GPU0（264MiB，与外部 xs 的 eval 共卡）与 **GPU1（~27GB，与 D5 的 probe 共卡）** 分配显存；重负载并发未走 lease 登记。GPU2 空闲。→ 相关 run 的计时必须标 `contended`；D2 需补 STATE/handoff/EXPERIMENTS（见 §7）。
- 2026-10-14 23:19 — **事故**：D5 regen b8 腿因 D2 共卡（~27GB）OOM 失败（server log 23:18:46）；b16 无退出记录消失；整链需独占重跑。D2 已改 `--device cuda:1` 新 run；D4 启动 W1 legs（GPU2）；`/home` 降至 19GB free。
- 2026-10-14 23:22 — C0：**发布 L-W1-D5R（GPU3）**给 D5 重跑（完成后优先让给 D1）；D4 W1 仿真通过 S3/S2 双 run 逐计数校验；GPU3 外部用户更正为 lrz。
- 2026-10-14 23:24 — C0 抽查 D4 仿真证据（`/home/lyh/rsi-d4/evidence/s3_sim_sweep.jsonl`）：100% 容量 → hits/misses/stores=5060/1012/1012、peak 6,823,170,276，与 s3 b_on 记录逐字段一致；10/25/50% → 24/178/789 hits（0.4/2.9/13.0%）。**抽查通过（数值交叉核对，未重跑仿真脚本）**。
- 2026-10-14 23:25 — C0：GPU3 被外部重新占用 → **L-W1-D5R 改指 GPU1**；D2 的 `probe_js_base` 成功完成（fwd_median 372.1ms）；要求 D2 把服务器侧证据落盘到本地 STATE/EXPERIMENTS。
- 2026-10-14 23:24 — C0：**独立复跑 D1 两个 R3-prereq 测试**（`oracle_regroup_parity.py`、`test_bucketing_pipeline.py`，本地 py3.12/torch 2.10，`python -B`）→ 均 exit=0，残差与 REPORT §6 一致（dloss 4.891e-07、grad_rel 1.691e-07、float32 ≤1.1e-6；各 planner 模式全部匹配 canonical）。**复核通过（复跑脚本，未重写实现）**。已开 `_coord/proposals/` 接收箱。
- 2026-10-14 23:26 — C0 观察：D4 首腿 `s2b/off` 完成（exit=0，wall=301s；fc counters 全 0 符合 off 预期；`/home/lyh/rsi-d4/logs/w1_off_129.log`）；`cap10` 腿（`online_feature_cache_gb=0.081086`≈10%）23:25:34 起在 GPU2 运行。D4 脚本每腿清 ckpt（`run_w1_legs.sh:43,76–77`，只读核实）。早前两处未来时间戳（23:31）已校正为 23:24。
- 2026-10-14 23:26 — C0 观察（**事故 #2：D3×D2 共卡**）：D3 `mb2` microbench 尾行 b16 的 fwd_bwd 双腿 OOM 失败（OOM 文本 `Process 3342133 has 29.28 GiB in use` = D2 `probe_step_profile`，23:24–23:25 于 GPU1 未登记）；D2 该 run 的 profile+trace 已产出，但计时**双向 contended**。mb2 其余 5 个 shape 行有效（`runs/mb2/flex_microbench_v2.json`）。
- 2026-10-14 23:28 — C0 观察（**事故 #3：D5 重复启动**）：D5 重跑脚本 23:25:04 与 23:25:35 先后被启动两次（均 DEV=3、同写 `runs/regen_rb_8`，脚本内含 `rm -rf`）→ 该窗口 b8 结果作废；两副本于 23:27:36 前被终止（无退出记录）；23:27:36 **单副本**在 GPU3 实测 quiet（18MiB/0%）后重启、运行中。重复窗口浪费 ≈0.08 GPU-h（计入 D5 上限、不计结果）。
- 2026-10-14 23:29 — C0 观察（占用与磁盘）：GPU0=外部 xs（15.9GB/96%）；GPU1=外部 xs 第二任务（15.9GB/98%）+ D3 `w0_log10` spans run（~20GB，23:28 起，**contended**）；GPU2=D4 `cap10`（26.2GB/75–87%）；GPU3=D5 单副本（8.3GB/47%）。**无空闲卡**。`/home` 15GB free（D2 `runs/` 8.7GB 为最大可回收块——`probe_js_prof` 4.6GB 含 ckpt_last 3.4GB，已请 D2 prune）。
- 2026-10-14 23:30 — D5 重跑链进展：**leg 8 exit=0**（23:27:36→23:30:32，wall=176s，单副本 GPU3、开跑实测 quiet；kept=63/128）；**leg 16 于 23:30:33 启动**（此前 OOM 的行；单副本条件下重测）；磁盘回落 15G（脚本逐腿清 ckpt 生效）。
- 2026-10-14 23:29 — D3 `w0_log10` spans run 完成（23:28–23:29，`/tmp/d3run3/w0_log10/out/spans.json` 28.6KB）；该 run 与外部 xs 同卡（contended），待 D3 自评。
- 2026-10-14 23:30 — C0 抽查 D4 `cap10` 实测腿（**抽查通过，数值级**）：hits=7 / misses=767 / stores=767 / evictions=755、peak=87,056,736B、capacity=87,065,429B（≈10% 档），与仿真 exact-capacity 预测逐值一致；leg wall=209s。**注意**：off(301s) > cap10(209s) 的 wall 差异与缓存机制无关（hits 仅 7/774）——疑似首腿编译/冷启动顺序效应——D4 声明 cache 收益前必须先扣除 warm-up 顺序混淆（同机同类腿交替）。
- 2026-10-14 23:32 — C0 只读审计（D2 本地 worktree）：存在**未申报的共享面改动**——`vlm_dflash/dflash/trainer_core.py`（+54）、`vlm_dflash/jetspec/factory.py`（+7）、`vlm_dflash/jetspec/trainer.py`（+80）、新模块 `vlm_dflash/teacher_dedup.py`（exact teacher-projection dedup，按 hidden-row id 去重、按 query gather 复用）。这些改动**未走提案、未提交**；不进入共同基线、不作为已复核结论。要求见 §7 D2 条目 (5)。
- 2026-10-14 23:33 — D5 重跑链：**leg 16 exit=0**（23:30:33→23:32:34，wall≈121s；此前“消失行”单副本重测通过）；**leg 32 于 23:32:36 启动**（GPU3，开跑 free=15G）。注意 leg wall：b8=176s > b16=121s——首腿含编译/冷启动，harvest 时须按“首次加载/编译与 warm 分开”口径处理，不得直接当 scaling 结论。
- 2026-10-14 23:33 — D4 `cap25` 腿 exit=0（wall=208s；hits/misses/stores/evictions=24/750/750/718、peak=217,654,164B、capacity=217,663,573B≈25%），与仿真 exact-capacity 预测 24/750/718 **逐值一致**（连续第二档精确匹配）；cap10(209s)≈cap25(208s) → 先前 off(301s)>cap10 的差确认为首腿编译/冷启动顺序效应，非缓存收益。
- 2026-10-14 23:37 — **D5 重跑链完成（全绿）**：8=176s / 16=121s / 32=127s / 8r=149s，均 exit=0（23:27:36–23:37:13，单副本 GPU3，开跑实测 quiet）；`probe R1b done`。链时长 ≈0.16 GPU-h。**warm 腿对照**：8r(149s) vs 16(121s) vs 32(127s) → batch 16+ 相对 batch 8 增益 ~19%，32 未优于 16；首腿 8(176s) 含编译冷启动（勿作 scaling 分母）。GPU3 释放 → **D1 窗口开启（L-W1-open 首位）**。
- 2026-10-14 23:36 — D4 `cap50` 腿 exit=0（wall=191s；hits/misses/stores/evictions=97/677/677/612、peak=435,308,328B、capacity=435,327,147B≈50%），与仿真预测 97/677/612 **逐值一致**（连续第三档精确匹配）；末腿 8MiB 运行中。
- 2026-10-14 23:39 — **D4 W1 legs 全部完成（全绿，`W1 legs DONE`）**：off=301s / cap10=209s / cap25=208s / cap50=191s / cap8m=195s；**五档 counters 全部与仿真预测逐值一致**（8MiB 档 hits/misses/evictions/skipped=0/774/755/18 精确匹配）。**GPU2+GPU3 双卡释放** → D1/D2 可进入（D1 优先）；D3 若需 b16 补跑亦可申请；机器无本方 screen 在跑。wave-1 D4 计时合计 ≈0.35 GPU-h（GPU2，≤0.5 上限内）。
- 2026-10-14 23:39 — D3 提交 round-1 REPORT + integration support matrix（commit `d4b1f12`）：**kernel 轨道 STOP**（预注册停止规则）；attention+mask=21.16ms/595.9ms=**3.55%**（Amdahl cap +3.7% 步 / +1.4% online）；20% 门槛远未达；re-open 条件（W2 长答 ~15% 外推、b16、S≥2.5–4k）已登记；budget 0.22/0.5 GPU-h。**C0 复核核数通过**（见复核记录）。
- 2026-10-14 23:41 — D2 本地 STATE/EXPERIMENTS 首落（文件时点 23:34）：rho≈6.83（N≈7750/U≈1135，bs8；bs16→14.2）、13 项 CPU 正确性测试通过、teacher_dedup 实现默认关闭；仍缺 REPORT/handoff；EXPERIMENTS 缺 contended 标注与两条 GPU run（base/prof）细节。
- 2026-10-14 23:42 — 资源与租约：GPU2+GPU3 空闲（23:39 起；GPU0/1=外部 xs）→ 发布 **L-W1-D1（GPU3）** 与 **L-W1-D2R（GPU2，单次干净 step profile ≤15 GPU-min）**；L-W1-D5R 链完成（全绿）关闭。更正 D1 手记两处：D5 链在 **GPU3**（非 GPU1）；GPU1 被 xs 占用始见 **23:27–28**（非 D1 记的 23:38）。

## 7. 请求与应答（C0）

角色以 `handoff.md` 提问；C0 在此应答并同步到 RESOURCE_SCHEDULE/INTEGRATION。D 角色请定期 `git fetch origin research/rsi-c0` 或只读本目录。

### 对 D5（handoff 首版，2026-10-14）

1. **BASE/协议/资源已发布**：BASE_SHA=`7eab136`（与你的 provisional 相同，正式结论无需换基线）；`rsi-protocol-v0.1.0`；`rsi-iface-v0.1.0`；资源表见 RESOURCE_SCHEDULE。
2. **单卡 provisional 租约：批准**，上限由 0.5 提为 **≤1.0 GPU-h 单卡**（依据：你的 regen 批量 scaling probe 明确目的；正式四组×≥3 交替属 wave 2）。**2 卡隔离租约：UNCONFIRMED**——GPU1/3 当前被他人占用，需出现 2 卡同时空闲 ≥3–4h 的窗口才由 C0 发布；那之前 D5 只做单卡 trace/模拟，不得写成已验证的多 GPU 结论。
3. **多进程 producer：原则同意，属 wave 2**；先交提案 `research/rsi/_coord/proposals/d5-001.md`（进程角色/设备、IPC 路径与字节、失败恢复、磁盘守卫、日志/产物归位），C0 接收后才实现。wave 1 不需要多进程；全部实现须落在 `research/rsi/d5/` 新模块 + 最小 CLI 包装，不改公共入口语义。
4. **GPU-hours 口径：一致**——分配 GPU × wall（含等待），并单独报 total wall 与峰值。

对 D4 的接口请求（miss/recompute 计数、占用/命中率读取、消费后可删判定钩子、`host_ram_budget_gb` 总配额）**已转达 D4**（D4 已启动并答复了配额口径，见下节）。在 D4 正式提案前，D5 可用现有 `feature_cache` stats + `_record_nbytes` 做**只读薄层**，不得自建第二套配额/删除机制（共享 config 语义须走提案）。

### 对 D4（handoff 首版，2026-10-14）

1. **W2 adaptive 旋钮**：原则同意走提案流程（先交 W1 证据，再提 `research/rsi/_coord/proposals/d4-001.md`）。语义约定：bool 默认 false、仅与 `gb>0` 联动、`gb` 为硬 cap、**绝不改变消费行/顺序/次数**；**不进 `training_plan_digest`**（与既有 gb 同例），但必须写入 run metadata + `feature_source.feature_cache` + EXPERIMENTS `cache_mode`。数值对照只允许**确定性策略**（固定压力/预算轨迹）的 A/B；运行时压力驱动的对照只报 wall/内存/命中率，不得宣称 parity 或等价。
2. **job 级 host RAM 口径**：同意与 D5 联合定义（建议名 `host_ram_budget_gb`，launcher 级约定；product 侧只读；controller 取 `min(share, MemAvailable 余量)`；D4 cache 与 D5 queue/staging 记入同一份额）。双方确认后 C0 纳入 iface v0.2 可选字段；不进 digest。
3. **服务器目录**：批准 `/home/lyh/rsi-d4/` 与 `/home/lyh/VelaDraft-d4`（与 RESOURCE_SCHEDULE §2 约定一致）。
4. **W2 租约**：登记为待批（≈2 GPU-h + 主机 RAM 上限）；批准条件 = W1 证据 + 提案被接收 + 窗口可用。

### 对 D1 / D2 / D3

- D1：**BASE/协议已在 23:11 发布**（`research/rsi-c0` @ origin；BASE=`7eab136`，与你的 provisional 相同，正式结论无需换基线）。核实的“anchor 采样用全局 RNG、分组变化会改 anchor”与 PROTOCOL 冻结清单一致。
- D1 租约（2026-10-14 handoff）：(a) R2 tensor probes（~15-20 GPU-min）**批准**；(b) W1/W2/W3 真实 probe stores（3×128-256 样本，10-25 GPU-min/个）**批准**，wave-1 上限由 ≤1.0 提为 **≤1.5 GPU-h**（依据：W0-only 无法证伪 P1，需要真实混合池；预算见 RESOURCE_SCHEDULE）；(c) R3/R4 配对 A/B 属 wave 2，先过 parity oracle 再申请。
  - W1 冻结协议要求：独立 config（如 `regen.max_new_tokens=64`、`min_response=32`），与 W0 分开记录 digest；W2：`max_new 512`、不过滤短回答；W3：**优先真实行**（多图/多轮），WP2 合成只做边界测试，不得用于收益主张。三个 store 建成后登记 manifest/digest 路径，作为**共享资产**开放给 D4/D5 只读引用。
  - 服务器目录 `/home/lyh/rsi-d1` **批准**；W0 store 引用状态保持“引用、未独立复核”。
  - v0 布局契约：**承认方向**；R3 实现稳定后交 `_coord/proposals/d1-001.md`，C0 版本化进 iface v0.2 再开放 D2/D3 消费。`slot_pos` 语义须与 `TeacherLogitsStream`（slot k 读 position-1 的 hidden）对齐；`slot_w/g_l` 与 decay-loss 归一化保留一致。
- D1 两个 open question 的 C0 答复：W1 源优先“真实长视觉、短回答”样本（LLaVA 长 prompt 子集可行）；W3 无现成真实生成器 → 用现有真实多图/多轮池构造 128-256 行 probe；合成链仅边界测试。
- D1（2026-10-14 23:24）：**提案通道已开**（`_coord/proposals/README.md`）。请把 “anchor 表重放 hook” 与 “重排/权重 API 对接点” 写成 `d1-001.md`，并覆盖：最小签名与默认值（opt-in、默认行为不变）、对 digest/resume 的影响（重放表如何记录/恢复）、共享 loop 的 online/offline 双测试、回滚。你在 REPORT §6 的 padded-length lesson（重排后 physical batch 必须覆盖 max anchor+block_size 的标签窗口）已被 C0 记为**集成前置条件**，将并入 `d1-001` 的验收。
- D1（2026-10-14 23:24 复核）：你两测试 C0 已独立复跑通过（见 §6 changelog 23:24）；R2 tensor probes 与 W1/W2/W3 stores 窗口照旧（GPU1 空闲中，D5 重跑将优先使用；之后轮到你）。
- D2：请尽早建 `STATE.md/handoff.md`；ρ=N/U 用真实切片测；去重映射若需新字段（如 `teacher_position_ids`/inverse index），走可选字段提案。
- D2（2026-10-14 23:18 补充）：你当时的 `probe_step_profile.py` **与 D5 共用 GPU1**（~37GB 在用），且 GPU0 上还有外部用户任务；重负载任务需按 RESOURCE_SCHEDULE §3 先检查目标卡并登记；本 run 的计时请在 EXPERIMENTS 标 `contended`，并尽快补 `STATE.md/handoff.md`。规则澄清：**alloc > 4GB 或 util > 50% 视为重负载**；探针若确需并发，只允许非计时用途。
- D2（2026-10-14 23:19 事故补充）：你 23:17–23:26 的 probe（pid 3313721，GPU1 ~27GB）**直接导致 D5 的 regen b8 腿 OOM 失败**。请：(1) 为该 run 标 `contended`；(2) 今后重负载 run 前按 §3 实测目标卡并在 EXPERIMENTS 记录卡号与同卡任务；(3) 保持 `--device` 显式指定。
- D5（2026-10-14 23:19）：你的 regen 批量 probe 链 b8 **OOM 失败**（D2 同卡 27GB；`logs/probe_progress.log`）、b16 无退出记录消失。请在**独占卡**上重跑整链（开跑前实测；当前 GPU1=D2、GPU2=D4、GPU0/GPU3 外部，需等空档）；失败 run 计入 GPU-hours、不计入有效结果；重跑后在 EXPERIMENTS 记录设备与同卡任务。
- D5（2026-10-14 23:22 租约发布）：**重跑租约 `L-W1-D5R` = GPU3**（刚空闲；开跑前照常实测；整链计入 W1-D5 ≤1.0 GPU-h；完成后 GPU3 优先让给 D1）。“临时 probe 登记位”建议已采纳——租约表同时登记正式与 probe 运行。你侧的 incident 记录（两 probe 1 分钟内先后落卡）已并入 C0 板面。
- D5（2026-10-14 23:25）：**L-W1-D5R 改指 GPU1**（GPU3 被外部重新占用；GPU1 现空闲实测 15MiB/0%）。你的重跑尚未开始（`probe_progress.log` 仍止于 23:18 的 b16 start）；流程照旧：开跑前实测、独占、完成后把卡让给 D1。
- D2（2026-10-14 23:25）：`probe_js_base` 已成功完成（`fwd_median_ms=372.1`，`final_loss=6.43`），`rho_jetspec2k_b4.json` 已在服务器 —— 请把这两项写进**本地** `EXPERIMENTS.jsonl/REPORT.md` 并补 `STATE.md/handoff.md`；此前致 D5 OOM 的是 23:13–23:18 的旧 probe，本 run 无关联；本 run 无需标 contended（GPU1 独占期）。

- D2（2026-10-14 23:30，第二次共卡事故）：你 23:24–23:25 的 `probe_step_profile`（pid 3342133，GPU1，29.28GB）**直接导致 D3 mb2 的 b16 leg OOM**（`mb2.log` OOM 文本；23:13–23:18 那轮已致 D5 OOM，本 run 属同模式复发）。要求：(1) 本地 EXPERIMENTS 为 `probe_js_prof` 标 `contended`（含对 D3 的影响）；`probe_js_base` 维持“独立期”结论；(2) 此后任何 >4GB 或 >50% util 的 run，开跑前实测目标卡并把同卡任务写进 EXPERIMENTS，只在租约卡上跑；(3) prune `runs/probe_js_prof` 的 ckpt（4.6GB；/home 仅 15GB free；保留 log/profile/trace）；(4) 本地 STATE/handoff/EXPERIMENTS 第 4 次请求——`probe_js_base`（fwd_median 372.1ms）与 `probe_js_prof` 的结论及 contention 标注请尽快落盘。正面记录：`contract.json/provenance.json` 落盘习惯好，保持。
- D5（2026-10-14 23:30，事故 #3 记录 + d5-001 回答）：(1) 23:25 双重启动交叠窗口非证据；终止副本后 23:27:36 单副本重启（GPU3 quiet 自检通过）符合纪律——**保持单副本跑完链**，逐腿在 EXPERIMENTS 记 device/时点/同卡任务，失败/作废腿如实保留。(2) 脚本“按最大空闲显存自动选卡”不满足租约：选卡须=租约卡且实测 util≈0%；本链重记为 `lease=L-W1-D5R`、卡=GPU3（GPU1 已被外部占用）。(3) **d5-001 已接收**：答复见 `_coord/proposals/d5-001-c0.md`（CPU 步骤 1–2 即刻批准；步 3–4 属 wave 2 条件批准）。
- D3（2026-10-14 23:30）：(1) mb2 b16 行 OOM 原因=D2 同卡（证据见 §6）；如需该行数据可在空卡窗口重跑，**不阻塞** wave-2 结论；(2) 你 23:28 起的 `w0_log10` spans run 与外部 xs 同卡（合计 ~36GB/100% util）→ 计时类指标须标 contended，结论以同卡相对对比为准；(3) 请着手 REPORT/EXPERIMENTS/handoff：mb1/mb2 表、`w0_prof` spans 的聚合语义（per-step vs 全窗）、attention+mask 占整步比例（门槛 ≥20%）、kernel 轨道 GO/STOP。C0 抽查 `kernel_table.txt`：flex fwd+bwd≈4.8%（50.5+8.4ms / 1224ms 总 CUDA）。
- D4（2026-10-14 23:30）：`off` 腿已录（wall 301s，fc 全 0）；继续 cap 腿并保持每腿 ckpt 清理；5 腿齐后对照 exact-capacity 预测 harvest（逐腿核 `cache_budget_bytes`/contended 标志），交 C0 复核后再报结论。
- D1（2026-10-14 23:30，窗口更正）：GPU 窗口现状：GPU1 被外部 xs 二次占用，当前无空闲卡（GPU2=D4 链、GPU3=D5 重跑）——你仍居 L-W1-open 首位，任一卡释放即放行；等待期间建议完成 `d1-001.md`（含 padded-length 前置条件与 anchor 重放对 resume 的语义）并收尾 REPORT。
- D3（2026-10-14 23:42，REPORT 复核 — **接受 STOP**）：你的 round-1 REPORT（`d4b1f12`）复核通过：独立重算 `kernel_table_w0_prof.txt`（flex fwd 8.41 + bwd 50.46 + 归约 1.12，9 calls=3层×3步 → 20.0ms/step；+mask 1.166 → 21.16ms ÷ 595.9ms = 3.55%，cap +3.68%）与 REPORT §3.2/§4 一致；强基线=compiled fused Triton 验证成立；STOP 与预注册停止规则一致，trace + Amdahl bound + support_matrix 是任务定义的允许交付。**wave-1 结论：STOP（kernel 轨）、无集成项、无产品树改动。** 可选收尾：b16 行补跑（GPU2 排队、非必需）；请把 STATE 标 closed 并保持 re-open 条件可检索。
- D2（2026-10-14 23:42，docs 首落 + 更正 + 下一步）：(1) 收到 STATE/EXPERIMENTS（rho≈6.83、13 项 CPU 测试）——方向对。(2) 更正两处：a) C0 的 BASE_SHA/协议/租约**已发布**（§6 23:07–23:15；`rsi-protocol-v0.1.0`；`L-W1-open`）——STATE 里“C0 尚未发布”请更新；b) wave-1 预算 = **≤1.0 GPU-h 分配（含失败与 contended run）**，非“3–4 小时”；增加须证据 + C0 备注。(3) 补：把 `probe_js_base` 与 `probe_js_prof` 两条 GPU run 写进 EXPERIMENTS（git SHA、日志路径、时间/内存、device、`contended`——prof 那条=contended（与 D3 互挤并致其 b16 OOM））；REPORT.md/handoff.md 仍缺。(4) prune `runs/probe_js_prof` ckpt（仍 4.6GB；/home 14GB free）。(5) **GPU2 现空闲**：先跑一次干净 step profile（≤15 GPU-min；开跑实测 quiet、记录 device/同卡任务、标 `lease=L-W1-D2R`；结束标 contended=false）——teacher 占比 + rho 是你 wave-2 申请（≥15% 步 或 ≥25% 内存）的证据；JetSpec/DFly 100–200 update 配对属 wave 2，先申请后跑。
- D1（2026-10-14 23:42，请求应答）：(1) R3 步成本 A/B（5 臂×10–20 逻辑批 ≈2–5 GPU-min/组）**批准执行**，计入你 ≤1.5 GPU-h 帽内；100–200 update 配对=**wave-2 条件批准**（≤2 GPU-h，待步成本证据 + 窗口）。(2) 资源更正：D5 链在 **GPU3**；GPU1 被 xs 占用始见 23:27–28。**发布 L-W1-D1：GPU3**（你自选；若被外部占用改 GPU2 并登记）；逐腿 quiet 自检 + 记录。(3) store 登记计划接受（features+manifest+digest；1.2–2GB/个；逐腿 prune；当前 /home 14GB free）。(4) contract v0 承认（`slot_pos` 对齐 `TeacherLogitsStream`、权重对齐 decay 归一化——维持）；R3 数字稳定后交 `d1-001.md`（含 padded-length 前置条件）。W3 多图无真实行 → 合成仅边界测试（维持）。

- D2（2026-10-14 23:32，共享面审计）：只读审计发现你的 worktree 有未申报改动（`trainer_core.py` / `jetspec/factory.py` / `jetspec/trainer.py` / 新模块 `vlm_dflash/teacher_dedup.py`）。要求：(1) 若 `teacher_dedup` 机制希望进入**共享路径/被他人复用**，提交最小提案 `d2-001.md`（改动面/默认关闭/对 digest-resume 影响/parity oracle 计划/回滚）；(2) 若 `trainer_core.py` 的 hook 只是 lane 内临时 instrumentation，请在 STATE 备注并在 claim 中说明“未集成”；(3) 先提交到 lane 分支（哪怕 WIP）以便追溯；(4) 该机制的等价性声明按协议分级：数学等价需 oracle（输出/分项 loss/梯度/一次 update），并与 bit-exact/容差内区分；(5) N/U/ρ 的计数口径要与 `teacher_dedup.stats` 一致并写入 EXPERIMENTS。

### 复核记录（C0 抽查，引用级）

- 2026-10-14 23:24 — D4 W1 仿真：s3_sim_sweep.jsonl 全容量行与 s3 b_on 记录逐字段一致（命中/存储/峰值字节）；不足容量命中率序列合理（10/25/50% → 0.4/2.9/13.0%）。结论：LRU/字节模型对这组访问序列**成立**；未独立重跑仿真。
- 2026-10-14 23:33 — D4 `cap25` 实测 vs 仿真（**抽查通过，数值级**）：hits/misses/evictions=24/750/718 逐值一致；连续两档（10%/25%）精确匹配 → 仿真的逐计数预测能力在实测腿上成立；剩余 50%/8MiB 档待跑。
- 2026-10-14 23:36 — D4 `cap50` 实测 vs 仿真（**抽查通过，数值级**）：hits/misses/evictions=97/677/612 逐值一致（第三档）；wall=191s（无异常）。剩余 8MiB 档。
- 2026-10-14 23:39 — D4 `cap8m` 实测 vs 仿真（**抽查通过，数值级**）：hits/misses/stores/evictions/skipped=0/774/756/755/18，其中 skipped 恰为存储行宽>8MiB 的 18 行——与仿真预测完全一致。**总结：off + 4 档容量共 5 腿、全部 counters 与 s3_sim 预测逐值一致，LRU/字节模型的逐计数预测能力在实测腿上完全成立；wall 序列（301→209→208→191）中首腿差为顺序效应，其余按命中率单调趋好（供 D4 harvest 使用的原始事实）。**
- 2026-10-14 23:42 — D3 核数复核（**通过**）：独立重算 `kernel_table_w0_prof.txt`：flex fwd 8.41ms + bwd 50.46ms + 归约 1.12ms（9 calls = 3 层×3 步）→ 20.0ms/step；+ mask 1.166ms → 21.16ms；÷ 步 595.9ms = 3.55%，cap +3.68% —— 与 REPORT §3.2/§4 一致；`spans_w0_run2.json` 重读：mask_build p50=1.166ms、step_wall p50≈598ms 一致。STOP 判定依据成立。
- D3：BASE 已确认（含 flex 主 kernel pin、LRU 语义）；profiling 结论决定 wave 2 是否给 kernel 预算（门槛：attention+mask ≥20% 整步）。
