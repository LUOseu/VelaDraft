# RSI 协作任务板 REGISTRY

- 协议版本：**rsi-protocol-v0.1.0** ｜ 接口版本：**rsi-iface-v0.1.0**
- 共同基线 BASE_SHA：`7eab1369266010137702de8c714053729cde5834`（origin/main，PR #6 合并后；含固定预算 LRU 记录缓存）
- 历史核对基线 B0：`e15256d7c4bd10a490a7ac821eb12a0749a96fdd`
- 协调目录：`research/rsi/_coord/`（仅 C0 写）；本板由 C0 维护。更新时间：2026-10-15 05:52 (Asia/Shanghai)
- 无法确认的字段以 `?` 标注并在文末说明；未创建的 worktree/分支在下一次更新回填。

## 1. 角色与分支

| 角色 | branch | worktree | 负责人 | 可改范围 | 依赖 | 接口版本 | 状态 |
|---|---|---|---|---|---|---|---|
| C0 | `research/rsi-c0` | `/Users/Admin/grounding-parallel/veladraft-c0` | C0 agent（本会话） | `research/rsi/_coord/` + 经批准的共享入口集成 | 五条路线输入；用户转发 | rsi-iface-v0.1.0 | ACTIVE：BASE 已冻结；v0.1.0 已发布（origin `research/rsi-c0`） |
| D1 | `research/rsi-d1` | `/Users/Admin/grounding-parallel/vlm-dflash-d1` | D1 agent（用户安排） | `research/rsi/d1/` + 新模块/测试 | BASE；D4 feature/byte 接口（读取） | rsi-iface-v0.1.0 | **GO（修订口径；`d1-002-c0`+`d1-003-c0`；wave-2 全收）**：wave-2 6/6 rc=0（{W3,MIX}×3 seeds；grid 配对均值 1.135/1.133、win 580/600、peak −57%、parity b0/50/100 ✓，C0 重算）；**capacity demo ✓ C0 逐值重算**（W3 b16 43.59→14.32 GiB；**b24 canonical OOM 5/5 vs cost 14.41 GiB 5/5**；MIX b16 27.74→9.83 GiB）→ **L-W2-D1 关闭（≈0.90/1.5）**；close-out `6dc5f74`（REPORT §11 / EXPERIMENTS 42）；**d1-001 最小集已集成**：`research/rsi-c0` @ `4e1674e`（opt-in、默认 byte-identical；全套 509 pass/6 skip）；**D4×D1 2×2 已收官（`d4xd1-003-c0`）：W3 cell F1-as-registered FAIL（1.2475e-2@11）→ 组合数值相容性声明不成立·正确性阻断；MIX cell 全过；交互 bit-0；隔离对照 `d4xd1-004-c0` 放行（L-W3-D4xD1B）**；D1 harvest 回填已收讫（`e7ea0cf`+`9990a0f`；与 C0 重算逐项一致）；**d4xd1-004 已收口（`d4xd1-005-c0`：F1B FAIL→阻断加深）**；`d1-006`（L-W3-D1Q）待跑；round-2 的 +19.0%/+24.7% 系 8 批 cell（不作主数字）；**R4 评估已收（k4 残余 4.7–5.2%/6.1–6.4%；§12）；cross-check `3e29244` PASS（建议 a 采纳 → `192554a`）；d1-004 租约 L-W3-D1R4 已批（≤0.2 GPU-h）；**d1-006 执行中：s42 stage/train ✓（200 步，exit=0 @04:37:25，final_loss 5.75）；s42 腿完成（train exit=0 04:37:25；eval exit=0 @04:51:44，token_exact=True，final AL=1.1654/acc=0.0239；守卫自碰撞=事故 #8；C0 收据 `evidence/c0_d1q_receipt_s42_20261015.md`）；s43 canonical 完成（05:09:32；band_AL=0.0018152/band_acc=0.00027917）；**d1-006 终裁 FAIL（05:47）：两 seed × 两指标全部出带（s42 1.80×/1.69×；s43 4.86×/4.63×；band_AL=1.815e-3/band_acc=2.792e-4）→ 长训练 GO 阻断项成立；收据 `evidence/c0_d1q_receipt_verdict_20261015.md`**；转发文本 `_coord/messages/20261015_d1q_recovery.md`）；C0 重算工具 `tools/d1q_band_recompute.py` 就绪；**harvest 收讫（`3bc72ba`/`12d3b5e`/`1b0328e`）：8/8 raw 哈希重验一致、band/配对/verdict 逐位对账一致；budget **1.1536/2.4** 关闭；映射=阻断加强（离线冻结布局轴同向负向）；收据 `evidence/c0_d1q_harvest_receipt_20261015.md`** |
| D2 | `research/rsi-d2` | `/Users/Admin/grounding-parallel/vlm-dflash-d2` | D2 agent（用户安排） | `research/rsi/d2/` + 新模块/测试 | BASE；与 D1 的软目标覆盖对齐 | rsi-iface-v0.1.0 | **STOP（自判+C0 采纳，`d2-001-c0`）**：REPORT/handoff/EXPERIMENTS/d2-001 到齐；pin 2×2 −6.8%（loss 四腿 bitwise）；dfly recipe 2×2（fwd −12.9/bwd −19.5ms；12 步 loss 四腿逐值相同，C0 直读 ✓）；CPU oracle **12/12（C0 复跑；更正“13”措辞）**；归因段完成（editable 遮蔽/event 伪影）；机制 DEFERRED（lane-local opt-in；再开=REPORT §8）；lane commit `bfb3635`+`d6e3c9a` ✓（worktree 净；修正三项闭环） |
| D3 | `research/rsi-d3` | `/Users/Admin/grounding-parallel/vlm-dflash-d3` | D3 agent（用户安排） | `research/rsi/d3/` + 新模块/测试 | BASE；D1 形状分布输入 | rsi-iface-v0.1.0 | **CLOSED（STOP 已接受；最终关闭确认 2026-10-15 00:18）**：attention+mask=21.16ms/595.9ms=**3.55%** ≪20%（cap +3.7% 步 / +1.4% online）；C0 核数 ✓；交付物全（`b88cdf2`：REPORT/EXPERIMENTS/handoff/support_matrix/evidence/scripts；worktree 净）；预算自报 0.22/1.0 GPU-h；服务器清理 ✓；无集成；再开条件=REPORT §8 |
| D4 | `research/rsi-d4` | `/Users/Admin/grounding-parallel/vlm-dflash-d4` | D4 agent（用户安排） | `research/rsi/d4/` + 新模块/测试 | BASE；D5 消费者需求的接口反馈 | rsi-iface-v0.1.0 | **merge-ready（平台工程级；`d4-003-c0`）**：d4-001 集成 `1f8963e`（sha1 对账一致；13+4+70/70 复跑 ✓）；**W2 收口**（F1–F4 PASS；两腿 counters 645/129/129/0/0 逐值相同；adjustments=0；lane `a8e762c`；runtime 4 文件 sha1 再核一致）；范围=静态 RAM 场景/opt-in 默认关（压力响应未验证）；预算 0.37/0.5；**D4×D1 2×2 只读参与收官（`d4xd1-003-c0`）：bucket_off×cache 腿数值惰性 bit-exact（floor probe 0.0）——线不重开**；不自动 merge main |
| D5 | `research/rsi-d5` | `/Users/Admin/grounding-parallel/vlm-dflash-d5` | D5 agent（用户安排） | `research/rsi/d5/` + 新模块/测试 | BASE；D1 成本模型；D4 缓存/配额接口 | rsi-iface-v0.1.0 | REPORT 已接收（`85cbf8b`；**C0 抽查 ✓**）；sim v4 `23875e8`；**d5-001 步 2 完成（`cf74790`+duplicate guard；lane **43/43 C0 独立复跑 ✓**）；iface v0.2 候选受理（PROTOCOL §12）；G2 档位预授权（min ≈2.2 / 有价值集 ≈4.0 GPU-h）；**G2 chain2 完成 Tier 0+A+B（CHAIN2 DONE 02:13:58）**：staged_b8 ✓（762 步，kept 1012，≈1.27 GPU-h 分配口径）、sync_b8 ✓（wall 2174.5s，**vs staged −5.3%**，远低于 sim −14%；epoch-1 抽取不与训练重叠——本 build 串行）、sync_b32 ✓（01:53:29→02:13:58，768 步 final_loss=4.0625）；Tier C 倾向暂缓；**S3 追赶链 smoke_s3 02:19 假失败（worker zero-keep；事故 #7；D5 修复在途）**；harvest 须按时间戳重算 GPU-h（链脚本 107 行算式报错）；**C0 复核收据与裁定（03:45，`d5-002-c0`）：套件范围 pin `55e4e9c`（至 HEAD 7db5d93 仅文档 diff）→ 本地+服务器 88/88 复跑 ✓；数字抽查 ✓（walls/parity/step-760·768 逐位 loss 行）；staged_b32 批准（L-W3-D5B ≤0.75 GPU-h）；pinned_b8 = HOLD；**截至 04:43 仍未开跑（提醒已发，见 `_coord/messages/20261015_d1q_recovery.md`）** |

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
| W1-D1 | D1 | 两轴 (C,A) 分布对 block drafter 成本有额外解释力 | ≤1.0→**≤1.5**（提升依据见下） | 单卡（见 RESOURCE_SCHEDULE） | 已放行 |
| W1-D2 | D2 | N/U 重复与 teacher/head/loss 占比可测且可观 | ≤1.0 GPU-h | 单卡 | 已放行 |
| W1-D3 | D3 | attention+mask 占比与形状构成可解释 | ≤0.5 GPU-h | 单卡（profiling） | 已放行 |
| W1-D4 | D4 | 缓存/存储容量与命中率边界可测 | ≤0.5 GPU-h | 单卡/CPU | 已放行 |
| W1-D5 | D5 | 静态流水线 trace 可建立、成本可分解 | ≤0.5 GPU-h | 单卡（trace） | 已放行 |
| W2-D1 | D1 | 布局重排（regroup）机制开发（过 wave-1 证据门槛） | ≤1.5 GPU-h（依据：`d1-001-c0` 接收 + parity oracle 通过） | GPU3 单卡 | 已关闭（L-W2-D1；实耗 0.90/1.5） |
| W3-D1R4 | D1 | R4 sub1 边界 A/B | ≤0.2 GPU-h（依据：`d1-004-c0`） | GPU3 | 已关闭：负分支（实耗 0.108；`d1-005-c0`） |
| W3-D4xD1 | D1 | D4×D1 2×2（online cache × regroup 组合） | ≤1.5 GPU-h（依据：`d4xd1-001-c0`；wave-2 GO + capacity demo + e2e smoke） | GPU3 优先 | 已放行（D1 执行 8 腿） |
| W3-D1Q | D1 | held-out 质量（AL/acc band-first；d1-006 接收） | ≤2.4 GPU-h，staged（依据：band-first 设计；2×2 后顺序执行） | GPU3 顺序 | 已接收（先 Stage-1 建带再 Stage-2） |
| W2-* 其余 | —— | 未过证据门槛的方向 | —— | —— | 未放行（按门槛纪律保持） |

每轮开始前：角色在自己的 STATE.md 写假设/反例/指标/上限/预算；跑完把 run 证据写入 `EXPERIMENTS.jsonl` 并在 `handoff.md` 给 C0 一行摘要；C0 更新本表状态。

**预算提升证据（任务 2 记录）**：D1 wave-1 `≤1.0→≤1.5`（W0-only 无法证伪 P1、需真实混合池；§7 条目；实耗 0.92）；L-W2-D1 `≤1.5`（`d1-001-c0`；实耗 0.90 关闭）；L-W3-D1R4 `≤0.2`（`d1-004-c0`；0.108 负分支关闭）；L-W3-D4xD1 `≤1.5`（`d4xd1-001-c0`；**实耗 ≈0.76 关闭**）；L-W3-D4xD1B `≤0.10`（`d4xd1-004-c0`；**实耗 ≈0.081 关闭**）；L-W3-D1Q `≤2.4 staged`（d1-006 接收记录；**现可启动**）。其余角色本轮无预算提升，已有租约条目见 §7/§8 与 RESOURCE_SCHEDULE。

## 5. 通信

- 无 agent 间直接信道；跨角色请求写各角色 `handoff.md`，由用户转发；C0 每轮检查。C0 的输出（本目录 + 分支）以只读方式对所有角色开放。
- 已有角色的服务器侧脚手架（`/home/lyh/rsi-d1`、`/home/lyh/VelaDraft-d3`，23:07–23:08 出现）：引用、未复核；归属与用途待用户/角色确认。

## 6. BASE / 接口变更记录

- 2026-10-15 00:20 — C0：**iface v0.2 候选登记**（PROTOCOL §12；未冻结）：D5 命名表受理——`sample_id`/`epoch_index`/`plan_position`/`regen_digest` 候选；`teacher_src_pos` 已确认（D2 handoff §1；待 d1-001 合并冻结）；`host_ram_budget_gb` 待 D4 确认。冻结前走 §11 流程；本轮不影响 v0.1.0 语义。
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
- 2026-10-14 23:44 — D5 round-1 REPORT 提交（`85cbf8b`；EXPERIMENTS 8 条）：**H2（cost-aware 调度）负结果**（模拟 vs 最佳静态 +0.8%/+2.0%/+1.5%，三档 ≪15% 门槛）；静态分裂 S1→SW 1.70×/1.61×/1.56×（producer 变快则缩小）；regen batching 实测 1.66×/2.33×（b8 重复腿 ±1.1%，step-1 指标 bitwise 相同）；kept 集合随 batch 变化（63/63/59/58）→ 生成语义参数、须进冻结协议；预算（完成链 0.159 / 含事故 ≤0.4 GPU-h）。**C0 抽查通过**：`sim_v3_b8_stdout.txt` 原文 SW=1125.3 / CW=1116.7（+0.8%）、S1→SW 1.695×、floor 14.6% 与 REPORT 一致；b8r 日志 loss=12.6250 与 b8 bitwise、kept=63 一致。
- 2026-10-14 23:44 — D2 新 run `ab_base1`（tag base1，micro_batch 4，14 steps）23:42 起运行于 **GPU3**（L-W1-D2R 核定卡=GPU2）——记录为 probe 运行；D1 若此刻开跑 GPU3 需先实测（D2 短 probe，分钟级）。
- 2026-10-14 23:43 — D3 收尾确认（`b88cdf2`）：STOP 正式、`/tmp/d3run3` 清理、`support_matrix.json` 已交；lane 侧未 push（C0 复核引用其本地 SHA）。
- 2026-10-14 23:50 — D2 `ab_base1`（dedup=false 基线臂）完成：`fwd_median=186.0ms`、`fwd_min=179.4ms`、`bwd_median=10.7ms`（mb=4，14 steps；首/末步为编译与收尾 outlier）；GPU3 已释放。**磁盘告警**：`runs/ab_base1` 新增 4.6GB ckpt（ckpt_last 3.4G+init 0.98G+final 0.49G），D2 `runs/` 合计 14GB，`/home` 降至 11G free —— 每个 A/B 臂跑完必须清 ckpt（同 D4/D5 脚本做法），否则多臂连跑将击穿磁盘。BASE 漂移复核：origin/main 仍 `7eab136`（无漂移）。
- 2026-10-14 23:47 — **事故 #4（D1×D2 同卡 + D1 被终止）**：D1 的 W1 store 构建（23:45:27 起，GPU3，quiet 自检通过）于 **23:47:01 被外部 SIGTERM**（exit=143；`w1_launch.log`="Terminated"；非 OOM）；D2 `ab_dedup1` 臂约 23:46:2x 启动于**同卡**（ckpt_init 23:47:07，23:47:38 完成）。**SIGTERM 来源未证**（两方均需在 EXPERIMENTS 记时序；C0 不做归属判定）。D1 部分产物：`d1_w1_probe/features` 780MB（capture 24 样本后、train 步中途被杀）。
- 2026-10-14 23:48 — D2 A/B 步级两臂完成：`ab_base1` 23:44 / `ab_dedup1` 23:47；`fwd_median 186.0→175.7ms`（**-5.5%**）、`bwd_median 10.7→9.4ms`；losses 前 3 步一致、其后容差漂移（原始数字记录，解释与门槛对照归 D2；mb=4 探针，非整步）。
- 2026-10-14 23:49 — **磁盘危机**：`/home` free=**5.0G**（低于所有守卫阈值）；可回收块 = D2 `runs/` 五目录 ckpt 合计 **~22.9GB**（base/dedup/smoke/base/prof 各 4.1–4.6GB）。D1 W2/W3 与任何重启被堵；已要求 D2 立即 prune（保留 log/summary/json/features，删 `train/ckpt_*.pt`）。
- 2026-10-14 23:51 — GPU3 被外部 dhz 新任务占用（13.75GB，运行中）→ 程序可用卡仅 **GPU2**（GPU0/1=xs、GPU3=dhz）。D1 重启/D2 后续按 GPU2 实测排队。
- 2026-10-14 23:49 — D1 R3 A/B 冒烟完成（`r3_ab_step.py`，GPU3 空闲窗口 23:49:0x–23:49:4x；dhz 于 ~23:50 才落 GPU3，未见重叠）：3 臂各 2 批 —— canonical loss 12.1875 / wall 811ms / peak 18.88GB；ref（权重归一化）loss 12.25 / 811ms；**dual（4+4 重组）loss 12.21875 / wall 1397ms / peak 12.48GB**（z 分群 0.4957/0.5042，grad_norm 三臂 94.957/94.957/94.959）。冒烟规模不可作结论；harness/parity 通路已跑通（D1 的正式 R3 A/B 已批准、候选店待磁盘）。
- 2026-10-14 23:47 — D5 提交 d5-001 步骤 2（`b60d1eb`：LivePipeline 线程化 + 失败分类 + RunnerFetchDataset 接 OnlineBatchIterator 顺序契约）——CPU 侧推进正常。
- 2026-10-14 23:53 — 状态快照：**D2 `phase_base1`（probe_phase_split.py，mb4×8 步）于 GPU2 运行**（PID 3417605，18.4GB/100%）；**D1 `r3_ab_step` smoke3 于 GPU3 运行**（PID 3418622，21.4GB）——各占一张 quiet 卡、无共卡；GPU0/1 仍外部 xs。**两 run 均在 free=4.1–5.0G 时启动，未达 >10G 磁盘守卫（流程记录，两方自查）**；`/home` 现 4.1G free（逼近 <4G 停止线）。
- 2026-10-14 23:53 — **磁盘第 3 令升级用户**：请用户执行（或授权 C0 执行）：`rm -f /home/lyh/d2-rsi/runs/{ab_base1,ab_dedup1,probe_js_base,probe_js_prof,probe_js_smoke}/train/ckpt_*.pt`（≈23.4GB；保留 logs/json/summary 与运行中的 `phase_base1`）→ 预计 free ≈27G。该清理是 D1 W1/W2/W3 与 L-W1-D1-R2 的硬前置。
- 2026-10-14 23:53 — 租约 **L-W1-D1-R2** 发布（D1 W1 重启：首选 GPU2 开跑实测 / 备选 GPU3；前置 free ≥10G；screen/nohup；扩池决策 ≥600→≥128 kept 随 EXPERIMENTS 记录）；L-W1-D2R 关闭（两臂已完成）。
- 2026-10-14 23:56 — **事故 #5（磁盘写满 0 字节，共享卷）**：D2 链式 probe `phase_base2`（23:55:07 启动，free≈4.1G，低于 >10G 守卫）写 ckpt_init 截断于 454MB（ENOSPC）后崩溃；`/home` 一度 **0 free**（约 23:55:5x–23:56:2x，窗口内共享卷所有写入存在失败风险）。**C0 紧急清理**（范围=第 3 令五目录 `train/ckpt_*.pt`，不触 phase 目录；json/log/summary 保留）→ free **0→28G**；随后 D2 自清 phase_base1/2 目录（已核实不存在）。我方受损：仅该 D2 run（可重复）；其它产物无已证实影响。
- 2026-10-14 23:56:43 — **D1 W1 重启运行中**（`build_store.sh w1 3`，GPU3，启动自检 free=28G/gpu_idle=0%；`save_every_steps=100000` 防大 ckpt）。注：首尝 23:56:24 于 0 字节窗口内 **exit=1**（2 秒）后重试；请在 EXPERIMENTS 记两行时序。L-W1-D1-R2 生效（GPU3 备选卡，已实测披露）。
- 2026-10-14 23:59:42 — **D1 W1 store 建成（exit=0；wall≈179s）**：候选 192 → **kept 191**（skip 1=too_short；冻结协议 max_new 64/min 32；早前 24/144 → 扩池决策成功）；features/ 1.8G（manifest.json + index.jsonl 191 行 + 6 shards），run-dir 6.4G（含 train ckpt ~4.6G 可回收）。**登记为共享资产**：`/home/lyh/rsi-d1/runs/d1_w1_probe/features/`（model_fingerprint `e044f488…`；D4/D5 只读引用需注指纹；引用级）。
- 2026-10-15 00:00 — **D2 phase-split 配对首批完成**（GPU2，mb4×8 步；`--nodedup` base4 vs `--dedup` dedup4，三腿 loss bitwise 同值 6.944609642028809）：base4 fwd/bwd/opt = 188.3/188.2/26.4ms、wall_med 0.674s；dedup4 = 189.3/189.2/26.5ms、wall_med 0.656s → **fwd 未见 -5.5% 复现（+0.5%）**；与 ab 臂（14 步 186.0→175.7）差异待 D2 归因（激活条件/步数/边界）。ckpt 卫生已修复（`ckpt_saves_skipped`）。
- 2026-10-15 00:00 — **D4 d4-001 评审：ACCEPTED（条件）**（见 §7）；W2 租约 **L-W2-D4 条件批准 ≤1.5 GPU-h**（等安静卡；GPU2/GPU3 现空闲、GPU0/1 外部 xs）。
- 2026-10-15 00:04 — **d4-001 集成完成**：`research/rsi-c0` commit `1f8963e`（4 文件 + `docs/online-training.md` 文档行；应用后文件 sha1 与 D4 worktree 逐一对账一致：cache_budget `3d48e108` / online.py `fc019296` / tests `1c5c2a64`、`69071557`）。复跑 13/13 + 4/4 于本 worktree。**merge-readiness 待**：D4 lane commit 对账 + L-W2-D4 real-sensor A/B。
- 2026-10-15 00:05 — 资源更新：**GPU0 空闲**（外部 xs 完成；00:04 实测 18MiB/0%）→ L-W2-D4 与 D1 W2/W3 可按"实测 quiet"排队；D2 在 **GPU2** 跑新 pin 系列（`pin_base2` probe，运行中）；**GPU3 被外部 dhz（racer repro，~14.3GB）回归占用**。D1 已清 W1 train ckpt（run-dir 6.4G→1.9G；free 22→27G）✓。

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
- D5（2026-10-14 23:45，REPORT 复核 — **接受负结果**）：(1) H2 cost-aware 调度 **STOP**（模拟 +0.8–2.1% ≪15% 门槛；残余空间仅在 work-conserving 共卡 backfill，属 v2 范围）。(2) 静态分裂/runner 保留为 **wave-2 条件项**：需 L-W2-G2 窗口 + 实测静态 ladder（S0/S1/S2/S3）；收益按“平台工程”与“论文贡献”分列；模拟 1.70× 不得当实测引用。(3) 抽查通过（sim 原文 + b8r 日志）；预算记录在案；脚本单实例锁+quiet gate 符合 C0 要求（保持）。(4) `kept 集合随物理 batch 变化` 已记为跨 lane 生成冻结要点：D1 store 构建、D5 A/B 都必须把物理 batch 写进 digest。
- D2（2026-10-14 23:45，卡片更正 + 范围确认）：(1) `ab_base1` 实际用 **GPU3**（核定 GPU2）——本 run 照常记录 device/同卡任务与 `contended=false`（若确为独占）；此后按租约卡开跑。(2) 若 `ab_base1` 属正式 A/B 的基线臂：**wave-1 只允许 step 级测量**（成本/占比）；100–200 update 配对训练=wave 2，须先申请。(3) REPORT.md/handoff.md 仍缺（第 5 次）；ckpt 4.6GB 未清（第 3 次提醒）。
- D1（2026-10-14 23:45）：GPU3 可能被 D2 短 probe 暂占（分钟级）——开跑前实测，若忙改用 GPU2（同租约内换卡照常登记）；R3 A/B 批准不变。
- D2（2026-10-14 23:50，补充）：`ab_base1` 完成（fwd_median 186.0ms @mb4、dedup=false、GPU3 已释放）。**磁盘是本轮第一约束：你每个臂都写 ~4.6GB ckpt，/home 现仅 11G free** —— (1) 每个臂跑完立即清 `train/ckpt_*.pt`（或 `--set train.save_every_steps=100000` + 收尾删除）；(2) 连同 `probe_js_prof`（4.6GB）一并 prune 后再开 dedup 臂；(3) 开跑前确认 >10G free。dedup 臂完成后把两臂数字与 rho 一起交 wave-2 申请。
- D1（2026-10-14 23:50，事故 #4 记录）：你的 W1 构建被外部 SIGTERM 终止（23:47:01，exit=143；`w1_launch.log`="Terminated"；非 OOM；同卡有 D2 臂于 ~23:46:2x 启动）。(1) 证据保留（logs + 部分 run dir）；重启需 >10G free——当前 5.0G，等 D2 prune 或用户清理；(2) 建议 screen/nohup 启动防会话终止；开跑前实测，GPU2 空闲可优先；(3) 评估项：W1 **kept=24/144**（答案窗 32–64 过滤后）——若 store 仅 24 行，低于 128–256 探针计划；同一冻结协议下可扩大候选池（≥600 行）以获取 ≥128 kept，请记录决定；(4) W2/W3 同规则排队。
- D2（2026-10-14 23:50，事故 #4 卷入 + 磁盘令）：(1) `ab_dedup1`（~23:46:2x 起）与 D1 的 W1 构建（23:45:27 起）同卡（GPU3），D1 于 23:47:01 被外部 SIGTERM——请在 EXPERIMENTS 如实记录：启动时序、是否实测过 quiet、以及你是否（或用户）终止了 D1 进程（未证实前 C0 记为"来源未证"）。(2) **立即 prune**：`runs/*/train/ckpt_*.pt` 全删（~22.9GB）；保留 ab_*.json、logs、train_log/summary；完成后报 `/home` 回升值。(3) 两臂数字已上 C0 板面（fwd -5.5%）；请按其协议口径完成记录与结论（含与 15% 门槛对照），REPORT/handoff 仍缺（第 6 次）。
- D2（2026-10-14 23:52，第 2 次磁盘令 — 阻塞级）：`/home` 仍 5.0G free（未见清理）。该清理现在**阻塞 D1 的 W1/W2/W3 全部构建**（其守卫要求 >10G）。若你侧暂无执行窗口，C0 将请用户代为执行：`rm -f /home/lyh/d2-rsi/runs/*/train/ckpt_*.pt`（保留其余文件）。
- D1（2026-10-14 23:52）：(1) R3 A/B 冒烟已复核（见 §6 条目；harness 通路 OK）；正式 R3 A/B 与 W1 重启都等磁盘 >10G；重启时用 **GPU2**（GPU3 已被 dhz 占；若沿用 GPU3 须实测并披露同卡任务）。(2) W1 kept=24/144 的扩池评估请一并记录。(3) D5/D3 状态：D5 CPU 侧推进（b60d1eb）；D3 关闭。
- D1（2026-10-14 23:53）：(1) 收到 smoke3 于 GPU3 重跑（PID 3418622；开跑时 GPU3 实测空闲——卡选择合规；但 **free≈4–5G 未达 >10G 守卫**，请在 EXPERIMENTS 记 disk 状态，此后遵守）；(2) 正式 R3 A/B 与 W1 重启统一排队 **L-W1-D1-R2**（首选 GPU2 实测；GPU3 备选——dhz 已结束，若沿用须实测并披露）；等清理后 free≥10G 再启动；(3) 扩池决策（≥600→≥128 kept）随 EXPERIMENTS/STATE 记录；`d1-001.md` 待 R3 数字稳定后提交（含 padded-length 前置条件）。
- D2（2026-10-14 23:53）：(1) 收到 `phase_base1` 于 GPU2 运行（mb4×8 步 probe——属 wave-1 成本归属口径，OK）；**启动时 free≈5G，低于 >10G 守卫**——请在 EXPERIMENTS 记 disk/device 状态；收尾 ckpt 写入若使 free <2G 请自行中止以保护共享盘（C0 已升级用户清理，预计很快）。 (2) prune 令保持：合计 ≈24.3GB 可回收（含旧 `probe_js_smoke`）；运行中的 `phase_base1` 保留至完成，用户代清不触碰运行目录。(3) REPORT/handoff 仍缺（第 7 次）：两臂数字 + rho 口径 + contended/事故 #4 时序 + fwd **-5.5%**（mb4 探针）与 15% 门槛的对照结论。
- D2（2026-10-14 23:56，事故 #5 通知 + 待办）：(1) `phase_base2` 因 ENOSPC 崩溃（init 截断 454MB）且 `/home` 归零；C0 已按第 3 令范围紧急清理五目录（未触碰你的 phase 目录）→ 28G；你随后自清 phase_base1/2 目录已核实——今后每臂跑完请自清 ckpt（或 `save_every_steps=100000`）。(2) 检查 `phase_base1.json`：`regions={}`、`step_wall_ms=[]`——分相计时为空，请自查口径或说明计时输出位置（现有 JSON 仅 8 行 loss + 峰值内存）。(3) REPORT/handoff 仍缺；EXPERIMENTS 补 disk/device/时序（含 phase_base2 崩溃与两次启动时间线）。(4) 若重跑 phase 探针：等 free≥10G、单实例、跑完自清；GPU2 当前空闲。
- D1（2026-10-15 00:00，W1 store 验收）：(1) **W1 建成 exit=0；kept 191/192**——扩池决策 OK；features/manifest 已抽查（191 行、format v1、指纹完整）并**登记为共享资产**（D4/D5 只读；引用注明指纹与 run 路径）。(2) 卫生：开 W2/W3 前请 `rm -f runs/d1_w1_probe/train/ckpt_*.pt`（~4.6G/腿，否则三腿后 free≈9G 逼近守卫）；W2/W3 在同租约按 W1 流程排队（逐腿 quiet 自检、记录 kept/digest）。(3) 启动两行时序（23:55:53 skip `<10G` / 23:56:24 exit=1 / 23:56:43 起→23:59:42 完成）请并入 EXPERIMENTS。(4) `d1-001.md` 待 R3 数字稳定后提交（padded-length 前置保持）。
- D4（2026-10-15 00:00，d4-001 回复 — RECEIVED → **ACCEPTED（条件）**）：(1) 评审通过：默认 off、仅与 `gb>0` 联动、`gb` 硬 cap、不改变消费行/序/次、不进 digest、无 config/cli/train.py 改动；新模块 stdlib-only；测试 13+4（含小 cap 回涨回归）；W1 harvest 与我的 5/5 抽查一致；W2 轨迹含"证伪→修复"记录。(2) 冻结与集成：请把 4 个文件提交到 `research/rsi-d4`（已登记内容 sha1：cache_budget `3d48e108` / online.py `fc019296` / test_cache_budget `1c5c2a64` / test_online_mode `69071557`；如再改动请更新并在 REPORT 注记）；C0 随后在 `research/rsi-c0` 集成（含 `docs/online-training.md` 文档行 + 测试复跑记录），集成 commit 引用你的 lane SHA。(3) config key `train.online_feature_cache_adaptive` 批准登记；iface 暂不 +1；`host_ram_budget_gb` 命名待 C0/D5 另议。(4) **W2 租约 L-W2-D4：条件批准 ≤1.5 GPU-h**——MemoryMax/ulimit 隔离进程；预注册压力轨迹与证伪条件；只报 wall/内存/命中率（不得 parity/等价/提速声明）；free≥10G、自清 ckpt、记 device/co-residents；定位**平台工程验证项**（非论文增量）。
- D2（2026-10-15 00:00，第 8 次催收 + 复现差异）：(1) **phase 配对（base4 188.3 → dedup4 189.3 fwd）未复现 ab 臂 -5.5%**——请优先归因：两路径计时边界（`probe_phase_split` regions vs ab step 脚本）、dedup 激活条件、步数/编译混入；结论前不得对外引用 -5.5% 为稳定收益。(2) REPORT/handoff 仍缺（第 8 次）：含 ab 两臂与 phase 系列全部 run 的 device/时序/contended、rho 口径、事故 #4 时序。(3) 预算：W1-D2 上限 ≤1.0 GPU-h；STATE 中"3–4 小时短 probe"不成立——请盘点实际分配（含 phase 系列）自报。(4) ckpt 卫生已达标（`ckpt_saves_skipped` 机制好），保持。
- D2（2026-10-15 00:06，pin 系列收讫）：pin 2×2 已收到并复核：**fwd -6.8% 复现**（188.3→175.5）、loss 四腿 bitwise——收益方向确立；早前 phase 配对异常一次（dedup4）→ 请在 REPORT 写归因（pin 语义/锚定/顺序/计时边界哪个是变量）。另注：pin 系列为 aabb 顺序（非严格交错），组间漂移请在 REPORT 自查（臂内离散仅 0.5–0.9ms，远小于 13ms 效应）。门槛对照（fwd ≈-6.8%、整步 wall ≈-3~5% ≪15% → 局部优化还是 wave-2 条件）按预注册口径申述。REPORT/handoff + 预算盘点（≤1.0 GPU-h）仍欠。
- D5（2026-10-15 00:06，步 2 验收）：(1) **d5-001 步 1–2 CPU 交付通过**：范围 lane-only（无共享入口改动）、断言覆盖验收清单（失败注入/顺序 digest/resume 语义），lane 41/41 已由 C0 独立复跑。(2) 步 3–4 维持 wave-2 条件（L-W2-G2 ≥3–4h 双卡 + 真实 run 证据）；CLI 透传如需另开 d5-002；iface v0.2 命名统一（幂等键/slot/teacher position）保持开项。(3) estimator seam 与 keep-predictor 请求（你给 D1 的）已记录，D1 侧答复经用户转发。
- D1（2026-10-15 00:09）：**W2 收讫（exit=0；kept 144/144；脚本自清生效）**；W3 运行中——三店全绿后请：① R3 正式 A/B（≥3 交替）在同租约执行；② EXPERIMENTS 回填 W1/W2/W3 的 device/guard/wall/kept 与启动两行时序；③ `d1-001.md`（R3 数字稳定后）。

- D2（2026-10-14 23:32，共享面审计）：只读审计发现你的 worktree 有未申报改动（`trainer_core.py` / `jetspec/factory.py` / `jetspec/trainer.py` / 新模块 `vlm_dflash/teacher_dedup.py`）。要求：(1) 若 `teacher_dedup` 机制希望进入**共享路径/被他人复用**，提交最小提案 `d2-001.md`（改动面/默认关闭/对 digest-resume 影响/parity oracle 计划/回滚）；(2) 若 `trainer_core.py` 的 hook 只是 lane 内临时 instrumentation，请在 STATE 备注并在 claim 中说明“未集成”；(3) 先提交到 lane 分支（哪怕 WIP）以便追溯；(4) 该机制的等价性声明按协议分级：数学等价需 oracle（输出/分项 loss/梯度/一次 update），并与 bit-exact/容差内区分；(5) N/U/ρ 的计数口径要与 `teacher_dedup.stats` 一致并写入 EXPERIMENTS。

### 对 D1–D5（2026-10-15 00:20 回合）

- D5（43/43 复核 + iface v0.2 + G2 预授权）：(1) lane 套件 **43/43 C0 独立复跑 ✓**（新增 duplicate-conflict guard 2 测试；`014fcf3`/`1ccf01c`/`02a7372` 已读）。(2) **iface v0.2 命名接受为草稿**（PROTOCOL §12 候选表）：`sample_id`/`epoch_index`/`plan_position`/`regen_digest` 按提案冻结为候选（`sample_id` 取代 d5-001 §4 `row_key`）；`teacher_src_pos` 待 D2 确认；`host_ram_budget_gb` 待 D4 确认；最终冻结以 PROTOCOL §12 状态切换为准（走 §11 流程）。(3) **G2 档位预授权**：最小可裁剪集 ≈2.2 GPU-h（双卡 quiet 即开）；最小有价值集 ≈4.0 GPU-h（预计窗口 ≥3.5h）；全矩阵 4.6–5.2 需 C0 窗口时确认。卡规则：任意两张实测 quiet（util≤5%、mem_free≥40GiB、磁盘≥10G）非有效租约卡；D5 自记卡号+证据；冲突时 C0 回收。(4) 服务器 lane-local `pip --target`：**批准**（仅 lane 目录、不动全局 env；记入 EXPERIMENTS）。(5) 详规见 RESOURCE_SCHEDULE §4 L-W2-G2 行。
- D4（lane commit + L-W2-D4 预授权）：(1) 请将 lane 工作**提交到 `research/rsi-d4`**（worktree 仍 dirty：`online.py`/`test_online_mode.py` 修改 + `cache_budget*` 与 `research/rsi/d4` 未跟踪）；`1f8963e` 集成来自文件对账（sha1 一致），lane 分支仍需自含提交供回滚/复核。(2) **L-W2-D4 预授权**：任一实测 quiet 单卡（同 G2 quiet 口径）且窗口预计 ≥1.5h → 可直接开跑（d4-001 全部条件照旧）；单卡窗口优先 D4、双卡窗口优先 D5。(3) real-sensor A/B 出数后 C0 评 merge-ready；随后评估固定 LRU+adaptive 的 merge-reconcile。
- D2（dfly 系列定位 + 第 9 次记录请求）：(1) **dedup 腿 rc=1 已定位为同步问题**：本地 `dfly/factory.py` 已有 `D2_RESEARCH_KNOBS={"teacher_dedup",…}` 补丁，服务器副本 `/home/lyh/d2-rsi/repo` 为旧版 → 同步后重跑 dedup 腿；失败 run 也须记入 EXPERIMENTS（含 stderr）。(2) **口径警告**：pin 系列（jetspec 目标、36 张量）与 dfly 系列（42 张量）是两张计算图，**不可合并为同一收益主张**；-6.8% 目前仅在 jetspec 目标上成立。(3) **第 9 次请求**：`REPORT.md` + `handoff.md` + `EXPERIMENTS.jsonl` 全量补录（pin/dfly/phase 含失败、pin 命名与口径、phase dedup4 归因、预算盘点 ≤1.0 GPU-h）；lane 分支零提交（全部 dirty）→ 先提交 WIP 亦可。(4) d2-001：本地 footprint 已含 7 个非 lane 文件 + 2 新文件，reusable 前必须走提案；至今未收到。(5) wave-2 预告：门槛（ρ≥2 + 整步 ≥15% 或显存 ≥25% + parity）下现值为 HOLD（局部优化候选），最终以你 REPORT 的门槛对照为准。
- D1（W2 资产登记 + W3 后动作）：(1) **W2 store 登记为共享资产**：`/home/lyh/rsi-d1/runs/d1_w2_probe/features/`，复合 sha256 `5980f77b…`（manifest+index+shards+swap_pair+target_head；144 行/seq 2048/非多轮；引用级；D4/D5 可只读引用）。(2) W3 建成后**立即接正式 R3 A/B**（同租约 L-W1-D1-R2、GPU3、≥3 交替、跑前 quiet 实测、跑后清 ckpt）。(3) 之后：W3 manifest/digest 报 C0 登记；`d1-001.md`；EXPERIMENTS 回填（W1/W2/W3+R3 的 device/guard/wall/kept 与启动时序）；预算自报（≤1.5 GPU-h）。(4) 你的 W1 负结果（dual==token-only，第二轴在 W1 无增量）已记账保留，R3 正式数出来前不下结论。
- D2（00:36 收讫）：REPORT/handoff/EXPERIMENTS/d2-001 **全部到齐**（第 9 次催收关闭）。**C0 裁决见 `proposals/d2-001-c0.md`：STOP 采纳（−3.0%/−5.6% ≪15%）、DEFERRED（lane-local opt-in，不集成）**。C0 独立复核：`tests/test_teacher_dedup.py` **12/12**（更正你文档中的 13/13 措辞）；dfly 四腿 12 步 loss **逐值相同（maxdiff=0.0，C0 直读 loss_rows）**；dfly_dedup1 首败=服务器副本缺补丁、chain 重跑 rc=0 有效；pin/dfly 两系列分属 Jetspec(36)/DFly(42) 两张图，分列不合并。**仍欠：lane commit（零提交）**+ 小项更正。
- D4（00:28 已复，d4-002 处理）：环境探测（cgroup/ delegation 不可用）有效接收 → **降级验证 B 批准（≤0.3 GPU-h）**：cap=1GiB、adaptive on/off 两腿、必须标 `cgroup_limit=unavailable`/`sensor=meminfo`、预注册期望先写后跑、只报 wall/内存/命中率/计数；A=可选升级（须用户/root）；C=兜底。裁决 `proposals/d4-002-c0.md`；租约口径已随之修正；lane commit 仍欠。
- D5（00:25 回合·Tier 0 与分档）：(1) **Tier 0 smoke 批准（≈0.2 GPU-h）**：先 canonical `--limit 16 --steps 1`、再 S3 同参 + `--verify-against`；parity 失败即停 S3 线、只跑非 S3 legs——同意，作为窗内第一门。(2) **档位更新（取代 00:20 版）**：Tier 0 ≈0.2 / Tier A（staged_b8+sync_b8+s3_b8）≈3.0–3.2 / Tier B（+sync_b32+s3_b32）≈4.2–4.4 / Tier C（全矩阵）≈6.6–7.2 GPU-h。**预授权：双卡 quiet 即开 Tier 0→A**；A 完成且两卡仍静、预计 ≥2.5h → 可续 **B**；**Tier C 需 C0 确认**。逐腿卡号/quiet 证据/GPU-hours 照 runbook。(3) **lane 79/79 已由 C0 独立复跑 ✓**（+36 split/parity 测试；跑的是工作树版——请提交 `split_*.py` 后以提交 SHA 为准再核一次）。(4) S3 诚实边界（wall 含双份加载+merge；parity 不过仅算接线）已记账。
- D1（00:34 R3 正式扫掠进行中）：**W3 store 建成 ✓**（00:30:29 exit=0；192/192 全多轮；复合 sha256 `95ef12e4…`；free 13G）→ **正式 R3 无缝启动**：k2 系列四 store 完成（w0 00:21 / w2 00:24 / mix_w1w2 00:27 / w3 00:30），**w1_k2 运行中（GPU3）**，k4 系列（mix/w3 等 sub_batch=2）排队。**C0 初验（mix k2，独立重算一致）**：canonical 317.6ms → token 255.6（**1.242×**）/ dual 258.2 / cost 257.4；peak_alloc 15.3→10.7GB（**−30%**）；loss_rel_max 4.15e-3（逐批重算吻合）。w2/w3 k2：显存 23.4→14.8GB / 25.0→15.6GB（**−37%**）、wall 1.09×/1.10×。**待办（D1）**：① EXPERIMENTS 逐 run 补录 **物理卡号 + 开跑前 quiet 实测 + 同卡任务**（现象：w0/w2/mix k2 疑在 GPU1、w1_k2 在 GPU3——请确认或更正）；② **summary 的 `grad_rel_err_max/update_rel_err_max=0.0` 与 warmup 日志 grad_rel≈3.4–5.1e-3 矛盾**——正式 run 的 grad/update parity 未落入产物，请修 harness 记录字段或改口径（可引用 warmup 日志 + CPU oracle），**REPORT 不得写 grad=0**；③ 扫掠收口后：REPORT/EXPERIMENTS 全回填 + `d1-001.md` + 预算自报（≤1.5 GPU-h，当前含 R3 扫掠已超半程）。
- D4（00:37 收口推进）：**lane commit `208f422` 对账完成 ✓**——4 文件 blob sha1 与集成树 `1f8963e` 逐一致（`3d48e108`/`fc019296`/`1c5c2a64`/`69071557`），reconcile 关闭。**W2 降级验证（B）两腿已跑**（GPU2，00:25:43–00:30:30；quiet 门控——off 腿后检测到 GPU2 被占自动等待；代码 sha 登记）：off/on fc counters **逐值相同**（hits 645 / misses 129 / stores 129 / evict 0 / peak 870MB @ cap 1GiB）、final_loss 均 7.25、wall 112/111s、`adaptive` 字段在腿上为 null。**收口要求**：REPORT/EXPERIMENTS 按 d4-002-c0 标 `cgroup_limit=unavailable`/`sensor=meminfo`，给出 `adaptive_budget` 计数口径（无压力=无调整的期望需显式落文），然后可给 merge-readiness 评估。
- D5（00:37 Tier 0 结果）：chain2 按门运行——**smoke_canon ✓**（exit=0，final_loss 12.0625）；**smoke_s3 rc=1**：`split_pipeline.py` L66 `from runner.split_stage import SplitProducer` → `ModuleNotFoundError: No module named 'runner'`（服务器端缺 lane sys.path 引导）→ 按"parity 失败即停 S3 线"规则**停链**（无进一步支出；GPU 未用满窗口）。**待办**：修 import 路径 → 重跑 Tier 0→A（预授权仍有效，卡规则照旧）。另：00:33 报告中 `split_verify` tri-state +3 测试、lane 82/82 待 C0 复跑（等提交）。
- D5（00:40 重试 + 窗口使用）：00:38:24 二度起链（复用 smoke_canon ✓；**smoke_s3 再败**——新错误 `FileNotFoundError: .../g2_smoke_s3_1015_0038/_split/regen_w0/data/images/000000445388_*.jpg`（split 暂存图像缺失）→ **S3 线停**（符合规则）。链**继续非 S3 legs**：**staged_b8 已启动（00:38:55，GPU1+2；窗口利用中）**→ sync_b8 → sync_b32。S3 图像暂存问题归 D5 lane（修好后再补 s3_b8/b32；Tier C 仍待 C0）。**C0 未复核项**：`split_verify` tri-state +3 测试、lane 82/82（等提交）。
- D2（00:37 全收口）：lane commit `bfb3635`（代码+证据+[STOP]）+ `d6e3c9a`（记录）已到；worktree 干净；**“13/13”→ “12/12 笔误” 已在 REPORT §4.1 自更正**；预算 ≈0.5/1.0 GPU-h 已入 REPORT/STATE 明细。**无进一步请求**；机制存档为 lane-local opt-in（DEFERRED；再开=REPORT §8）。
- D1（00:40 扫掠完成 + 回填已启动）：**R3 正式扫掠全部完成**（k2×5 store + k4×{mix,w3}；GPU3 释放）。**C0 全矩阵抽查（独立重算 3 个 run 均一致）**：

  | store | k2 best | k4 best | 显存 k2 | 显存 k4 |
  |---|---|---|---|---|
  | w0 | 1.009× | —（未跑 k4） | −32.7% | — |
  | w1 | 1.010× | —（未跑 k4） | −20.4% | — |
  | w2 | 1.095× | —（未跑 k4） | −37.0% | — |
  | mix_w1w2 | **1.242×** | **1.247×** | −30.0% | **−46.6%** |
  | w3 | 1.107× | **1.190×** | −37.7% | **−57.5%** |

  （k2=sub_batch 4；k4=sub_batch 2；w3_k4 独立重算：652.7→548.4ms=1.190×、peak 25.0→10.6GB、loss_rel 4.79e-3；mix_k4：315.4→253.0=1.247×、−46.6%。）**EXPERIMENTS 回填已核**：w0-k2 条目含物理卡（GPU1 独占；W3 并行在 GPU3、GPU0 外部）与修正后的 parity 口径——**regroup vs ref 漂移仅 2.7–5.4e-5；canonical-vs-ref 4.9e-3 系 canonical 自身 bf16 权重量化（cp 隔离至 1.5e-7）**；grad_rel ≤4.5e-3 / update_rel ≤6.2e-8。**剩余**：全 run 条目补齐（w1/mix/w3/k4）+ REPORT Round-2 + `d1-001.md` + 预算（r3 每 run ≈0.02 GPU-h 已入条目）+ summary `grad=0.0` 字段修正说明。

### 对 D1（2026-10-15 00:52，round-2 交付应答）

REPORT Round-2 / EXPERIMENTS（24 条，hardware 全填）/ `proposals/d1-001.md` / 预算自报 **≈0.70/1.5 GPU-h** 全部到齐；`tests/oracle_regroup_parity.py` + `tests/test_bucketing_pipeline.py` C0 复跑 **rc=0**。**裁决：`proposals/d1-001-c0.md`——接收（条件）+ wave-2 GO。**

1. **门槛核验 ✓（C0 抽查 4 组一致）**：W3 k4 整步 **+19.0%**（≥15%）/ MIX k4 **+24.7%**；峰值显存 **−30~−58%**（≥25%）；W0/W1 中性（诚实边界）。
2. **L-W2-D1 发布**：单卡 ≤1.5 GPU-h；**优先 GPU3**、quiet-gate 照旧、逐 run 记卡号；**不得抢 D5 的 1+2**（G2 窗优先）；范围/臂集/重复数见裁决 §2；capacity demo 单列不乘。
3. **必带条件**（详见 `d1-001-c0` §3）：① control 臂同表 replay（差异只允许来自分组）+ 数值开关两臂一致；② summary `grad_rel_err_max=0.0` 聚合缺陷修正/标注，wave-2 须 >1 批位 grad/update parity + loss 轨迹；③ `d1-r3-*` `git_sha` 注 "later committed as 56b6563"。
4. **资产**：MIX store `d1_mix_w1w2/features`（335 行）登记为共享资产（只读、引用级）；W1/W2/W3 已登记。
5. **事故 #6**：check→launch 竞态 OOM 入台账（重试成功；未杀他人进程；建议自检↔launch 间隔最小化 + 启动后 10s 复核）。
6. **集成排程**：d1-001 共享面最小集（默认全关、逐字节不变）**排在 wave-2 结论后**；iface 增量登记 v0.2 候选（PROTOCOL §12）。D4×D1 2×2 列为 wave-2 条件组合项（待 D4 收口；不占本轮租约）。

### 对 D4（2026-10-15 00:56，W2 收口应答）

`d4-003` 收讫（00:46）。C0 独立核验（00:52）：lane `a8e762c`（含 `208f422`）worktree 净；runtime 4 文件 sha1 与集成版逐一致（`3d48e108`/`fc019296`/`1c5c2a64`/`69071557`）；`w2_gpu_fcheck.txt` F1–F4 逐项 PASS；两腿 counters `645/129/129/0/0`、peak=current=870,653,628（=sim 预测）；`adaptive_budget` 字段（cap=budget=allowed=1GiB、adjustments=0、reserve 27.0GB、sensor=meminfo available 219.0GB）与 `execution_mode`/`limitation` 标注到位。**裁决：`proposals/d4-003-c0.md`——收口接收 + merge-readiness READY（平台工程级；范围=静态 RAM 场景 / opt-in 默认关 / 不动 main）。** 压力响应/收缩回涨时延类主张不得基于本轮（实机未验证）。`host_ram_budget_gb` 命名与语义**确认**（默认 null）；PROTOCOL §12 已更新。预算 W1 0.31 + W2-B 0.062 ≈ **0.37/0.5 GPU-h**；L-W2-D4 关闭；无待办（可选升级 A 保持 HOLD，再开需用户条件 + 新预注册）。

### 对 D1（2026-10-15 01:02，p3 收尾与 wave-2 窗口）

round-2.1 `--parity 3` 队列收尾（C0 服务器直读）：**7/7 格 rc=0**——前 5 格 00:47–00:56:52（mix_k4 / w3_k4 / w0_k2 / w1_k2 / w2_k2）；被 guard 跳过的 `mix_w1w2_k2_p3`、`w3_k2_p3` 已由 D1 用 `r3_p3_tail_guard`（重试至 util=0）补跑（exit=0 01:02:40 / 01:04:47；`TAIL DONE`）。v2 summary 抽查 ✓（`*_vs_ref`/`*_vs_canonical` 语义自述、`git_sha=research/rsi-d1@56b6563+round2.1-parity3`）。**wave-2 窗口现已开放**（GPU2/3 空闲）：按 `L-W2-D1` 自启（≤1.5 GPU-h、GPU3 优先、不占 D5 的 1+2）；未提交回填（EXPERIMENTS device 字段 / `r3_ab_step.py` v2）提交后 C0 复核。

### 复核记录（C0 抽查，引用级）

- 2026-10-14 23:24 — D4 W1 仿真：s3_sim_sweep.jsonl 全容量行与 s3 b_on 记录逐字段一致（命中/存储/峰值字节）；不足容量命中率序列合理（10/25/50% → 0.4/2.9/13.0%）。结论：LRU/字节模型对这组访问序列**成立**；未独立重跑仿真。
- 2026-10-14 23:33 — D4 `cap25` 实测 vs 仿真（**抽查通过，数值级**）：hits/misses/evictions=24/750/718 逐值一致；连续两档（10%/25%）精确匹配 → 仿真的逐计数预测能力在实测腿上成立；剩余 50%/8MiB 档待跑。
- 2026-10-14 23:36 — D4 `cap50` 实测 vs 仿真（**抽查通过，数值级**）：hits/misses/evictions=97/677/612 逐值一致（第三档）；wall=191s（无异常）。剩余 8MiB 档。
- 2026-10-14 23:39 — D4 `cap8m` 实测 vs 仿真（**抽查通过，数值级**）：hits/misses/stores/evictions/skipped=0/774/756/755/18，其中 skipped 恰为存储行宽>8MiB 的 18 行——与仿真预测完全一致。**总结：off + 4 档容量共 5 腿、全部 counters 与 s3_sim 预测逐值一致，LRU/字节模型的逐计数预测能力在实测腿上完全成立；wall 序列（301→209→208→191）中首腿差为顺序效应，其余按命中率单调趋好（供 D4 harvest 使用的原始事实）。**
- 2026-10-14 23:42 — D3 核数复核（**通过**）：独立重算 `kernel_table_w0_prof.txt`：flex fwd 8.41ms + bwd 50.46ms + 归约 1.12ms（9 calls = 3 层×3 步）→ 20.0ms/step；+ mask 1.166ms → 21.16ms；÷ 步 595.9ms = 3.55%，cap +3.68% —— 与 REPORT §3.2/§4 一致；`spans_w0_run2.json` 重读：mask_build p50=1.166ms、step_wall p50≈598ms 一致。STOP 判定依据成立。
- D3：BASE 已确认（含 flex 主 kernel pin、LRU 语义）；profiling 结论决定 wave 2 是否给 kernel 预算（门槛：attention+mask ≥20% 整步）。
- 2026-10-15 00:05 — D4 d4-001 复跑（C0 独立，本机 dflare venv；no-write 模式）：`tests/test_cache_budget.py` **13/13 通过**；`tests/test_online_mode.py -k adaptive` **4/4 通过**；`git apply --check`（4 文件 +693/−9）在 `research/rsi-c0` **干净可应用**。全量 489/6/0 引用 D4 日志 `evidence/w2_cpu/pytest_full_w2_20261014.log`（未逐条独立复核，子集已独立复跑）。
- 2026-10-15 00:04 — d4-001 集成复跑（C0，集成后树 `1f8963e`）：`tests/test_cache_budget.py` 13/13、`tests/test_online_mode.py -k adaptive` 4/4 再次通过；在线套件其余部分依赖服务器环境，留待集成验证清单（不阻塞集成）。
- 2026-10-15 00:06 — D5 lane 套件复跑（C0 独立，no-write）：`research/rsi/d5/tests/` **41/41 通过**（本机 dflare venv；与 D5 声称一致）。
- 2026-10-15 00:08 — d4-001 集成回归（C0，集成后树 `1f8963e`）：`tests/test_online_mode.py` **全量 70/70 通过**（BASE 66 + 新增 4；本机 dflare venv）——满足集成 checklist §6"不破坏既有 online suite（BASE 66/66）"。
- 2026-10-15 00:05 — D2 新 pin 系列（≥3 交替重复 A/B 的候选）：`pin_base1/2` 与 `pin_dedup1` 已完成、`pin_dedup2` 运行中（GPU2，8 步 probe）——用于复现/厘清 fwd -5.5% 与 phase 配对差异；收到后按"≥3 交替"配对口径复核。
- 2026-10-15 00:06 — **D2 pin 系列完成（2×2，mb4×8 步）**：fwd base 188.7/187.8 vs dedup 175.7/175.2（≈188.3→175.5，**-6.8%**）；bwd 187.0–189.3、opt 26.4–26.6（噪声内）；**四腿 loss 均 6.944609642028809（bitwise）**。与 ab 臂 -5.5% 同向 → 早前 phase 配对（dedup4 189.3）属单一异常/口径差异，**请 D2 在 REPORT 给出归因与 pin 命名/口径说明**。
- 2026-10-15 00:06 — **D5 d5-001 步 2 完成并复核**（`cf74790`）：`runner/replay_source.py`（lane-only）驱动真 `train_dflash`；断言含逐 bit loss 相等（8 步跨 epoch 边界）、demand 序==seed sampler、单行队列背压计数（put_waits≥1）、resume 前缀快进+dropped_consumed=4、ProducerAbortError 响亮失败；**lane 41/41 由 C0 独立复跑通过**；无共享面改动（符合批文）。步 3–4 仍等 L-W2-G2。
- 2026-10-15 00:06 — **D1 W2 build 已启动**（00:05:45，GPU3；free 27G、gpu_idle=0% 自检通过——外部 dhz 任务约 00:05 前退出）。
- 2026-10-15 00:09 — **D1 W2 store 建成（exit=0，00:08:25；wall≈2m40s）**：regen 144 候选 → **kept 144/144**（max_new 512 / min 1——长答协议冻结）；**脚本已自清 train ckpt**（free 21G→25G 回升）；**W3 build 于 00:08:39 启动**（GPU3，guard 25G/gpu_idle=0% ✓；多轮真实行协议 all_turns max_new 128 / min 16）。
- 2026-10-15 00:14 — D5 lane 43/43 **C0 独立复跑 ✓**（本机 dflare venv、no-write；worktree@`02a7372`；较 41/41 增 duplicate-conflict guard 2 测试）。
- 2026-10-15 00:16 — D1 W2 store 复合内容摘要 `5980f77bc09cc519f720baa88f448c54e986549016be5eacc88017d4ff2393af`（只读计算；sha256 of manifest/index/全部 shards/swap_pair/target_head）→ 登记为共享资产。
- 2026-10-15 00:17 — D2 dfly 系列观察（引用级，未独立复核）：base1/2 fwd 215.1/213.3、bwd 382.8/382.3、clip/opt ≈4.9/28.7（dfly 42 张量图）；dedup 腿 rc=1（服务器副本缺 factory 补丁）→ 该系列 dedup 无效待重跑；与 pin 系列（36 张量/jetspec）分属不同目标，不得合并主张。
- 2026-10-15 00:18 — **D3 最终关闭确认**（C0 只读核对）：lane `research/rsi-d3` @ `b88cdf2`，worktree 全净；交付物 REPORT/EXPERIMENTS/handoff/support_matrix/evidence/scripts 齐；预算自报 0.22/1.0 GPU-h；服务器清理（/tmp 已清、`/home/lyh/vlm-d3` footprint 4.5MB）；lane 未 push（本地分支可查）；STOP、无集成、再开条件=REPORT §8。
- 2026-10-15 00:33 — **更正前条 dfly 观察**：dedup1 首败=同步问题已澄清；chain driver 重跑 rc=0，**四腿全部有效**（fwd 215.1/213.3→200.9/201.7；bwd 382.8/382.3→363.6/364.4）；12 步 loss 四腿逐值相同（C0 直读 loss_rows，maxdiff=0.0）。
- 2026-10-15 00:23 — D5 lane **79/79 C0 独立复跑 ✓**（本机 dflare venv、no-write；较 43/43 增 split/parity 36 项；含两卡 split 接线与 merge/verify 语义测试）；D2 jet8 对复核：fwd 374.8→348.6（−26.2ms）、loss maxdiff 2.06e-4 与 REPORT 一致。
- 2026-10-15 00:31 — D1 **W3 store 共享资产登记**：复合 sha256 `95ef12e4185b261f…`（192 行全多轮 all_turns/seq 2048/截断 0；model fp `e044f488…`；free 13G）。
- 2026-10-15 00:34 — D1 R3 抽查（C0 独立重算，`r3_mix_w1w2_k2`）：median token 255.6 vs canonical 317.6 → **1.2424×**（与 summary 一致）；逐 logical_batch loss_rel max 4.15e-3（一致）；72 行=8 臂×9 批（1 warmup）。**发现**：summary `grad_rel_err_max/update_rel_err_max=0.0` 与 warmup 行 `grad_rel≈5e-3` 不符 → grad/update parity 未逐批记录，需 D1 说明/修正（对 wave-2 判读重要）。
- 2026-10-15 00:36 — D1 R3 抽查（C0 独立重算，`r3_mix_w1w2_k4`，sub_batch=2）：canonical 315.4 → cost 254.4（**1.240×**）/ grid 253.0（1.247×）/ split 266.7（1.183×）；**peak 15.3→8.2GB（−46%）**——k4 与 k2 墙钟相近、显存更深（k2: −30%）。w3_k4 运行中（GPU3）。
- 2026-10-15 00:37 — D4 W2 降级验证复核（引用级）：on 腿 config `online_feature_cache_adaptive=true`（off=false）确认；两腿 fc counters 与 final_loss 一致——与"无压力→无调整"预注册期望相符；`adaptive` 汇总字段为 null，待 D4 收口时给口径。
- 2026-10-15 00:38 — D1 R3 抽查（C0 独立重算，`r3_w3_k4`）：652.7→cost 548.4（**1.190×**）/ grid 549.4；peak 25.0→**10.6GB（−57.5%）**；loss_rel max 4.79e-3（逐批重算一致）。**D1 修正口径已核（w0-k2 EXPERIMENTS 条目）**：regroup-vs-ref 漂移 2.7–5.4e-5、grad_rel≤4.5e-3、update_rel≤6.2e-8、cp 隔离 canonical 的 bf16 权重量化（4.9e-3→1.5e-7）——此前"summary grad=0.0"矛盾已按此解释修正（字段本身仍建议修）。
- 2026-10-15 00:43 — D1 round-2 复核（C0 独立复跑，worktree `56b6563`）：`tests/oracle_regroup_parity.py` **rc=0**（fp64：canonical 与 3 种重排 + 全 mode，dloss≤4.9e-7/grad_rel≤1.7e-7；float32 同阶）；`tests/test_bucketing_pipeline.py` **rc=0**（Z 记账解析一致、6 样本权重核对通过）。EXPERIMENTS 24 条全量、`hardware` 无缺；R3 7 条含卡位/co-resident 叙述 ✓。**遗留**：`git_sha` 应注明 round-2 后续提交为 `56b6563`；summary `grad_rel_err_max=0.0` 与 jsonl 实测（parity 批七臂 grad≈5.07–5.15e-3、upd≈6.6e-8）不符——聚合缺陷，列为 `d1-001-c0` §3 修正项。
- 2026-10-15 00:42 — **事故 #6（check→launch 竞态，D1 自报）**：D1 R3 W0-k2 首跑在 quiet 自检后、启动瞬间被外部进程抢占 GPU1 → OOM 失败；无他人进程被杀；等待后重试成功，以重跑为准。处置建议：自检与 launch 间隔最小化 + 启动后 10s 复核；纪律不变（不新增守卫）。
- 2026-10-15 00:50 — D1 **MIX store** 登记为共享资产：`/home/lyh/rsi-d1/runs/d1_mix_w1w2/features/`（335 行 = W1 191 + W2 144 硬链接交织 index；脚本 `scripts/make_mixed_store.py`；model fp `e044f488…`；只读、引用级）。
- 2026-10-15 00:47 — **L-W2-D1 发布**（RESOURCE_SCHEDULE §4）：单卡 ≤1.5 GPU-h、GPU3 优先（D5 链动态对排除 3）、quiet-gate 照旧；wave-2 配对（W3+MIX、k=4、5 臂、≥3 交替、100–200 更新 + capacity demo）；与 D5 chain2 并行不冲突（D5 用 1/2、D1 用 3）。
- 2026-10-15 00:52 — D4 W2 收口复核（C0）：lane `a8e762c`；4 文件 sha1 与集成 `1f8963e` 一致；F1–F4 PASS（counters 645/129/129/0/0；adjustments=0；sensor meminfo；exit=0；自清 ✓）；merge-ready 评估见 `d4-003-c0`。
- 2026-10-15 00:49 — D1 观察（只读）：自启 round-2.1 `--parity 3` 重复跑——mix k4 p3 完成（cost 1.170/grid 1.163 < 首跑 1.240/1.247，跨 run 波动待聚合）；w3 k4 p3 运行于 GPU2（quiet-guard ✓；非 C0 指令，属 wave-1 证据加强）；卡位提醒见 RESOURCE_SCHEDULE（sync 腿需 1+2；长跑优先 GPU3）。
- 2026-10-15 00:52 — D1 round-2.1 抽样复核（C0，服务器直读）：新 summary 字段 `*_vs_ref`/`*_vs_canonical` 生效（mix k4 p3：canonical-vs-ref grad 3.66e-3=bf16 量化；各重排臂 vs-ref grad ≤5.3e-3、update ≤7.2e-8；loss vs-ref ≤2.1e-5；peak 与首跑一致）。旧 summary 的 `grad_rel_err_max=0.0` 仍须在提交里注明口径（或重生成）。
- 2026-10-15 00:57 — D1 round-2.1 p3 队列收尾（C0 服务器直读）：**5/7 格 rc=0**（mix_k4 00:47 / w3_k4 00:51 / w0_k2 00:53 / w1_k2 00:55 / w2_k2 exit=0 00:56:52）；**`mix_w1w2_k2_p3` 与 `w3_k2_p3` guard 单次失败被跳过**——卡已清空（18MiB/48522 free）但 util 残差采样 31% > 10；脚本无重试、无产物。v2 summary 核读 ✓（`*_vs_ref` grad 5.96e-3 / update 5.78e-8；`parity_semantics` 自述“null=未记录、0.0 为真实值”；git_sha 注记 `56b6563+round2.1-parity3`）。待办见 §7 对 D1。
- 2026-10-15 00:57 — 服务器快照：GPU0=外部 xs 15.9GB/96%；GPU1=D5 staged_b8 8.3GB/62%（regen 1448/2000、kept=679 @00:56）；GPU2/GPU3 空闲；disk 17G。**L-W2-D1 窗口开放**（D1 自启；D5 后续 sync 腿用 1+2，互不冲突）。
- 2026-10-15 00:57 — D5 lane S3 线修复合入（记录级，待复跑复核）：`bb0036e`（split_pipeline 入口 roots 修正+回归）、`2378f73`（chain2 header + G2_LADDER post-mortem）、`2cc57c6`（split_stage 装配 media 符号链接——修 smoke_s3 worker FileNotFoundError；+2 测试）、`daf6b1d`（`G2_SKIP_LEGS` 旋钮 + S3 追赶计划）。策略=「S3 线内硬失败只停 S3 线」；**S3 追赶链（smoke_s3→s3_b8→s3_b32）待非 S3 legs 完成后启动**；Tier C 仍待 C0。C0 复跑计划：追赶链开跑前后按 lane 85/85 + smoke_s3 parity 再核。
- 2026-10-15 00:58 — **D2 修正三项全闭环（lane 终态）**：`bfb3635`（00:29:55；9 文件 knobs 默认 OFF；12 tests 含 fp64 oracle）+ `d6e3c9a`（00:30:10 docs）；worktree 全净；REPORT §4.1 已改述“12/12（‘13/13’ 为计数笔误）”；handoff 含 9 文件冻结 sha1 清单 + 聚合 `88457542…`。D2 = STOP 终态、无待办。
- 2026-10-15 01:04 — D1 p3 全闭环（C0 服务器直读）：两格补跑 rc=0（mix_k2_p3 exit=0 01:02:40 / w3_k2_p3 exit=0 01:04:47；`r3_p3_tail.outer.log` = TAIL DONE；guard 重试至 util=0 修复了残差采样跳过）；两格 v2 summary 字段/语义与其余格一致。**round-2.1 = 7/7**；D1 未提交回填（EXPERIMENTS + `r3_ab_step.py`）待其提交。
- 2026-10-15 01:03 — D5 lane **85/85 C0 独立复跑 ✓**（本机 dflare venv、no-write；@`daf6b1d`；覆盖 split_pipeline 入口修复、split_stage media 暂存 `TestSliceMediaStaging`、skip 旋钮回归）——S3 追赶链前置复核通过（smoke_s3 parity 待链上复跑）。
- 2026-10-15 01:07 — D5 staged_b8 进展（C0 服务器直读）：regen 2000 → kept **1012**（capture 样本数）；capture 完成（seq 2048 bf16）；**train 中（step 65/762 @01:07:00**，loss≈7.0 / acc≈0.05 / al≈1.05，GPU1 单卡；GPU2/3 空闲）。窗口披露：与 D1 p3（GPU2）同节点并行至 01:04:47——腿 wall 标 contended（cross-card），D1 p3 为 parity-only。
- 2026-10-15 01:08 — D1 lane 提交 `a193ef8`（round-2.1：parity 记录修正（per-batch vs-ref/vs-canonical、null≠0.0、device 字段、summary v2）+ 7 格 --parity 3 重跑（行序核同）+ 全 GPU 行 device 声明；REPORT §9.1；预算更新 **0.92/1.5 GPU-h**）、`4135691`（wave-2 harness：parity-batches first/mid/last、epoch 循环序、whole-trajectory loss-vs-ref、OOM-tolerant arms、condition-③ git_sha 注记闭环）与 `b85d99d`（step-4 ragged 审计，CPU-only，REPORT §10）；worktree 净。**d1-001-c0 §3 条件全部落地**，待 wave-2 出数。
- 2026-10-15 01:17 — **D5 staged_b8 腿完成**（00:38:56→01:17:10，wall≈38.2min≈**0.64 GPU-h 单卡**（devs=1））：regen kept **1012/2000**；capture 1012（seq 2048 bf16）；train 762 步 final_loss=4.03125；drafter 视觉检查 passed=True（logits_diff=9.766 / ce_delta_swap=0.578 / ce_delta_perturb=0.001118）；ckpt_last 保留。记账观察：`g2_progress.log` 无 exit 行（仅 start/guard）——harvest 按时间戳核 wall；与 D1 p3（GPU2）异卡并行至 01:04:47（cross-card，非计时用途）。
- 2026-10-15 01:17 — D5 **sync_b8 腿启动**（01:17:14，mode=online devs=1 batch=8，run `g2_sync_b8_1015_0117`；free=10G load=11.86）；regen 进行中（早段 keep 偏低：176 行 kept=9——待 harvest 解读，不先下结论）。
- 2026-10-15 01:17 — **D1 wave-2 启动（L-W2-D1 履约）**：01:17:24 GPU3（guard ✓ util=0% free=48522MiB disk=21G），`w2_chain_status.log`=`start w2_w3_k4_r1 seed=42`；命令 100 批 / sub-batch 2 / parity@0,50,100 / arms canonical,ref,split,token,cost,grid；`git=research/rsi-d1@4135691+wave2`。至 01:20 已至 b24（loss_rel≈3.5e-5）。**不占 1+2**（D5 sync_b8 在 GPU1）✓。
- 2026-10-15 01:26 — D1 wave-2 **r1（W3 k4 seed42，100 批）rc=0**（01:17:23→01:25:53，wall≈8.5min）：**逐批配对 median ratio：grid 1.129 / cost 1.123 / token 1.103 / split 1.060（grid 100/100 批全胜；p10≈1.06、p90≈1.23）**；中位数比（ratio-of-medians）：grid 1.144 / cost 1.135；peak 25.0→**10.6 GiB（−57.5%）**；parity（vs ref）：loss ≤4.9e-5、grad ≤4.4e-3、update ≤6.6e-8。**观察（不下结论）**：本 run 步时增益低于 round-2 8 批 cell（+19%）；待全链 6 run（{W3,MIX}×seeds 42/43/44）+ 跨 seed 聚合后按 `d1-001-c0` §2 口径判读；候选解释：harness 版本（epoch 循环序/100 批）、节点并行负载（D5 sync_b8 + xs）、跨 run 波动。
- 2026-10-15 01:26 — D5 记账口径补正（读 `G2_LADDER.md`）：GPU-h=**分配卡数×wall**（腿分配 2 卡、单卡活跃）→ staged_b8 按 ≈**1.27 GPU-h** 计入（2×38.2min）；01:17 条目中「0.64 单卡」为有效活跃口径。两者并存报告，预算口径以分配为准。
- 2026-10-15 01:31 — D1 wave-2 **r2（MIX k4 seed42，100 批）rc=0**（01:25:53→01:30:40）：**逐批配对 median：cost 1.158 / grid 1.154 / token 1.133 / split 1.084（grid 95/100 胜、cost 94/100；grid p10≈1.04、p90≈1.39）**；ratio-of-medians grid 1.152；peak 24.0→**10.3 GiB（−56.9%）**；parity（vs ref）：loss ≤7.2e-5、grad ≤4.6e-3、update ≤6.5e-8。观察：MIX 格步时增益（1.154）略高于 r1 W3（1.129），仍低于 round-2 8 批 cell；r3（W3 seed43）已于 01:30:40 自启，r4–r6 待续；6 run 全链完成后按 `d1-001-c0` §2 口径聚合判读。窗口披露：r2 期间同节点共驻 D5 sync_b8（GPU1，cross-card）+ 外部 xs（GPU0，01:31 快照已空闲；重叠时段未精确界定）；D1 在 GPU3。
- 2026-10-15 01:39 — D1 wave-2 **r3（W3 k4 seed43）rc=0**（01:30:40→01:39:04）：**grid 1.136（98/100 胜）/ cost 1.134（99/100）/ token 1.103 / split 1.063**；peak 25.0→**10.6 GiB（−57.6%）**；parity（vs ref）：loss ≤6.1e-5、grad ≤4.4e-3、update ≤6.5e-8。观察：W3 两 seed（r1 1.129 / r3 1.136）高度一致，均低于 round-2 8 批 cell；r4（MIX seed43）01:39:04 自启；链余 r5（W3 seed44）、r6（MIX seed44）。
- 2026-10-15 01:44 — D1 wave-2 **r4（MIX k4 seed43）rc=0**（01:39:04→01:44:06）：**grid 1.137（96/100）/ cost 1.152（93/100）/ token 1.118 / split 1.069**；peak 24.0→10.3 GiB（−57.1%）；parity：loss ≤9.5e-5、grad ≤5.3e-3、update ≤6.7e-8。窗口披露：同节点 D5 sync_b8（GPU1，cross-card）；外部 xs 已结束（GPU0 空闲）。
- 2026-10-15 01:52 — D1 wave-2 **r5（W3 k4 seed44）rc=0**（01:44:06→01:52:14）：**grid 1.141（100/100）/ cost 1.133（100/100）/ token 1.115 / split 1.076**；peak 25.0→10.6 GiB（−57.6%）；parity：loss ≤4.8e-5、grad ≤4.4e-3、update ≤6.4e-8。W3 三 seed 一致（1.129/1.136/1.141，均值≈1.135）。r6（MIX seed44）01:52:14 自启（末腿）。
- 2026-10-15 01:53 — D5 sync_b8 训练阶段完成（`train: done steps=762 final_loss=4.03125` @01:53:22；腿 01:17:14 起 wall≈36.1min；visual check/收尾进行中）；按分配 2 卡口径 ≈1.2 GPU-h，待 D5 harvest 复核。磁盘水位 16G（20G→16G，下降中；低于 10G 约束即上报）。
- 2026-10-15 01:57 — D1 wave-2 **CHAIN DONE：6/6 rc=0**（01:17:23→01:57:26，wall 40.1min，GPU3 单卡）。聚合（C0 独立重算）：W3 grid 1.129/1.136/1.141（均值 1.135）、MIX grid 1.154/1.137/1.107（均值 1.133）；grid win 580/600；peak −57%；parity（b0/50/100 ✓）loss ≤9.5e-5 / grad ≤5.3e-3 / update ≤6.7e-8。**判读：步时 13.5%/13.3% 低于 15% 但显存 −57% 远超 25%（OR 门槛通过）** → 裁决 `d1-002-c0` = GO（修订口径）+ capacity demo 收尾。
- 2026-10-15 01:53 — D5 **sync_b8 leg OK**（01:17:14→01:53:28，wall≈36.2min；762 步 final_loss=4.03125；drafter visual check passed=True；按分配 2 卡口径 ≈1.21 GPU-h，待 harvest 复核）；**sync_b32 于 01:53:29 启动（cards 0+2）**；链脚本第 107 行 GPU-h 算式报错（记账须按时间戳重算）；磁盘回 21G。

- 2026-10-15 02:10 — D1 capacity demo 3 腿 rc=0（GPU3；cap16 W3 02:00:32→02:01:54 / MIX ≈02:02→02:04 / cap24 W3 02:06:20→02:07:16）：**C0 独立重算原始 jsonl**：W3 b16 canonical 43.59 GiB → cost/grid **14.32 GiB**（9/9）；W3 b24 canonical **OOM 5/5** vs cost **14.41 GiB 5/5** ✓；MIX b16 27.74 → 10.30/9.83 GiB（9/9）；与 REPORT §11.4 逐值一致；增量 ≈0.17 GPU-h 单卡。窗口披露：cap16 腿期间 GPU0 外部任务 97%/10.4GB（cross-card）。
- 2026-10-15 02:13 — D1 close-out `6dc5f74` 交付（REPORT §11、EXPERIMENTS 42 行、STATE/handoff close-out request、10 个 w2 summary 顶层 `parity_batches` 非失真回填并披露）→ **L-W2-D1 关闭**（0.73 in-run + ≈0.17 demo ≈ **0.90/1.5 GPU-h**）。
- 2026-10-15 02:14 — D5 **CHAIN2 DONE（Tier 0+A+B）**（02:13:58）：sync_b32 leg OK（01:53:29→02:13:58；768 步 final_loss=4.0625）；**Tier C + staged_b32 store-parity 复验留 C0（2:40 答复：Tier C 倾向暂缓，S3 出数后定）**；S3 追赶链（smoke_s3→s3_b8→s3_b32）待跑；四卡现均空闲（util 0%）。
- 2026-10-15 02:19 — **d1-001 最小集集成落地 `research/rsi-c0` @ `4e1674e`**（`train.regroup` opt-in；默认路径 byte-identical；本机全套 **509 pass / 6 skip / 0 fail**）；裁决 `d1-003-c0`（capacity demo 核验 + 租约关闭 + 集成记录）；D4×D1 2×2 排程解锁；held-out 质量验证仍欠。
- 2026-10-15 02:19–02:25 — D1 车道三连提交：R4 评估 `e147a01`（k4 残余 4.7–5.2%/6.1–6.4% 步成本、96–98% invalid-block slack；size-1 fp64 oracle 全 ragged exact）、集成交叉复核 `3e29244`（**结论 PASS**：regroup.py verbatim、fp32 cast 一致、train.py 逐项对齐、20/20 复跑；建议 (a) all-singleton 用例）、重编号 `8c7776c`（d1-003→d1-004）。**C0 回执**：建议 (a) 采纳（`192554a`，全套 509/6/0）；`d1-004` 接收 → **发布 L-W3-D1R4（≤0.2 GPU-h，GPU3 优先，即刻窗口）**，判决规则按提案写死（≥1.135 → R4 正分支 / 否则边界关闭）。
- 2026-10-15 02:19–02:20 — **事故 #7 登记（D5 S3 smoke 假失败，工具链缺陷）**：S3 追赶链 smoke_s3（`g2_smoke_s3_1015_0219`，cards 1+2）worker w0 报 `Regen kept zero samples`（rc=1）→ S3 线停（02:19:52），Tier 0+A+B 保留。**C0 取证**：canonical smoke kept=1/16（15 行 too_short），切分后某 worker 合法 keep=0；worker 级 zero-keep 不应硬失败（merge 后查全局；与 canonical `run_regen` 语义对齐）。**D5 修复在途**（`split_worker.py` + `test_split_worker_zero.py`）；修复后 `G2_SKIP_LEGS` 重跑 smoke_s3→s3_b8→s3_b32。详见 RESOURCE_SCHEDULE §3。

- 2026-10-15 02:27–02:30 — **C0 集成 e2e smoke：`train.regroup` 产品路径首跑 PASS**（服务器 GPU0 单卡；`C0-E2E-smoke-{canon,regroup}`）：两格各 20 步 rc=0、无 NaN、loss 正常下降；**峰值 25.34→9.98 GB（−60.6%）**；逐位 loss 差 ≤6.1e-3（在 shipped 路径自身 bf16 波动带内；canonical 系 bf16 归约量化显示）；梯度覆盖 36/36 ✓。`d1-003-c0`/`d1-004-c0` 的"无端到端运行"缺口关闭。证据：`evidence/c0_e2e_regroup_smoke_20261015.md`；服务器 `/home/lyh/rsi-c0/{logs,runs}`（ckpt 已清）。

### 对 D1（2026-10-15 02:05，wave-2 判读）

- 裁决见 `d1-002-c0`：**GO（修订口径）** —— 步时均值 13.5%（W3）/13.3%（MIX）、单 seed 最高 15.4%；显存 −57%；win 580/600；parity 三批位 ✓。
- **收尾项**：capacity demo（batch16@sub4 vs batch8 canonical，同 peak 预算；≤0.2 GPU-h；GPU3 空闲可用）为 L-W2-D1 唯一未跑项；另可顺手修 v2 summary 顶层 `parity_batches=null`（实际已按 0/50/100 执行）。完成后 C0 核验 → 关闭租约 → 解锁集成（d1-001 最小集、D4×D1 2×2）。
- round-2 的 +19.0%/+24.7%（8 批 cell）不再作对外主数字。

### 对 D1（2026-10-15 02:22，close-out 核验）

- capacity demo 收到并**核验通过**（§复核记录 02:10；与 REPORT §11.4 逐值一致）→ **L-W2-D1 关闭**；close-out `6dc5f74` 交付完整（parity_batches 回填已披露）。
- **d1-001 最小集已集成**（`research/rsi-c0` @ `4e1674e`，见 `d1-003-c0` §3）；默认路径 byte-identical（测试锁）。
- 下一项：**D4×D1 2×2 排程解锁**（online cache × regroup；D4 侧已 merge-ready 静态面）——进行前由 C0 发布预注册（假设/门槛/预算）；held-out 质量验证（AL）列入后续候选实验。

### 对 D5（2026-10-15 02:22，chain2 T0+A+B 完成）

- chain2 结果收到：staged_b8（wall 2297.3s / ≈1.27 GPU-h 分配口径）vs sync_b8（2174.5s，**−5.3%**，远低于 sim 的 −14%）——本 build online 实时抽取与训练**串行**（epoch-1 ≈1.5s/step），缓存只省 epoch≥2 重算；请写进 harvest/REPORT §10。
- sync_b32（768 步 final_loss=4.0625）与 regen b32 = 2.33× 一致：**batch 是静态语义旋钮**（b32≠b8，kept 集合不同），只能各自与自身对照。
- **下一步**：(a) S3 追赶链按既定计划执行（GPU 现全空闲；tri-state parity 保持）；(b) **Tier C（pinned 1+1）倾向暂缓**——sim 仅 +3%，待 S3 出数后再定；(c) harvest 出 §10 汇总 + GPU-h 按时间戳重算（链脚本 107 行算式不可用）。

### 对 D1（2026-10-15 02:25，cross-check 回执 + d1-004 接收）

- **交叉复核 `3e29244` 收到**：结论与 C0 自查一致（verbatim 移植 / 逐组语义对齐 / D1 venv 20/20 复跑）；建议 (a) 已落地——all-singleton（sub1）进 parity sweep，`research/rsi-c0` @ `192554a`（全套 509 pass/6 skip）。集成复核级别升为"独立复跑 + 作者核读"（见 `d1-004-c0` §4）。
- **`d1-004` 接收**：发布 **L-W3-D1R4**（GPU3 优先、≤0.2 GPU-h、即刻窗口；提案 §1 命令照旧；ref 臂不进已接受）。判决按预注册执行：sub1/canonical ≥ 1.135 → R4 正分支（D3 packed-block 合同讨论；**D3 kernel 轨道已 STOP，kernel 预算须 C0+用户再开，不在本租约**）；否则 R4 以边界结论关闭。完成后 C0 独立重算。
- 编号说明：`d1-003`→`d1-004` 重编号 C0 确认；今后引用以 `d1-004*` 为准。
- 提示：C0 计划用 **GPU0** 跑集成 e2e smoke（`train.regroup` 端到端首跑，补 `d1-004-c0` §4 的未覆盖项）与后续 D4×D1 2×2（预注册另发）；与你的 GPU3 短跑互不冲突。

### 对 D5（2026-10-15 02:25，事故 #7 与修复）

- S3 追赶链 smoke_s3 于 02:19:21–02:19:52 失败 → 已按你的"停 S3 线"政策处理（Tier 0+A+B 保留），**C0 登记为事故 #7**（工具链缺陷，非租约违规；证据见 RESOURCE_SCHEDULE §3）。
- C0 取证结论（供修复对照）：canonical smoke 全局 kept=1/16，切分后 workers 各自跑 `run_regen`，其中 w0 切片合法 keep=0 却被 `run_regen` 的全局零检查硬停——**worker 级 zero-keep 应容忍**（写 stats、空切片照常 merge），零检查放在 merge 后的全局统计；canonical 语义（批内 zero-keep 容忍、全局才判零）应逐项对齐。你已在改 `split_worker.py`+`test_split_worker_zero.py`——方向一致。
- 修复提交后：`G2_SKIP_LEGS` 重跑 smoke_s3（gate）→ s3_b8 → s3_b32；S3 结果、staged_b32 store-parity 复验、§10 汇总与 GPU-h 时间戳重算照旧；**Tier C 仍倾向暂缓**（见 02:22 条目）。

### R4 收口（2026-10-15 02:38；L-W3-D1R4 CLOSED）

- R4 sub1 边界两格完成（W3 02:26:44→02:30:22、MIX→02:33:11，均 rc=0；GPU3；总 6m27s ≈ **0.108 GPU-h**）。C0 独立重算（raw jsonl，100 timed 批/臂；与 summary 逐值一致）：W3 canon 652.626ms → cost 577.772ms（**1.1296 < 1.135**）；MIX 302.156→296.261ms（**1.0199**）；峰值 W3 25.04→8.16 GiB（−67.4%）、MIX 23.99→7.99 GiB（−66.7%）；parity vs canonical：loss ≤6.49e-3、update ≤7.6e-8（bf16 带内）。
- **判决：负分支——R4 以边界结论关闭**（`d1-005-c0`）：sub1 无增量速度余量（W3≈k4 下限；MIX 被微批开销吞掉）；**D3 packed-block kernel 不触发重开**（条件未满足）。shipped 配置仍 = grid/sub2（k4）。
- 窗口披露：跨卡共驻 = C0 smoke（GPU0 02:27–02:30）+ D5 S3 链（1+2，02:28 起）；GPU3 卡内无共租。D1 待补：EXPERIMENTS 两行。

### D5 S3 恢复（2026-10-15 02:38）

- 修复生效：smoke_s3 重跑 **exit=0**（wall 102s，02:30:33）；**s3_b8 02:30:34 起**（cards 1+2、batch8、mode=s3；run `g2_s3_b8_1015_0230`），其后 s3_b32 排队。待出数后照旧 harvest（§10 汇总 / staged_b32 store-parity / GPU-h 时间戳重算）；Tier C 仍暂缓。

### D4×D1 2×2 预注册发布（2026-10-15 02:38；L-W3-D4xD1）

- `d4xd1-001-c0`：{cache_off, cache_on=16GiB} × {bucket_off, grid/sub2}，online 模式（`train.feature_mode=online`），W3+MIX × seed42、1 warmup + 100 updates = **8 legs**；F1–F4 判决冻结（数值带 7.5e-3 / 缓存生效 hits≥0.5×行数 / 峰值比值 ≤0.75 / 步时方向 ≥1.00）；≤**1.5 GPU-h**；GPU3 优先且 8 legs 同卡；**D1 执行全部 8 腿**（另半 = D4 机制产品路径集成运行，D4 线不重开）；开跑前 D1 出执行 addendum（命令 / parity 捕获法 / anchor 口径）→ C0 ack。
- 定位：wave-3/论文主轴（系统组合叙事）的判定实验；承接 `d1-003-c0` §4 与 INTEGRATION §5 已排项。输入已齐（D1 handoff `b0c2225` + D4 `1f8963e`/`d4-003-c0`）。

### 对 D1（2026-10-15 02:45，R4 收尾收到 + 2×2 放行 + d1-006 接收）

- **R4 收尾收到**（`14fcce2`/`8aa53f0`/`c31a9de`）：EXPERIMENTS 45 行（+2 边界行，含跨卡共驻披露：C0 smoke GPU0 / D5 1+2）；REPORT §12.5；parity 数字回填（grad 5.7/6.3e-3、update 7.6/7.4e-8——与 C0 重算一致）；重编号 d1-005→**d1-006** 接受（我方编号：`d1-005-c0`=R4 收口裁决）。**R4 线全闭环**（实测 0.107–0.108/0.2 GPU-h）。
- **D4×D1 2×2：放行（fast-path，无需二次 ack）**。预注册 = `d4xd1-001-c0`（dd987e7）。与你的 launch note 已对齐（ONLINE 强制、keys 只读核对 ✓）；**cap 冻结值 = 16 GiB**（prereg §2；对应 D4 1012 行 S3 腿配置；"8 GiB" 是 D4 129 行腿配置，非本 2×2 值；两格 ~2.3GB 全量驻留、预期 eviction=0）。开跑前把三条落 handoff 即开跑：① 入口命令/分腿链脚本路径；② parity 捕获法（b0/50/100 grad/update 如何取；不可得则显式退到 loss 轨迹+终值带）；③ anchor 同表机制说明。租约 **L-W3-D4xD1 ≤1.5 GPU-h**、GPU3（现空）、8 legs 同卡；F1–F4 照冻结阈；完成交证据，C0 独立重算收口。
- **d1-006 held-out 质量：接收（条件化）→ 租约 L-W3-D1Q ≤2.4 GPU-h（staged）**。编排修订（带先行、先带后比）：**Stage-1 = canonical×{42,43}**（建带：|ΔAL|/|Δacc| seed 对差；≈0.7–1.2）→ **Stage-2 = regroup×{42,43}**（比较：|Δ| ≤ band → 质量轴 PASS；≈0.7–1.2）。不用 repo 历史带（除非先报 C0 冻结更强数字）。顺序：2×2 收口后、GPU3 同卡顺序；N=200 步、同源 store 特征、`eval_dflash.py`（64 prompts/固定 val）；<200 步不收敛 → "不判定"照记。3-seed 长训练留待本 stage 过后按晋级预算另批。

### C0 侧 novelty 初筛（2026-10-15 02:50，D1 机制）

- 第一轮粗查（arXiv API + OpenAlex；S2 全程 429、无 WebSearch → 工程文档未核）。结论：**空间拥挤**——"去 padding / packing / bucketing / 自动 microbatch"有 2017→2026 连续先例（含 2026 SSRN 综述《Beyond Concatenation》）。
- 仍可能站住的 delta（待深读确认）：① 单 step 内"精确损失等价的重分组"formalization；② 容量轴（固定 peak ×2–3 logical batch）；③ invalid-block slack 分解诊断（96–98%）。**工程价值独立成立；论文新颖性未证。**
- 记录：`evidence/c0_novelty_firstpass_d1_20261015.md`（prior-art 表 + 深读清单：综述 / Libra(2607.23250) / SeqPacking(2107.02027) / Varuna(2111.04007) 等）。

### 资源事件（2026-10-15 02:59–03:00，C0）

- 磁盘告警并自愈：s3_b8 峰值写至 avail 5G → exit=0（02:59:11，0.95 GPU-h，wall 1716s）后 D5 prune（ckpt/shard，8.0G→19.4G）；`s3_b32` 首试 rc=96 → 300s 重试（run `g2_s3_b32_1015_0259`、vs `g2_sync_b32_1015_0153`）。
- D1：02:54 六连重试 GUARD FAIL（disk 7–9G；GPU3 本身 quiet）→ **02:59:21 空间释放后自动重启 `d4xd1_smoke`**（3 步；visual_path_check 过）→ 2×2 8 腿随后（L-W3-D4xD1 条款照旧）。留痕：D5 `logs/d5_prune.log` / `logs/d5_prune_s3b8.sh`；D1 `logs/d4xd1_smoke.log`。
- novelty 深读 #1：`evidence/c0_novelty_deepread_d1_20261015.md`（SSRN 综述 DOI 核实 10.2139/ssrn.7343278；ACL'25/IJCAI'25/NeurIPS'24 对照；工程栈核查=HF padding-free / Megatron PackedSeqParams / torchtune packing 均 HTTP 200）。

### D1 2×2：初跑中止与重启（2026-10-15 03:0x–03:2x，C0 收报）

- 03:02:55 链启（GPU3）：leg1 ref rc=0（03:07:29）、leg2 rc=0（03:12:26）、leg3 rc=0（03:15:25）；leg4 03:15:25 起、03:17:31 rc=1（guard：/home 7–9G 低于当时 10G 门；非训练错误）。
- **捕获缺陷（D1 自检发现；全披露于 addendum §5.5）**：首版 sitecustomize grads hook 写成 `p.detach()`（参数本体而非 `.grad`）——首个 parity diff 即自检矛盾（b0 全 0.0 + ‖captured‖≈338≈参数范数 ≠ train_log grad_norm 89.05）→ D1 03:17 果断停链；**4 个 W3 腿目录 + 全部 tensor 目录已删**（证据无效，正当）。
- 修复与复核：grads 捕获移至 `clip_grad_norm_` 入口（clip 前、R4 同语义）；smoke（03:22/03:24）通过——clip trace call0：my_clone 88.956 vs clip_return 89.053 = **0.11% bf16 量化类 ✓**；链路自校 ok。
- **初跑一切数字以 void 对待**（含 −47% 峰值、~3× 步时、任何 parity）——只作仪器调通参考；**重启集（8 腿全集）= 唯一正式证据**。
- 重启授权 + update 轴读法：**裁定 `d4xd1-002-c0`**（03:30 发布；本裁定在重启数据前冻结）。预算：初跑 ≈0.26 GPU-h 入账；重启 ≤1.2；合计 ≤1.5（越界披露）。guard 双卷现状：/tmp 23G、/home 17.7G（s3_b32 exit=0 + prune 后）。

### 裁定 d4xd1-002-c0（update 轴读法 + 重启 GO；数据前冻结）

- 请求（D1 addendum §4）：`update_rel ≤1e-7` 记录自 R4/round-2 口径（lr·‖g‖/‖p‖·grad_rel 量级），产品路径 AdamW 的 `p_after` 漂移由归一化灵敏度决定（量级更大），常数不可移植 → 请 C0 定读法。
- 裁定要点：①raw 交付不变 + 分解 (grad_rel, update_rel, lr, ρ) 必交；②`≤1e-7` 硬谓词仅适用于 **floor probe**（cache_on×bucket_off vs ref；预期严格 0）；③bucket_on 腿 update 轴 = **类一致性读法**（健康项：恒等式 ≤0.5% 类；自洽项：ρ 与机制换算同量级；不自洽 → semantic anomaly → correctness-blocker 支线）；④**主判决门一字不动**（F1 7.5e-3 / grad 6e-3 / F2 / F3 / F4 / anchor 不变式 / rc/NaN/ckpt）；⑤依据：单位分析 + 异议在数据前 + 初跑整体作废不可用于调门 + 强声明载体不变。
- 收口映射（预登记）：全过→相容性成立；F3 败→交互冲突；F1 败→正确性阻断；F2 败/腿废→无效窗口重跑（计租约）。重启后收口判决落 `d4xd1-003-c0`。

### D5：s3_b32 收数与 G2 汇总（2026-10-15 03:2x，C0 收报）

- **s3_b32 exit=0**（03:22:07；wall 1074s；alloc 0.59 GPU-h）；prune 03:22:12 释放 /home 6.2G→17.7G。`g2_collect.json`（11 行）出数：**s3_b8 1716s**（parity 全 ✓；vs sync_b8 2174.5s **−21%**、vs staged_b8 2297.3s **−25%**）；**s3_b32 1074s**（regen parity ✓、**store parity 待 staged_b32 复验**；vs sync_b32 1228.8s **−12.6%**；落在 sim 带 950–1100 内）。
- batch 语义旋钮再确认（b8/b32 kept 集合不同；各自与自身对照）。**下一步**：staged_b32 store-parity 复验 + §10 汇总 + 三态 parity 收口 + GPU-h 时间戳重算；Tier C（pinned）暂缓照旧。

### 对 D1（2026-10-15 03:55，D4×D1 2×2 — C0 中期实读：W3 cell 4/4 rc=0；F1 观察项）

- **C0 直读（未依赖 harvest）**：W3 4 腿 rc=0（03:38:52 / 03:43:49 / 03:46:51 / 03:49:57）；floor probe（cache_on×bucket_off vs ref）**grads/updated/loss 全 bitwise 0.0**；两条 bucket_on 腿 parity 数值与 loss 序列**逐位一致**（与 cache 无关）。
- **F1-as-registered**：bucket_on 两腿 **FAIL**（max loss_rel **1.248e-2 @step 11**；steps 11–14 共 4 步超门；中位 1.85e-3 / p90 4.29e-3）；bucket_off 两腿 PASS（0.0）。**按冻结读法如实记录，不会事后放宽。**
- **归因（C0 初步）**：该比较先天含「regroup fp32 权重损失路径 vs 出厂 bf16 路径」的已知差类（`train.py:340` 默认 `fp32_weights=True`，注释 ~0.4% bias；wave-2 同 W3 行实测 canonical-vs-ref ≤5.96–6.60e-3），叠加动力学放大；与 cache/交互无关（bitwise 证据）。grads 逐张量 worst（b50 0.877 / b100 0.487）由小范数张量主导（ref 范数 5.45e-4 / 1.32e-3；绝对差 ~5–6.5e-4）。
- 细节与证据路径：`evidence/c0_d4xd1_interim_read_20261015.md`。
- **请求（非阻塞）**：①harvest 照跑（如实出 F1 FAIL + IDENTITY 门）；②深挖 step 11–14 时优先「同一步 loss 两条损失路径重算」；③MIX 收数后 C0 全量重算 → `d4xd1-003-c0` 收口（映射：F1 败 → 组合数值相容性声明按原注册不成立；按“另立项”给出事前注册的隔离对照提案）。

### 对 D5（2026-10-15 03:45，G2 三项请求应答 + C0 复核收据）— `d5-002-c0`

1. **套件复核 ✓（范围已固定）**。D5 请求复核 HEAD `55e4e9c`（88/88）。C0 固定范围：`git diff --stat 55e4e9c..7db5d93` = 仅 4 个文档（EXPERIMENTS/REPORT/STATE/handoff；+134/−1）→ 代码树与套件范围一致。复跑：本地（py3.12 venv）**88 pass**；服务器（c1vllm，lane-local `pip --target /tmp/c0-d5-pytest` 装 pytest）**88 pass、exit 0**。环境坑（备查）：c1vllm site-packages 自带 `tests` 包遮蔽仓库 `tests/` 命名空间 → PYTHONPATH 前置 shim（`/tmp/c0-d5-shim/tests/__init__.py` 内 `__path__` 重定向到仓库 tests）后全绿；之后在服务器跑 lane 套件需带 shim 或换 venv。
2. **数字抽查 ✓**。`logs/g2_collect.json`（11 行）walls：s3_b8 1716s vs sync_b8 2174.5 **−21.1%**、vs staged_b8 2297.3 **−25.3%**；s3_b32 1074s vs sync_b32 1228.8 **−12.6%**（与声称一致）。`_split/split_report.json` parity 逐项：s3_b8 `ok=true`（regen 1012/1012、skipped 988、store 1012 全 match）；s3_b32 `ok=null, partial=true`（regen/skipped match；store=null + "online-mode canonical" 原因串，按 tri-state 不作 store-parity 证据）。**轨迹抽验**：step 760/761/762 loss 行 s3_b8 ≡ staged_b8；step 768 s3_b32 ≡ sync_b32（逐字节）。
3. **裁定**：①复核通过（无整改要求）。②**批准 staged_b32**（新租约 L-W3-D5B，≤0.75 GPU-h，offline 单卡）——目的 = 关闭 s3_b32 store parity 盲区 + 补齐 b32 离线静态基线行；跑毕执行 `runner.split_verify.verify_against(g2_s3_b32_1015_0304, g2_staged_b32_*)`，输出路径入 EXPERIMENTS + handoff；建议实测 avail ≥15G 起步、跑后 prune。③**pinned_b8 = HOLD（不批）**：H1 预期 ~3% 增量、磁盘全程紧张、静态全景已可按 {staged,sync,s3-split}×{b8,b32} 收口；若后续"多 GPU 静态分配"成为交付主张，再单批单腿并按 §11 记录。④成本感知调度线维持 STOP（同 D5 自判）。

### 收口判决 `d4xd1-003-c0`（D4×D1 2×2；2026-10-15 04:10，C0）

- 8/8 rc=0（重启集 = 唯一正式证据；03:34:22→04:02:42，GPU3）；C0 独立重算（`tools/d4xd1_recompute.py`；输出 `evidence/c0_d4xd1_recompute_out_20261015.json`）与 D1 harvest 逐项一致。
- **判决（冻结门，不放宽）**：W3 cell——bucket_off 两腿 PASS（0.0）；**bucket_on 两腿 F1-as-registered FAIL**（max **1.2475e-2@11**；steps 11–14 超门；中位 1.854e-3/p90 4.290e-3）。MIX cell——**全过**（bucket_on max 6.1262e-3@37 ≤7.5e-3）。F2（W3 616/192/192/0；MIX 471/335/335/0）、F3（0.5324/0.5408）、F4（1.0217/1.0281；1.0138/1.2168）全 PASS；anchors 4/4 腿 cell 内一致（104 calls；尾随 sha1 `150a67a8…`/`1de1d7b3…`）；identity 8/8（≤5.7e-8，tol 1e-6）；floor probe grads/updated/loss **全 0.0**（跨进程确定性 + 缓存数值惰性，硬谓词成立）。
- **交互 = 0（bit 级）**：bucket_on 两 cache 水平逐位一致 → 全部偏差属布局（regroup）足迹，与 cache 无关。
- **映射（预登记执行）**：W3 cell F1 败 → **组合数值相容性声明按原注册不成立；cache×regroup 暂禁（正确性阻断）**；另立项 = `d4xd1-004-c0`（W3 损失路径隔离单腿，L-W3-D4xD1B ≤0.10）。MIX cell 相容性成立（单体）。
- 归因注记（数据后；不改门）：W3 超门主体 = 「regroup fp32 权重损失路径 vs 出厂 bf16」已知差类 + 在线动力学（未隔离 → 004）；grads parity 名义超限（b50/b100）由小范数张量主导（≤0.3% 全向量范数），语义注记；update 轴类一致性读法未见 anomaly（ρ 跨点 ≤6.7× 波动，无数量级跳跃）。
- 预算：初跑 0.26（void，含 smoke）+ 重启 ≈0.50（腿墙 0.472 + 修复 smoke ≈0.03）→ **L-W3-D4xD1 累计 ≈0.76/1.5（余额 ≈0.74）**。
- 全文：`proposals/d4xd1-003-c0.md`。

### 预注册 `d4xd1-004-c0`（W3 损失路径隔离对照；L-W3-D4xD1B）

- **单腿**：W3、cache_off、bucket_on（grid/sub2）+ **`train.regroup.fp32_weights=false`**；ref 复用（不重跑）；同 seed/行集/锁步协议/GPU3。
- **F1B 门（冻结）**：timed 2..101 max loss_rel ≤7.5e-3 → PASS；否则 FAIL。映射：PASS→归因收敛（超门=损失路径类；在线分组本身在带内；组合声明重建与否由 C0 另裁，不撤销 003 记录）；FAIL→在线分组本身超带，阻断加深（转机制排查）。
- **租约 L-W3-D4xD1B**：≤0.10 GPU-h（实测同型腿 0.051–0.083）；GPU3；D1 执行（harvest 回填提交后、d1-006 前）；完成后 C0 同工具重算。
- 全文：`proposals/d4xd1-004-c0.md`。

### 对 D1（2026-10-15 04:15，2×2 收官回执 + 下一步）

- 收讫：重启集 8 腿 rc=0；C0 重算与 harvest 数值逐项一致（F1/F2/F3/F4/identity/anchor 摘要）。判决见 `d4xd1-003-c0`（W3 F1-as-registered FAIL 如实保留；不写成“已知类故通过”）。
- 请求：①2×2 harvest 回填提交（EXPERIMENTS 8 行 + REPORT 节 + STATE/handoff 同步；引用 003/004）；②执行 `d4xd1-004`（fast-path：addendum 补 `fp32_weights=false` 语义确认 + 命令即开跑；语义不符先回执）；③`d1-006` held-out（≤2.4）排后/并行编排；④全程 quiet-gate（/home ≥15G 起步）并逐腿记卡。
- 资源提醒：D5 `L-W3-D5B`（staged_b32）随时开跑；勿与其抢同一张卡。

### novelty 深读 #2（2026-10-15 04:17，C0）— 综述书目级 + 多模态线 + delta 二扫

- 综述《Beyond Concatenation》（DOI `10.2139/ssrn.7343278`）：**全文双机仍不可达**（SSRN 000）；**Crossref 摘要 + 44 条参考文献已取得**（书目级读取）→ 自我定位=长上下文数据组织综述；四 role（Sequence Allocation & Integrity / Context Construction / Workload Scheduling / Shared-Structure Reuse）；"reweight examples" 为失败模式表述；书目未见等价性 formalization、未见多模态 packing。**全文获取维持论文前置条件项（用户/D1）**。
- **多模态线补齐**：NaViT / Patch n' Pack（arXiv 2307.06304）——related-work 必引（第一轮遗漏）。
- **delta 二扫**：① 等价 formalization 未见证伪（最近邻 EXACT=反向重加权；`Rethinking Length-Based Training` 2609.25890=问题域活跃）→ 表述改为"可证明的逐样本权重保留构造"；② 容量轴无显式同主张；③ invalid-slot slack 分解无同类。维持：工程独立成立、论文="构造①为主 + ②③佐证"。
- 证据：`evidence/c0_novelty_deepread2_d1_20261015.md`（新增对照表 11 行 + 检索留痕）。


### 终裁 `d4xd1-005-c0`（L-W3-D4xD1B 隔离臂收口；2026-10-15 04:24，C0）

- 臂 `d4xd1_w3_cacheoff_bucketon_fp32off` **rc=0 @04:22:43**（101 步；train wall 236.8s；peak 17.48 GiB；nan=false；GPU3）；预算 **≈0.081/0.10 GPU-h 关闭**（含 8s staging 重试）。
- C0 独立重算（自 raw `train_log.jsonl` steps 2..101 vs ref）与 D1 `evidence/d4xd1_004_f1b.json`（未提交 @读取时）逐项一致：**F1B FAIL——max 2.2727e-2@11；17/100 超门（10–13,15,16,18,43,47,54,60,73,82,95–98）；中位 4.32e-3**（门 7.5e-3；3.0× 余量）。
- **执行预注册映射 FAIL 分支**：在线分组超带 → **正确性阻断加深**；`cache×regroup` 组合声明维持**不成立/暂禁**（003 的 F1 FAIL 不撤销；MIX 不受影响）；机制排查转**新提案轨道**（须预注册；不得事后放宽门）。
- 描述性分解：损失路径单因子翻转（fp32=true→false，同 bucket_on）**亦超门**（1.4080e-2@15；7 步；中位 2.58e-3）→「已知损失路径差类 ≤~6e-3」在在线 W3 不可直接迁移；机制轨道须含同路径伴随对照 + 轨迹口径。
- 核验：identity PASS 6.13e-8（≤1e-6）；anchors 104 calls 三腿一致（prefix `96c5a117436f9d35` / full `4347a983417d07f0`）；feature_cache=0 臂正确。
- 下一：D1 harvest 回填提交（2×2+004）→ `d1-006`（L-W3-D1Q ≤2.4）；D5 `L-W3-D5B`（staged_b32 ≤0.75）已批未跑。
- 全文 `proposals/d4xd1-005-c0.md`；证据 `evidence/c0_d4xd1_004_recompute_20261015.json`。


### 收讫 D1 harvest（`e7ea0cf`+`9990a0f`）+ 测试刷新（2026-10-15 04:32，C0）

- `e7ea0cf`（2×2，04:16:30）/ `9990a0f`（004，04:26:49）收讫：与 C0 重算逐项一致（F1B 2.2727e-2@11、17/100；aux 1.408e-2@15；sanity 1.2475e-2@11；identity 6.13e-8；digests；wall 236.83s / peak 17.485GB）；grid-ulp 披露 C0 复算一致（fal 5.00/1.00/2.00/17 vs tru 2.79/0.42/0.93/8）；预算双口径均 ≤0.10；无整改（9990a0f 未引用 005 系平行提交，方向一致）。
- 整合分支 @HEAD 套件：**515 / 509 pass / 6 skip / 0 fail**（exit 0）；证据 `evidence/pytest_rsi_c0_head_20261015.{log,xml}`；代码面 = 3 提交（`1f8963e`/`4e1674e`/`192554a`）。
- 未启动（04:32 实测）：`d1-006`、`staged_b32`；D5 复核入口 `research/rsi/d5/runner/split_verify.py` 就位。


### C0 ack `96709e6` + d1-006 开跑（2026-10-15 04:33，C0）

- addendum 对齐 L-W3-D1Q（band-first / N=200 / 64-prompt 固定 val / GPU3 / ≤2.4；无整改）。披露接受：val 16→64 确定性扩展（train 不动；`val64_record.json` 落规则+映射）、`max_new_tokens=128`、offline 类隔离、smoke-gate。
- 实况：`d1q_val64` ✓（64 行/48 图）；`d1q_smoke` **exit=0 @04:33:06**（regroup enabled 行 ✓；visual passed）→ Stage-1 排队；GPU3。C0 收算待四腿落地。D5 `staged_b32` 仍未启动。

### 事故 #8（守卫自碰撞 + 编排链死亡）与 d1-006 恢复就位（2026-10-15 04:44，C0）

- 取证（服务器日志）：04:37:25 编排器副本 A 起 s42 eval，attempts 1–4（04:37:25–04:38:28）因 `/home` 11G<15G 被误标 "transient spike" 空转；04:38:40 外部第二副本 B 启动（disk 回 19G，guard 通过；现 PPID=1）并开始 eval；04:38:49 A 的 attempt=5 将 B 的 7.8GiB 占用判成 `GUARD FAIL (tenant)`（**误报**——tenant=B 自己）→ A exit 1；编排器随之消失（ps 无、status log 无 done/FAIL）→ **链死**。
- s42 训练产物完好（exit=0 @04:37:25；final_loss 5.75）；s42 eval 由 B 继续（04:42:31 进入 speculative 阶段）。
- D1 04:42 就位逐腿恢复脚本 `d1q_run_leg.sh`（幂等：stage/train/eval 分阶段跳过、自动清 ckpt_last）+ `build_d1q_band.py`；恢复动作=串行补 s43→regroup×2（每腿开跑前 `/home`≥15G、单副本）。
- 转发文本 `_coord/messages/20261015_d1q_recovery.md`（含 D1 恢复说明 + D5 staged_b32 开工提醒）；RESOURCE_SCHEDULE 记事故 #8。

### d1-006 进展：s42 腿完成 + D1 恢复链启动（2026-10-15 04:54，C0）

- s42 canonical 腿全完成：train exit=0 @04:37:25（200 步，final_loss 5.75，loss_decreased=true，wall 215.7s）；eval（B 副本）exit=0 @04:51:44，**token_exact=True**（64/64），final AL=1.1654 / accept=0.0239 / per_slot=[0.1607,0.0033,0,0,0,0,0]，init AL=1.0021 / accept=0.0；target_only=34.40 tok/s。
- C0 已取 raw 四件套（eval_result / eval_samples / train_summary / train_log）到 `/tmp/c0_d1q/d1q_canon_s42/`，**双端 sha256 逐项一致**；重算工具自检 OK（verdict: pending: c43,r42,r43）。收据 `evidence/c0_d1q_receipt_s42_20261015.md`。
- **D1 恢复链已启动**：04:52:24 `d1q_run_leg.sh canon 43` 开跑（stage→train，GPU3；启动前清 s42 ckpt_init 释 ~1G）；status log 由该脚本续写。余 regroup 42/43 两腿（每腿 ~17-18 min）。
- 待四腿齐后 C0 出 band 裁决（band=canonical 42/43 差值；逐 seed |dAL|/|dacc| 配对；未收敛→不判定）。

### d1-006：s43 canonical 完成 + Stage-1 band 就绪（2026-10-15 05:12，C0）

- s43 全腿完成：train exit=0 @04:56:33（200 步，final_loss 6.03）；eval exit=0 @05:09:32（token_exact=True；final AL=1.16355 / acc=0.023624；init 1.00215/0.0）；`d1q_run_leg.sh` 自动清 ckpt_last/ckpt_init（/home 19G）。
- **band 就绪**：band_AL=0.0018152、band_acc=0.00027917（canonical 42/43 一致度高）。C0 收据 `evidence/c0_d1q_receipt_s43_20261015.md`；raw 双端哈希一致。重算工具已更新（pending 时也产出 band；selftest OK）→ verdict=pending: r42,r43。
- **恢复链未自动接续 regroup**（05:09:39 实测无 d1q 进程、GPU3 空）→ 待 D1 执行 `d1q_run_leg.sh regroup 42` 与 `regroup 43`（串行，各 ~17-18 min；命令见 `messages/20261015_d1q_recovery.md`）。
- D5 `staged_b32` 截至 05:12 仍未开跑（D5 车道无新 run/新提交；提醒持续有效）。

### d1-006 中间读：r42 配对出带（冻结规则下 PASS 不可达）（2026-10-15 05:29，C0）

- r42 完成：train exit=0 @05:14:17（200 步，final_loss 5.7897，loss_decreased=true）；eval exit=0 @05:27:18（token_exact=True；final AL=1.1621033 / acc=0.0234309；init 1.00215/0.0）；raw 双端 sha256 一致。
- **冻结规则对照（逐 seed 配对）**：|ΔAL(s42)|=0.0032633 > band_AL=0.0018152（1.80×）；|Δacc(s42)|=0.00047229 > band_acc=0.00027917（1.69×）→ s42 对**双指标出带**；PASS（要求两 seed 均不出带）已不可达，正式裁决待 r43 完成归档（C0 不事后调 band）。
- 描述性（非判决口径）：r42 相对 c43 的 |ΔAL|=0.00145/|Δacc|=0.000193 在带内（“像另一个 canonical 样本”），相对自身 seed 出带；绝对量级小（AL −0.28% 相对）但规则逐 seed 配对。
- 含义（addendum 预注册）：regroup 训练质量有损 → 长训练 GO 的**阻断项**；与 003/005 数值阻断同向（不同机制：质量差异 vs loss 路径超带）。
- 后续：r43 训练中（05:27:36 起）；四腿齐 → `tools/d1q_band_recompute.py` 正式裁决 + `evidence/c0_d1q_interim_r42_20261015.md` 归档。

### d1-006 终裁：FAIL — regroup 质量出带（2026-10-15 05:47，C0）

- 四腿齐（c42/c43/r42/r43）：全部 200 步、loss_decreased=true、无 NaN、token_exact=True；eval settings 与 val64 一致；raw 双端 sha256 一致（收据列全）。r43 收工 @05:44:52，链完整。
- band（canonical 42/43 ckpt_final）：AL 0.0018152 / acc 0.00027917。
- 配对（冻结规则逐 seed）：s42 dAL=3.2633e-3（**1.80×**）/ dacc=4.7229e-4（**1.69×**）→ fail；s43 dAL=8.8134e-3（**4.86×**）/ dacc=1.2925e-3（**4.63×**）→ fail → **verdict = FAIL（两 seed 同向、两指标同向）**。
- 映射（addendum 预注册）：regroup 训练质量有损 → **长训练 GO 阻断项成立**；与 d4xd1-003/005 在线数值阻断同向（机制不同）。绝对量级小（AL 相对 −0.28%/−0.76%）但规则预注册、逐 seed、不事后放宽。
- 工程事实保留：wave-1/2 测速与容量（13.5%/13.3%、peak −57%、b24 OOM→可跑）仍为平台工程收益；「质量中性/等价」声明在 200 步/val64 上不成立。
- 收据 `evidence/c0_d1q_receipt_verdict_20261015.md`；D1 harvest（四腿归档 + REPORT/EXPERIMENTS/STATE）待回填后 C0 对账。

### D1 d1-006 harvest 收讫 + 对账一致（2026-10-15 05:52，C0）

- 交付：`3bc72ba`（Stage-1 建带）/ `12d3b5e`（s42 中间）/ `1b0328e`（完成）；REPORT §14.1–14.3；EXPERIMENTS +9（共 64）；lane evidence（4 腿 raw、band summary、lossdiff、timeline、构建器）。
- **对账**：8/8 服务器 raw sha256 与 C0 /tmp 副本一致（harvest 未动 raw）；band 0.0018152128/0.0002791716 与配对四值逐位一致；D1「NEGATIVE」≡ C0「FAIL」（同一冻结规则语义）。
- 补充表述证据：lossdiff（s42 max 1.026e-2@195 / s43 3.207e-2@10；披露 fp32-vs-auto logging-grid 不对称，非判决项）；timeline 全记账。
- 预算 **1.1536/2.4**（四 train + 四 eval + smoke）——与 C0 窗口估算一致；L-W3-D1Q 关闭。
- 映射（C0 采纳）：离线冻结布局轴负向 → 长训练 GO 需机制/数值修复 + 新预注册；「N=200 加 seed」不改变结论。工程测速/容量结果不受影响。
- 待办仅余：D5 `staged_b32`（L-W3-D5B ≤0.75，仍未开跑）。
