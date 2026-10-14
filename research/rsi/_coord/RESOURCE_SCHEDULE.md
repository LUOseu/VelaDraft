# 资源与租约 RESOURCE_SCHEDULE v0.1.0

测量时间：2026-10-14 23:06–23:10（Asia/Shanghai）。空闲状态动态变化，每次开跑前必须实测。

## 1. 节点

### a6000xuyang（主算力；ssh `oab-remote`，10.65.14.8，user lyh）

- GPU：4× RTX A6000 48GB。实测（23:06）：**GPU0 空闲**（0%，15MiB）；**GPU2 空闲**（0%，18MiB）；GPU1 外部占用（18.8GB，100%）；GPU3 外部占用（44.3GB，100%，dongzp）。**不能假定四张卡可用。**
- 共享：主机有其他用户活跃进程（dhz、wmk、dongzp 等）；不得结束/干扰他人进程。load avg ~7.6–9.1。
- RAM：251GB 总，~205GB available；swap 7/7GB 已用满。
- 存储：/home 15T，**29GB free（100% used）**；历史记录存在 host 级写回停顿（同协议 wall 波动最高约 1.7×）。**磁盘是本轮第一约束。**
- 既有测试环境：`/home/lyh/vlm-dflash/.venv`（py3.11，pytest 9.1.1）（引用、未复核）。

### 本地 Mac（M4，16GB）

- 仅 CPU 测试/文档/分析；无多机/大卡资源。不假设 A100 等额度。
- 本机测试解释器：`/Users/Admin/grounding-parallel/vlm-dflash-dflare/.venv/bin/python`（py3.12，torch 2.10，pytest 9.1.1）；在各自 worktree 根目录运行 pytest 时以 cwd 导入（C0 已在 veladraft-c0 验证）。

## 2. 服务器目录（现状）

- `/home/lyh/VelaDraft-systems`：上一 lane 快照 @e15256d（含 lane 补丁备份 `lane_patch_backup_20261014`）；**只读**。
- `/home/lyh/vlm-systems`：上一 lane 脚本/日志/evidence；**只读**。
- `/home/lyh/rsi-d1`（含 evidence/、scripts/）、`/home/lyh/VelaDraft-d3`（空 repo）：D1/D3 脚手架（引用、未复核）。
- 本程序新目录约定：脚本/证据 `/home/lyh/rsi-<role>/`；仓库工作树 `/home/lyh/VelaDraft-<role>/`。不得覆盖他人目录。
- 实际落位（23:29 只读核实）：D1=`/home/lyh/rsi-d1`；D2=`/home/lyh/d2-rsi`（repo/ 子目录）；D3=`/home/lyh/vlm-d3` + `/home/lyh/VelaDraft-d3`；D4=`/home/lyh/rsi-d4`（跑腿进程 cwd=`/home/lyh/vlm-dflash` 共享 venv）；D5=`/home/lyh/vlm-d5`（repo/ 子目录）。

## 3. 租约纪律

- 同一 GPU 同时只允许一个 VelaDraft 重负载任务；开跑前 `nvidia-smi` 实测该卡 0% util 且 >40GiB free；记录卡号与外部占用比例。
- 正式计时：选最安静的卡；同一实验两腿尽量同卡；不同卡或存在外部占用时必须在报告中披露。
- 不得 kill 他人进程、改全局环境、写他人输出目录；**空闲显存不是占用许可**。
- 磁盘守卫：启动前 >10GiB free；单任务净新增 ≤2–3GiB；harvest 后 prune checkpoint；<4GiB free 时停止提交 GPU 任务，改做 CPU 分析。
- 每个 run 在 `EXPERIMENTS.jsonl` 记录 device/contention/disk 状态；等待时间计入 GPU-hours。

## 4. 预算（wave 1 硬上限；增加须证据）

| 轮次 | 上限 | 依据 |
|---|---|---|
| W1-D1 两轴 probe | ≤1.5 GPU-h（23:17 由 1.0 上调：+W1/W2/W3 真实 probe stores；见 REGISTRY §7 对 D1 的应答） | R2 tensor probes + 三个 128–256 行 probe store；旧 lane 参考：129 行 × 6 epochs 两腿 ≈0.24 GPU-h |
| W1-D2 N/U + teacher 成本 | ≤1.0 GPU-h | 计数与成本占比，128–256 样本 |
| W1-D3 attention/mask profile | ≤1.0 GPU-h（23:17 由 0.5 上调：需 warm/编译基线与 span 网格；依据 D3 round-1 计划 ≤4 短 run） | 仅 profiling；未证明瓶颈（attention+mask ≥20% 整步）不写 kernel |
| W1-D4 收尾/容量探针 | ≤0.5 GPU-h | 容量/命中率为主，优先 CPU/trace |
| W1-D5 静态流水线/trace 可行性 | ≤1.0 GPU-h（单卡；23:12 由 0.5 上调，依据 D5 handoff 的 regen scaling probe） | 静态 trace + 单卡模拟；不建多 GPU 系统 |
| **wave 1 合计** | **≤4.0 GPU-h** | 当前 2 张空闲卡 + 紧张磁盘下的可执行上界 |

### 已发布租约

| 租约 | 对象 | 卡/条件 | 有效期 | 备注 |
|---|---|---|---|---|
| L-W1-D5 | D5 | 任一实测 quiet 卡（开跑前实测 util≈0%、>40GiB free；同一时刻只允许 D5 一个任务） | 2026-10-14 23:12 → wave 1 结束或他人占用出现 | 单卡 provisional ≤1.0 GPU-h；23:11 实际使用 GPU1（当时 4.8GB/2%，满足 quiet 条件） |
| L-W1-open | D1/D2/D3/D4 | 任一实测空闲卡（D1≥D2≥D3/D4 优先顺序由任务到达决定；同卡不并发重负载） | 同上 | 各角色 wave 1 上限见上表 |
| L-W1-D5R | D5 | **GPU3（23:27:36 实测 quiet：18MiB/0%）**；GPU1 于 23:27 被外部 xs 二次占用→不再可用。**强制单副本**（见事故 #3） | 2026-10-14 23:27 → 重跑链完成或目标卡被重新占用 | regen b8/b16/b32 重跑（计入 W1-D5 ≤1.0 GPU-h 上限，含 23:25–23:27 重复窗口 ≈0.08 GPU-h 浪费）；**23:37 链完成（8/16/32/8r 全 exit=0）→ 租约关闭** |
| L-W1-D1 | D1 | **GPU3**（D1 自选；23:38 实测 quiet；若被外部占用改用 GPU2 并登记） | 2026-10-14 23:42 → R2 tensor probe + W1/W2/W3 stores 构建完成或卡被占 | 计入 W1-D1 ≤1.5 GPU-h；单卡串行、逐腿 quiet 自检；含 R3 步成本 A/B（5 臂×10–20 批 ≈2–5 GPU-min/组）；完成后让给 D3（可选 b16）/D2 排队 |
| L-W1-D2R | D2 | **GPU2**（23:42 空闲；开跑前实测） | → 单次干净 step profile 完成（≤15 GPU-min） | 重做 23:24–25 的 contended run（本 run 须标 `contended=false`）；计入 W1-D2 ≤1.0 GPU-h；**23:48 步级两臂完成（fwd -5.5%）→ 关闭** |
| L-W1-D1-R2 | D1 | **GPU2**（首选；开跑实测 quiet）｜GPU3 备选（dhz 已结束，若沿用须实测披露） | → W2/W3 store + 正式 R3 A/B 完成或卡被占 | **W1 已建成（23:59:42，kept 191/192）**；剩余 W2/W3 store 与 R3 A/B 同租约；前置 free≥10G（当前 22G）；逐腿 quiet 自检、**跑后清 train ckpt（~4.6G/腿）**；计入 W1-D1 ≤1.5 GPU-h |
| L-W2-D4 | D4 | 任一实测 quiet 单卡（util≤5%、mem_free≥40GiB、磁盘≥10G；非他人有效租约卡） | → capped-process real-sensor A/B 完成或卡被占 | **条件批准 ≤1.5 GPU-h**（d4-001 条件照旧：MemoryMax/ulimit 隔离、预注册压力轨迹+证伪条件、只报 wall/内存/命中率、自清 ckpt）；**00:20 起预授权窗口**：预计 ≥1.5h 即可开跑，D4 自记卡号+quiet 证据；单卡窗口优先 D4、双卡优先 D5-G2；平台工程验证定位，非论文增量 |
| L-W2-G2 | **D5 步 3–4（静态 ladder）优先**；D4×D5 producer 重测次之 | 2 张实测 quiet 卡（同上 quiet 口径；非有效租约卡） | → 预登记 leg 集完成或卡被占 | **00:25 版档位（取代 00:20 值）**：Tier 0 smoke ≈0.2 GPU-h（窗内第一门；parity 失败即停 S3 线）/ Tier A `staged_b8+sync_b8+s3_b8` ≈3.0–3.2 / Tier B `+sync_b32+s3_b32` ≈4.2–4.4 / Tier C 全矩阵 ≈6.6–7.2。**预授权：双卡 quiet 即开 Tier 0→A**；A 后两卡仍静且预计 ≥2.5h → 可续 B；**C 需 C0 确认**。D5 自记卡号+quiet 证据入 EXPERIMENTS；冲突时 C0 回收。 |
| L-W2-D1 | D1 wave-2 配对（双轴重排） | **单卡；优先 GPU3**（D5 链动态对排除 3）；任一实测 quiet（util≤5%、mem_free≥40GiB、磁盘≥10G） | → 配对集完成或卡被占 | **批准 ≤1.5 GPU-h**（00:47）：W3+MIX、k=4、臂 {canonical,split,token,cost,grid}、≥3 交替、100–200 更新 + capacity demo（batch16@sub4 vs batch8，单列）；条件=双臂同表 replay、数值开关一致、grad 口径修正（`d1-001-c0`）；逐 run 记卡号/co-residents；与 D5 不共卡（D5 用 1/2/0；D1 用 3）；冲突时 C0 回收。**02:20 关闭：实测 ≈0.90/1.5（含 capacity demo ≈0.17），demo C0 逐值核验 ✓** |
| L-W3-D1R4 | D1 R4 sub1 边界 A/B（`d1-004`） | **GPU3 优先**；或任一实测 quiet 单卡（同 quiet 口径）；与 D5 的 1+2 不共卡 | → 2 cells 完成或卡被占 | **批准 ≤0.2 GPU-h**（02:25；`d1-004-c0`）：`--sub-batch 1`、W3+MIX seed 42、arms canonical,cost、parity 0/50/100、100 timed 批；预注册判决 ≥1.135 → R4 正分支 / 否则边界关闭；≈0.10 GPU-h 预估；与 C0 2×2（计划 GPU0）冲突时本短实验优先。**02:33 完成 rc=0（实测 0.108 GPU-h）→ 负分支关闭（`d1-005-c0`：W3 1.1296 / MIX 1.0199 < 1.135）** |
| L-W3-D4xD1 | D1（执行全部 8 腿；另半 = D4 机制产品路径集成运行，D4 线不重开） | **GPU3 优先；8 legs 同卡**；不共 D5 的 1+2；D5 新腿需要 3 时让行 | 即刻 → 8 legs 完成或卡被占 | **批准 ≤1.5 GPU-h**（02:38；`d4xd1-001-c0`）：{cache_off, cache_on=16GiB} × {bucket_off, grid/sub2}、online、W3+MIX seed 42、1 warmup + 100 updates；F1–F4 判决冻结；**02:45 fast-path 放行**：addendum 三条（命令/parity 捕获法/anchor 机制）落 handoff 即开跑 |
| L-W3-D1Q | D1 held-out 质量（`d1-006`） | GPU3（2×2 之后同卡顺序）；quiet-gate 同 L-W2-D1 | → 4 训练腿 + eval（staged）完成或卡被占 | **接收 ≤2.4 GPU-h**（02:45；条件化）：Stage-1 canonical×{42,43} 建带（≤1.2）→ Stage-2 regroup×{42,43} 比较（≤1.2）；先带后比；`eval_dflash.py` 64 prompts / 固定 val；<200 步不收敛 → 不判定 |

### 观察记录（C0 只读，未独立复核）

- 2026-10-14 23:19 — 四卡全部在用：**GPU1** = D5 regen 批量 probe（`/home/lyh/vlm-d5/repo`，--limit 128 --steps 1，regen.batch_size=8，pid 3311115）；**GPU2** = D3 attention/mask profiler（profile_drafter_step.py，limit 16/10 steps，pid 3317651）；**GPU0** = 外部用户 xs 的 Qwen2.5-Math eval（pid 3298367）；**GPU3** = 外部 dongzp（pid 3094389，44.3GB/100%）。无空闲卡 → D1/D2/D4 的 GPU probe 需等窗口。`/home` 29GB free。
- 2026-10-14 23:17 — D3 profiler 已退出（GPU2 释放，18MiB/0%）；GPU0 外部任务结束（287MiB/0%，近空闲）；D5 probe 仍在 GPU1（11.8GB/2%）；GPU3 外部继续（44.3GB/100%）。`/home` 27GB free（D5/D3 小量写入）。D1/D2/D4 可用 GPU0/GPU2 窗口。
- 2026-10-14 23:18 — GPU1 = D5（10.4GB）+ **D2（27GB，并发，未登记租约）**；GPU0 = 外部 xs（14.9GB）+ D2（264MiB）；GPU2 空闲；GPU3 外部。重负载定义：alloc > 4GB 或 util > 50%；同卡禁止两个未声明的重负载任务。受影响的计时标 `contended`。
- **23:19 事故（D5×D2 共卡）**：D5 的 regen 批量 probe（GPU1，CUDA_VISIBLE_DEVICES=1）b8 腿 **OOM 失败**（`torch.OutOfMemoryError ... Process 3313721 has 26.36 GiB in use`，即 D2 同卡）；b8 exit=1；b16 于 23:18:47 启动后进程无退出记录消失（日志止于权重加载）。**处理：b8 作废（contended/OOM）；b16 按未完成处理；整链需独占卡重跑；失败 run 计入 GPU-hours、不计入有效结果。**
- 2026-10-14 23:21（当前占用）：GPU0 = 外部 xs（15.8GB，98%）；**GPU1 = D2 新 run（显式 `--device cuda:1`，pid 3331562）**；**GPU2 = D4 W1 legs（screen `d4w1`，cache-off 腿 `s2b/off`，启动前 quiet 自检通过）**；GPU3 = 外部 dongzp（44.3GB，99%）。`/home` 19GB free（较 23:14 的 29G 明显下降，注意 prune）。D5 重跑需等独占卡。
- 2026-10-14 23:22 — **GPU3 空闲**（外部任务结束，18MiB/0%）；GPU1 = D2 probe（31.4GB/100%）；GPU2 = D4 leg（8.0GB，加载/运行中）；GPU0 = 外部 xs（97%）。`/home` 24GB free。**发布 L-W1-D5R：GPU3 给 D5 重跑**（完成后优先让给 D1）。更正早前记录：GPU3 外部用户为 **lrz**（sglang-spec，进程路径 `/home/lrz/...`），此前误记为 dongzp。
- 2026-10-14 23:25 — 变动：**GPU3 被外部重新占用**（pid 3338252，13.8GB；非 lyh 进程）；**GPU1 空闲**（D2 的 `probe_js_base` 已成功完成：fwd_median 372.1ms、final_loss 6.43，日志 `/home/lyh/d2-rsi/probe_js_base.log`）；GPU2 = D4 leg（26.2GB/79%）；GPU0 = 外部 xs（97%）。**L-W1-D5R 改指 GPU1**；D5 重跑尚未开始（`probe_progress.log` 仍止于 23:18）。
- 2026-10-14 23:26 — **事故 #2（D3×D2 共卡）**：D3 `mb2` b16 行 OOM（`mb2.log`：`Process 3342133 has 29.28 GiB in use` = D2 `probe_step_profile`，23:24–23:25 未登记于 GPU1）；D2 该 run 双向 contended（自身计时与对 D3 影响均记录）。mb2 其余行有效。
- 2026-10-14 23:25–23:27 — **事故 #3（D5 重复启动）**：D5 重跑脚本两次启动（23:25:04 / 23:25:35，均 DEV=3、同写 `runs/regen_rb_8`，脚本内 `rm -rf`）→ 双副本互相覆盖输出，该窗口结果作废；两副本在 23:27:36 前被终止（无退出记录）；23:27:36 单副本在 GPU3 实测 quiet 后重启。浪费 ≈0.08 GPU-h 计入 D5 上限。**流程缺陷记录**：脚本按“最大空闲显存”自动选卡（不查 util、不再对照租约）——选卡规则须修正为“租约卡 + 实测 quiet + 单副本锁”。
- 2026-10-14 23:29（当前占用）：GPU0 = 外部 xs（15.9GB/96%）；GPU1 = 外部 xs 第二任务（15.9GB/98%）+ D3 `w0_log10` spans（~20GB，contended）；GPU2 = D4 `cap10` leg（26.2GB/75–87%）；GPU3 = D5 单副本（8.3GB/47%）。**无空闲卡**；`/home` 15GB free（D2 `runs/` 8.7GB 待 prune；D4 每腿自清 ckpt 已核实）。
- 2026-10-14 23:30 — D4 `cap10` 腿 exit=0（wall=209s；counters 与仿真预测逐值一致：7/767/767/755，peak 87,056,736B）；`off`(301s) 与 `cap10`(209s) 的 wall 差为编译/冷启动顺序效应候选，非缓存收益（hits 7/774）。下一步 cap 腿继续（GPU2）。D5 **leg 8 exit=0**（wall 176s，单副本）→ **leg 16 启动**（23:30:33，GPU3）。`/home` 15G free。
- 2026-10-14 23:37 — **D5 链完成（8/16/32/8r 全 exit=0；warm 腿 121–149s；≈0.16 GPU-h）→ GPU3 释放，D1 窗口开启**（D1 第一优先：tensor probes + W1/W2/W3 stores；D2 次之）。D4：cap25/cap50 均与仿真逐值一致（24/750/718；97/677/612），cap8m 运行中（GPU2）。`/home` 14G free。
- 2026-10-14 23:39 — **D4 W1 legs DONE（5 腿全 exit=0，counters 与仿真逐值一致）→ GPU2 释放**。当前空闲：**GPU2 + GPU3**（GPU0/1 外部 xs）。D1 窗口=双卡可取（L-W1-open 首位；建议先 tensor probes 后 store 构建，单卡串行以免自扰）；D2 次之；D3 可选 b16 补跑；D5 已收链。`/home` 14G free。
- 2026-10-14 23:42 — GPU2/GPU3 仍空闲；发布 **L-W1-D1（GPU3）** 与 **L-W1-D2R（GPU2；D2 干净 step profile ≤15 GPU-min）**。D2 `probe_js_prof` ckpt 仍 4.6GB 未清（/home 14GB free；各腿开跑前须 >10GB）。D3 可选 b16 补跑排 GPU2 之后（非必需）。D5 链完成、卡已交还。
- 2026-10-14 23:44 — D2 `ab_base1` 于 23:42 起在 **GPU3** 运行（probe 登记；核定卡为 GPU2）→ D1 开跑前须实测两卡；GPU2 空闲。D5 已提交 round-1 REPORT（H2 负结果；静态分裂=wave-2 条件项）；D3 收尾清理完成（footprint 4.5MB）。`/home` 15G free；D2 ckpt（4.6GB）仍未清——再次提醒。
- 2026-10-14 23:50 — D2 基线臂完成（23:44），GPU3 释放。**`/home` 降至 11G free**：D2 `runs/` 14GB（新臂 ab_base1 4.6GB + probe_js_prof 4.6GB + 其他），已要求逐臂清理后再开下一臂；D1 store 构建（1.2–2GB/个）开跑前须确认 >10G。GPU2/GPU3 空闲；origin/main 复核 = `7eab136`（BASE 无漂移）。
- 2026-10-14 23:47 — **事故 #4**：D1 W1 构建（23:45:27 GPU3）与 D2 `ab_dedup1`（~23:46:2x GPU3）同卡 → D1 于 23:47:01 被外部 SIGTERM（exit=143，来源未证）；D2 臂 23:47:38 完成。GPU3 随后空闲；GPU0/1 外部 xs。
- 2026-10-14 23:49 — **磁盘危机：free=5.0G**（D2 五目录 ckpt ~22.9GB 为可回收主体；D1 W2/W3 与重启全部被堵）。待 D2 prune 或用户清理后恢复放行；GPU2/3 空闲待命。
- 2026-10-14 23:51 — **GPU3 被外部 dhz 占用**（新任务 `reproduction...racer_asymmetric_readout12`，13.75GB/运行中）→ 程序可用卡仅 **GPU2**；GPU0/1 仍为外部 xs。磁盘 5.0G（未变）。D1 重启与 D2 后续一律以 GPU2 为准（开跑前实测）。
- 2026-10-14 23:52 — D1 的 R3 A/B 冒烟在 GPU3 空闲窗口完成（23:49:0x–23:49:4x，冒烟级；dhz ~23:50 起占用 GPU3）。磁盘仍 5.0G（D2 prune 第 2 次要求；若 D2 无窗口→请用户代删 `runs/*/train/ckpt_*.pt`）。程序 GPU 排队：GPU2 空闲待 D1 W1 重启（等磁盘）。
- 2026-10-14 23:53 — GPU3 外部 dhz 任务结束（23:52:10 实测 18MiB/0%）→ 可用卡 **GPU2+GPU3**。随即 **D2 `phase_base1`→GPU2**（18.4GB/100%）、**D1 smoke3→GPU3**（21.4GB）——各占一卡、无共卡；两 run 均在 free 4.1–5.0G 启动，**未达 >10G 守卫**；`/home` 4.1G free。
- 2026-10-14 23:53 — 磁盘第 3 令升级用户：`rm -f /home/lyh/d2-rsi/runs/{ab_base1,ab_dedup1,probe_js_base,probe_js_prof,probe_js_smoke}/train/ckpt_*.pt`（≈23.4GB；保留 json/log/summary 与运行中 `phase_base1`）→ 预计 free ≈27G；为 L-W1-D1-R2 硬前置。
- 2026-10-14 23:56 — **事故 #5：0 字节事件**（D2 `phase_base2` ckpt_init 截断 454MB@ENOSPC 崩溃；窗口 ~23:55:5x–23:56:2x，共享卷写入普遍受影响风险）。**C0 紧急清理**第 3 令五目录 stale ckpt → free **0→28G**；D2 随后自清 phase_base1/2 目录。租约：**L-W1-D1-R2 生效**（D1 23:56:43 起 GPU3 运行 W1 build，启动自检 free=28G/gpu_idle=0%）；GPU2 空闲；GPU0/1 外部 xs。后续任何启动仍需 free≥10G + 逐卡实测。
- 2026-10-14 23:59 — **D1 W1 store 建成（exit=0）**：192→kept 191、features 1.8G（manifest+6 shards；已登记共享资产）、run-dir 6.4G（train ckpt 4.6G 建议跑后清）；wall≈179s。磁盘 22G free。
- 2026-10-15 00:00 — **D2 phase 配对首批**（base4 vs dedup4，mb4：fwd 188.3→189.3 未见 -5.5% 复现；loss 三腿 bitwise 同值；ckpt KB 级自清生效）；**GPU2/GPU3 空闲**（GPU0/1 外部 xs）。**D4 d4-001 ACCEPTED（条件）** + **L-W2-D4 ≤1.5 GPU-h 条件批准**（等窗口）；D1 W2/W3+R3 同租约排队。
- 2026-10-15 00:05 — **GPU0 空闲**（外部 xs 完成）→ 门开给 L-W2-D4 与 D1 W2/W3（按实测 quiet）；GPU2 = D2 pin 系列（运行中）；GPU3 = 外部 dhz racer repro 回归（~14.3GB）。D1 清 W1 train ckpt 生效（free 27G）。
- 2026-10-15 00:06 — D2 pin 2×2 完成（fwd -6.8% 复现、loss bitwise；GPU2 已释放）；**D1 W2 build 于 GPU3 运行**（00:05:45 起；guard 27G / gpu_idle=0% ✓）；D5 lane 41/41 C0 复跑 ✓（CPU）。当前：GPU0 空闲 / GPU1 xs / GPU2 空闲 / GPU3 D1；磁盘 27G。
- 2026-10-15 00:09 — D1 W2 store 建成（00:08:25 exit=0；kept 144/144；自清 ckpt，free 21→25G）；**W3 build 00:08:39 于 GPU3 启动**（regen 阶段较长，多轮真实行）。
- 2026-10-15 00:12 — D2 **dfly 系列**：base1/2 完成（fwd 215.1/213.3、bwd 382.8/382.3；42 张量 dfly 图）；**dedup 腿 rc=1**（`algorithm.dfly.teacher_dedup` 不受支持——服务器副本缺其 factory 补丁；同步问题非机制异常）→ 待同步重跑；GPU2 随后空闲。
- 2026-10-15 00:14 — D5 lane 43/43 C0 独立复跑 ✓（CPU、no-write）；W2 store 复合摘要登记（`5980f77b…`）。当前：GPU0 外部 ≈18GB/97%；GPU1 空闲；GPU2 空闲；GPU3 D1 W3；磁盘 25G。
- 2026-10-15 00:20 — 更新 L-W2-D4 / L-W2-G2 行为**预授权窗口**（见上表）；D1 W3 继续（GPU3）。
- 2026-10-15 00:33 — **D2 dfly 系列完成校核**：dedup1 首败=服务器副本缺 factory 补丁（同步问题）；chain 重跑 **rc=0**（00:12–00:13），四腿有效：fwd 215.1/213.3→200.9/201.7、bwd 382.8/382.3→363.6/364.4；**12 步 loss 四腿逐值相同**（C0 直读）。D2 交付 REPORT/handoff/EXPERIMENTS/d2-001（自判 STOP）；C0 裁决 d2-001-c0.md。**D1 W3 regen 进行中**（多轮行，~5–7s/行，00:08:39 起）。GPU1 被外部占用（~8GB），GPU2 空闲。
- 2026-10-15 00:30 — **D1 W3 store 建成（00:30:29 exit=0；192/192 全多轮；自清；free 13G）→ R3 正式扫掠即时启动**：k2 四 store（w0 00:21 / w2 00:24 / mix 00:27 / w3 00:30:38，疑用 GPU1→GPU3）+ w1_k2（00:33:52 起，**GPU3**，实测 11.0GB/运行中）+ k4 系列排队（`CUDA_VISIBLE_DEVICES=3`）。GPU0 外部 15.9GB；GPU1 于 00:34 释放；GPU2 空闲。
- 2026-10-15 00:34 — 卡位观察（C0）：w0/w2/mix k2 期间 GPU3 恒稳 8.1GB（=W3 regen 独占），R3 k2 未上 GPU3；GPU1 于 00:22–00:34 有 6.2→8.3GB 占用（疑似 k2 系列）→ **待 D1 在 EXPERIMENTS 逐 run 声明物理卡号与同卡任务**；w1_k2 已实测在 GPU3（pid→uuid 映射）。
- 2026-10-15 00:35 — D1 k4：**mix_k4 完成**（315.4→253.0–254.4ms；peak 15.3→8.2GB，**−46%**），**w3_k4 运行中（GPU3）**；D4 W2 降级腿于 00:30:30 完成（GPU2 释放）；D5 chain2 于 00:33:04 因 smoke_s3 导入 bug 停链（GPU 未再占用）。当前：GPU0 外部 ~15.9GB / GPU1 空 / GPU2 空 / GPU3 D1。
- 2026-10-15 00:38 — **D1 R3 扫掠全部完成**（w3_k4 @00:37；k2×5+k4×2），GPU3 释放；每个 run ≈0.02 GPU-h（D1 条目）。**GPU1/2/3 三卡全空**（GPU0 外部 ~15.9GB）——L-W2-G2 窗口出现（D5 若修好 import 可即时开 Tier 0→A；卡规则照旧）。磁盘 17G free（守卫 ≥10G）。
- 2026-10-15 00:39 — **窗口被 D5 及时使用**：chain2 二次起链（00:38:24）：smoke_canon 复用 ✓；smoke_s3 再败（split 图像暂存缺文件）→ S3 线停、非 S3 legs 继续：**staged_b8 00:38:55 起于 GPU1+2**；sync_b8/sync_b32 排队。当前：GPU0 外部 / GPU1+2 = D5 / GPU3 空。磁盘 17G。
- 2026-10-15 00:44 — D5 staged_b8 运行中（GPU1，regen 600/2000 @00:45，kept 154；GPU2 空备用）；GPU0 = 外部 xs（pid 3548763，15.9GB/97%）；GPU3 00:45 短暂被外部占用后已释放（18MiB/0%）；磁盘 17G。**D1 round-2 收讫**（`56b6563`；REPORT/EXPERIMENTS/`d1-001`/预算 0.70 全量），**wave-2 放行**（`d1-001-c0`）。
- 2026-10-15 00:47 — **发布 L-W2-D1**（见上表）。**事故 #6 登记**（D1 W0-k2 首跑 ~00:20：quiet 自检后启动瞬间被外部进程抢 GPU1 → OOM；等待后重试成功；无他人进程被杀；教训=自检↔launch 间隔最小化 + 启动后 10s 复核；详见 REGISTRY 复核记录）。
- 2026-10-15 00:46 — **D4 W2 收口**（`d4-003`）：方案 B 两腿 F1–F4 PASS（0.062 GPU-h）；lane `a8e762c`；runtime 4 文件 sha1 与集成版一致；C0 核验 ✓ → **merge-ready（平台工程级）**（`d4-003-c0`）；**L-W2-D4 关闭**（未追加）。
- 2026-10-15 00:49 — D1 自启 round-2.1 重复跑（`--parity 3` 多批位）：mix k4 p3 完成（cost 1.170 / grid 1.163，**低于首跑 1.240/1.247**——跨 run 波动，待 ≥3 重复聚合）；w3 k4 p3 运行中（**GPU2**，quiet-guard util≤10%/free≥40GiB ✓，脚本 `run_p3_one.sh`）；其余 cells 排队。**卡位**：chain2 的 sync 腿需卡 1+2——D1 短跑可继续（launch 自检照旧），**wave-2 长跑优先 GPU3 / chain2 结束后再用 2；D5 腿等 pair 时 D1 让行**（L-W2-D1 冲突条款）。
- 2026-10-15 00:52 — D5 chain2 结构确认（只读脚本）：S3 线停 → 余腿 = staged_b8 → sync_b8 → sync_b32（s3_b8/s3_b32 自动跳过）；按手册 wall：staged_b8 预计 ≈01:14 结束（2116–2191s），sync_b8 +≈31min、sync_b32 +≈18min → 链预计 ≈02:05 完成；分配 2 卡计时（Tier 0+A+B 档内）。
- 2026-10-15 00:52 — D1 round-2.1 harness 改进已见（未提交；`r3_ab_step.py`）：`--parity 3` + 新增 `*_vs_ref` 字段（mix k4 p3：loss ≤2.1e-5 / grad ≤5.3e-3 / update ≤7.2e-8，bf16 级）——对应 `d1-001-c0` §3-②。**披露**：p3 短跑（GPU2）与 D5 staged_b8（GPU1）并行（00:45–~01:00）→ **D5 S0 腿 EXPERIMENTS 需记 co-resident**（跨卡轻负载）。
- 2026-10-15 02:20 — D1 提交 R4 评估 `e147a01`（REPORT §12：k4 残余可移除 4.7–5.2%（W3）/6.1–6.4%（MIX）步成本，96–98% 为 invalid-block slack；size-1 fp64 oracle 全 ragged exact；递交租约请求 → 后重编号 `d1-004`，`8c7776c`）与集成交叉复核 `3e29244`（read-only，结论 PASS；建议 (a) all-singleton 用例）。GPU3 空闲。
- 2026-10-15 02:19–02:20 — **事故 #7（工具链缺陷，非租约违规）：D5 S3 smoke 假失败（worker 级 zero-keep）**。chain2 S3 追赶链 02:19:21 起 smoke_s3（run `g2_smoke_s3_1015_0219`，cards 1+2）；worker w0 报 `Regen kept zero samples` rc=1 → 链按政策停 S3 线（02:19:52），Tier 0+A+B 保留。**C0 取证**：canonical smoke 全局 kept=1/16（15 行 too_short），按 canonical 批切分后某 worker 合法 keep=0——worker 不应因自身切片 keep=0 硬失败（merge 后查全局；canonical `run_regen` 容忍批内 zero-keep）。**D5 修复在途**（`split_worker.py` 修改 + `test_split_worker_zero.py`）；修复后 `G2_SKIP_LEGS` 重跑 smoke_s3→s3_b8→s3_b32。
- 2026-10-15 02:25 — **发布 L-W3-D1R4**（`d1-004-c0` 接收 D1 边界实验 + cross-check 回执 + 编号更正说明）；C0 采纳建议 (a)：all-singleton parity 用例入 `research/rsi-c0`（`192554a`；全套 509 pass/6 skip/0 fail）。
- 2026-10-15 02:38 — **R4 收口（负分支）**：两格 rc=0（W3 02:26:44→02:30:22、MIX→02:33:11；GPU3；≈0.108/0.2 GPU-h）；C0 独立重算（raw jsonl，100 timed 批）W3 **1.1296** / MIX **1.0199** < 1.135；峰值 −67%；`d1-005-c0` → R4 以边界结论关闭（sub1 无增量余量；微批开销）。**D3 packed-block kernel 不触发重开**。窗口披露：跨卡共驻（C0 smoke GPU0 02:27–02:30；D5 S3 链 1+2 02:28 起）；GPU3 卡内无共租。
- 2026-10-15 02:38 — **D5 修复生效**：smoke_s3 exit=0（02:30:33，wall 102s）；**s3_b8 02:30:34 起**（cards 1+2、batch8、run `g2_s3_b8_1015_0230`）；s3_b32 其后排队。
- 2026-10-15 02:38 — **发布 `d4xd1-001-c0` / L-W3-D4xD1**：在线缓存 × regroup 2×2（8 legs、≤1.5 GPU-h、GPU3 单卡、D1 执行，另半=D4 机制集成运行；F1–F4 冻结；开跑前 addendum）。= 集成 smoke 的 online 缺口 + wave-3/系统组合叙事判定项。
- 2026-10-15 02:45 — **D1 三连**：R4 收尾/共驻披露/重编号（005→006）收到，R4 全闭环；**2×2 放行（fast-path；L-W3-D4xD1 ≤1.5、GPU3 现空）**；**`d1-006` held-out 质量接收（L-W3-D1Q ≤2.4，staged：先 canonical×{42,43} 建带 → regroup×{42,43} 比较；2×2 后跑）**。

- 2026-10-15 02:59–03:00 — **磁盘告警→自愈（C0 实测）**：s3_b8 尾部写入把 `/home` avail 压至 **5G**（02:58:55 实测）；02:59:11 s3_b8 `exit=0`（wall 1716s、alloc 0.95 GPU-h）后，**D5 prune 脚本 02:59:14 清理其 ckpt/shard（8.0G→19.4G avail）**；02:59:12 `s3_b32` 首试被磁盘守卫跳过（rc=96、run `g2_s3_b32_1015_0259`、vs `g2_sync_b32_1015_0153`），链已按 300s 重试（预计 03:04 通过）。D1：02:54 曾六连重试后 **GUARD FAIL（disk 7–9G；GPU3 本身 quiet）**；**02:59:21 释放后自动重启** `d4xd1_smoke`（3 步；visual_path_check 已过）→ 2×2 8 腿随后。**教训**：s3 腿峰值写 ~10–12G，10G 守卫仅够"起步"；后续腿前建议实测 ≥15G。候选余量清理（用户确认）：zoo-publish-20261013/14 7.5G、fpt_tmp 0.4G。

Wave 2（仅通过证据门槛的 1–2 项）：配对 100–200 更新实验 ≤4 GPU-h/方向；3 seeds canonical 合计 ≤6 GPU-h；参考量级：2K 切片两腿 ≈0.91 GPU-h。均须 C0 批准并在 REGISTRY 记录依据。

预算增加条件：`EXPERIMENTS.jsonl` 记录 + 路线 `STATE.md` 写明下一步信息增益 + C0 在 REGISTRY 备注批准。不满足则 HOLD。

## 5. 时间窗口

- 首波计划：2026-10-14 23:30 → 10-15 12:00（Asia/Shanghai）；夜间静默窗口优先正式计时。
- 连续两轮无新证据或到达预算 → 按协议决策收口（GO/HOLD/STOP/BLOCKED）。
