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
| L-W1-D5R | D5 | **GPU3（23:27:36 实测 quiet：18MiB/0%）**；GPU1 于 23:27 被外部 xs 二次占用→不再可用。**强制单副本**（见事故 #3） | 2026-10-14 23:27 → 重跑链完成或目标卡被重新占用 | regen b8/b16/b32 重跑（计入 W1-D5 ≤1.0 GPU-h 上限，含 23:25–23:27 重复窗口 ≈0.08 GPU-h 浪费）；完成后目标卡优先让给 D1；开跑前实测、单副本、选卡=租约卡 |
| L-W2-G2 | 通过门槛者 | 2×A6000 同时空闲 ≥3–4h 窗口 | **UNCONFIRMED**（GPU1/3 被占） | wave 2 由 C0 复核证据后发布 |

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

Wave 2（仅通过证据门槛的 1–2 项）：配对 100–200 更新实验 ≤4 GPU-h/方向；3 seeds canonical 合计 ≤6 GPU-h；参考量级：2K 切片两腿 ≈0.91 GPU-h。均须 C0 批准并在 REGISTRY 记录依据。

预算增加条件：`EXPERIMENTS.jsonl` 记录 + 路线 `STATE.md` 写明下一步信息增益 + C0 在 REGISTRY 备注批准。不满足则 HOLD。

## 5. 时间窗口

- 首波计划：2026-10-14 23:30 → 10-15 12:00（Asia/Shanghai）；夜间静默窗口优先正式计时。
- 连续两轮无新证据或到达预算 → 按协议决策收口（GO/HOLD/STOP/BLOCKED）。
