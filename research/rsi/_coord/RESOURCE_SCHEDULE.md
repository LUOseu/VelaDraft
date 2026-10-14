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

## 3. 租约纪律

- 同一 GPU 同时只允许一个 VelaDraft 重负载任务；开跑前 `nvidia-smi` 实测该卡 0% util 且 >40GiB free；记录卡号与外部占用比例。
- 正式计时：选最安静的卡；同一实验两腿尽量同卡；不同卡或存在外部占用时必须在报告中披露。
- 不得 kill 他人进程、改全局环境、写他人输出目录；**空闲显存不是占用许可**。
- 磁盘守卫：启动前 >10GiB free；单任务净新增 ≤2–3GiB；harvest 后 prune checkpoint；<4GiB free 时停止提交 GPU 任务，改做 CPU 分析。
- 每个 run 在 `EXPERIMENTS.jsonl` 记录 device/contention/disk 状态；等待时间计入 GPU-hours。

## 4. 预算（wave 1 硬上限；增加须证据）

| 轮次 | 上限 | 依据 |
|---|---|---|
| W1-D1 两轴 probe | ≤1.5 GPU-h（23:32 由 1.0 上调：+W1/W2/W3 真实 probe stores；见 REGISTRY §7 对 D1 的应答） | R2 tensor probes + 三个 128–256 行 probe store；旧 lane 参考：129 行 × 6 epochs 两腿 ≈0.24 GPU-h |
| W1-D2 N/U + teacher 成本 | ≤1.0 GPU-h | 计数与成本占比，128–256 样本 |
| W1-D3 attention/mask profile | ≤1.0 GPU-h（23:32 由 0.5 上调：需 warm/编译基线与 span 网格；依据 D3 round-1 计划 ≤4 短 run） | 仅 profiling；未证明瓶颈（attention+mask ≥20% 整步）不写 kernel |
| W1-D4 收尾/容量探针 | ≤0.5 GPU-h | 容量/命中率为主，优先 CPU/trace |
| W1-D5 静态流水线/trace 可行性 | ≤1.0 GPU-h（单卡；23:12 由 0.5 上调，依据 D5 handoff 的 regen scaling probe） | 静态 trace + 单卡模拟；不建多 GPU 系统 |
| **wave 1 合计** | **≤4.0 GPU-h** | 当前 2 张空闲卡 + 紧张磁盘下的可执行上界 |

### 已发布租约

| 租约 | 对象 | 卡/条件 | 有效期 | 备注 |
|---|---|---|---|---|
| L-W1-D5 | D5 | 任一实测 quiet 卡（开跑前实测 util≈0%、>40GiB free；同一时刻只允许 D5 一个任务） | 2026-10-14 23:12 → wave 1 结束或他人占用出现 | 单卡 provisional ≤1.0 GPU-h；23:11 实际使用 GPU1（当时 4.8GB/2%，满足 quiet 条件） |
| L-W1-open | D1/D2/D3/D4 | 任一实测空闲卡（D1≥D2≥D3/D4 优先顺序由任务到达决定；同卡不并发重负载） | 同上 | 各角色 wave 1 上限见上表 |
| L-W2-G2 | 通过门槛者 | 2×A6000 同时空闲 ≥3–4h 窗口 | **UNCONFIRMED**（GPU1/3 被占） | wave 2 由 C0 复核证据后发布 |

### 观察记录（C0 只读，未独立复核）

- 2026-10-14 23:19 — 四卡全部在用：**GPU1** = D5 regen 批量 probe（`/home/lyh/vlm-d5/repo`，--limit 128 --steps 1，regen.batch_size=8，pid 3311115）；**GPU2** = D3 attention/mask profiler（profile_drafter_step.py，limit 16/10 steps，pid 3317651）；**GPU0** = 外部用户 xs 的 Qwen2.5-Math eval（pid 3298367）；**GPU3** = 外部 dongzp（pid 3094389，44.3GB/100%）。无空闲卡 → D1/D2/D4 的 GPU probe 需等窗口。`/home` 29GB free。
- 2026-10-14 23:32 — D3 profiler 已退出（GPU2 释放，18MiB/0%）；GPU0 外部任务结束（287MiB/0%，近空闲）；D5 probe 仍在 GPU1（11.8GB/2%）；GPU3 外部继续（44.3GB/100%）。`/home` 27GB free（D5/D3 小量写入）。D1/D2/D4 可用 GPU0/GPU2 窗口。
- 2026-10-14 23:36 — GPU1 = D5（10.4GB）+ **D2（27GB，并发，未登记租约）**；GPU0 = 外部 xs（14.9GB）+ D2（264MiB）；GPU2 空闲；GPU3 外部。重负载定义：alloc > 4GB 或 util > 50%；同卡禁止两个未声明的重负载任务。受影响的计时标 `contended`。

Wave 2（仅通过证据门槛的 1–2 项）：配对 100–200 更新实验 ≤4 GPU-h/方向；3 seeds canonical 合计 ≤6 GPU-h；参考量级：2K 切片两腿 ≈0.91 GPU-h。均须 C0 批准并在 REGISTRY 记录依据。

预算增加条件：`EXPERIMENTS.jsonl` 记录 + 路线 `STATE.md` 写明下一步信息增益 + C0 在 REGISTRY 备注批准。不满足则 HOLD。

## 5. 时间窗口

- 首波计划：2026-10-14 23:30 → 10-15 12:00（Asia/Shanghai）；夜间静默窗口优先正式计时。
- 连续两轮无新证据或到达预算 → 按协议决策收口（GO/HOLD/STOP/BLOCKED）。
