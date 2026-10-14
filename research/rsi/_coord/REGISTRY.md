# RSI 协作任务板 REGISTRY

- 协议版本：**rsi-protocol-v0.1.0** ｜ 接口版本：**rsi-iface-v0.1.0**
- 共同基线 BASE_SHA：`7eab1369266010137702de8c714053729cde5834`（origin/main，PR #6 合并后；含固定预算 LRU 记录缓存）
- 历史核对基线 B0：`e15256d7c4bd10a490a7ac821eb12a0749a96fdd`
- 协调目录：`research/rsi/_coord/`（仅 C0 写）；本板由 C0 维护。更新时间：2026-10-14 23:20 (Asia/Shanghai)
- 无法确认的字段以 `?` 标注并在文末说明；未创建的 worktree/分支在下一次更新回填。

## 1. 角色与分支

| 角色 | branch | worktree | 唯一主责 | 可改范围 | 状态 |
|---|---|---|---|---|---|
| C0 | `research/rsi-c0` | `/Users/Admin/grounding-parallel/veladraft-c0` | 协议、资源、共享接口、集成、复核 | `research/rsi/_coord/` + 经批准的共享入口集成 | ACTIVE：基线已冻结，v0.1.0 已发布 |
| D1 | `research/rsi-d1` | `/Users/Admin/grounding-parallel/vlm-dflash-d1` | context/anchor 双轴组织、批内重排、ragged 布局 | `research/rsi/d1/` + 新模块/测试 | STARTED（worktree 已建 23:03，`research/rsi/` 未跟踪目录出现，内容未复核） |
| D2 | `research/rsi-d2` | `/Users/Admin/grounding-parallel/vlm-dflash-d2` | 重叠监督 teacher 去重、对应流式 loss | `research/rsi/d2/` + 新模块/测试 | STARTED（23:03，工作树 clean） |
| D3 | `research/rsi-d3` | `/Users/Admin/grounding-parallel/vlm-dflash-d3` | attention/mask profiling、元数据、必要的专用 kernel | `research/rsi/d3/` + 新模块/测试 | STARTED（23:04） |
| D4 | `research/rsi-d4` | 建议 `/Users/Admin/grounding-parallel/vlm-dflash-d4`（?未建） | 冻结 raw feature 保留、固定 LRU、adaptive budget、消费侧 I/O | `research/rsi/d4/` + 新模块/测试 | PENDING（worktree 未建；固定 LRU 已在 BASE） |
| D5 | `research/rsi-d5` | `/Users/Admin/grounding-parallel/vlm-dflash-d5` | 固定总预算下生产者-消费者与成本调度 | `research/rsi/d5/` + 新模块/测试 | STARTED（23:05） |

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

- 2026-10-14 23:20 — C0：冻结 BASE_SHA `7eab136`（科研前核查：六处 worktree 均 clean；review 四项问题核查结论见 INTEGRATION.md §1.1）。
