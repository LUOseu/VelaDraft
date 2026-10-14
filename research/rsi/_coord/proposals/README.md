# Proposals（C0 接收箱）

D 角色对**共享入口**（训练入口/CLI/config/checkpoint-resume/公共 batch schema/公共测试）的最小改动提案放在本目录：`<role>-NNN.md`（例 `d1-001.md`）。

每份提案必须包含：
1. **目标与证据**：链接本角色 REPORT/EXPERIMENTS 中的支持证据；
2. **最小改动面**：文件:行、函数/参数签名、默认值 —— 默认**不得改变现有行为**（opt-in）；
3. **涉及角色与兼容性**：谁消费、是否需要 iface/协议版本 +1、对 digest/resume 的影响；
4. **测试要求**：新增/修改哪些测试，如何在 BASE 上验证（含现有 online suite 不回归）；
5. **回滚方式**。

C0 处理状态记录在本目录同名补充或 REGISTRY §7：RECEIVED / ACCEPTED（C0 集成）/ DEFERRED（附条件）/ REJECTED（附理由）。
版本化后由 C0 更新 INTEGRATION §3 并通知受影响角色。

历史：D1 的“anchor 表重放 hook”与“重排权重 API 对接点”（REPORT §6 / HANDOFF 2026-10-14）等待正式提案编号。
2026-10-14 23:30：**d5-001 已收**（producer/consumer 分片调度 runner）→ RECEIVED（部分/条件批准），答复见 `d5-001-c0.md`。
2026-10-14 23:56：**d4-001 已收**（adaptive byte-budget 旋钮 `train.online_feature_cache_adaptive`）→ 评审通过 **ACCEPTED（条件）**：D4 将冻结实现（内容 sha1 登记于 REGISTRY §7）提交 lane 分支后，C0 在 `research/rsi-c0` 集成（含 `docs/online-training.md` 文档行 + 测试复跑）；W2 租约 L-W2-D4 条件批准 ≤1.5 GPU-h（real-sensor capped-process A/B，平台工程定位）。
2026-10-15 00:04：d4-001 状态更新 → **已集成 @ `research/rsi-c0` `1f8963e`**（sha1 对账一致；本机复跑 13+4 ✓）；merge-readiness 待 L-W2-D4 real-sensor A/B 与 D4 lane commit 对账。
