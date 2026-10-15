# C0 集成端到端 smoke — `train.regroup` 产品路径首跑（2026-10-15 02:27–02:30）

- 角色：C0（集成验收；补 `d1-003-c0` §3 / `d1-004-c0` §4 的"无端到端 train.regroup 训练运行"缺口）
- 环境：服务器 `oab-remote`；代码 = `/home/lyh/VelaDraft-c0`（origin `research/rsi-c0` @ `a39b80a`，含集成 `4e1674e`+`192554a`，PYTHONPATH 显式指向该树）；`c1vllm` env；**GPU0 单卡**（guard util≤5%/free≥40GiB ✓；无 co-resident 重负载）
- 协议：离线冻结 store 复用（D1 W3 probe store，192 行，只读）；`--features /home/lyh/rsi-d1/runs/d1_w3_probe/features`；同一 config（`/home/lyh/rsi-c0/configs/e2e_smoke_base.json` = D1 run config 副本）；20 步、seed 42、log_every=1、无 warmup 剔除；regroup 格仅加 `--set train.regroup.mode=grid --set train.regroup.sub_batch=2`（其余默认，fp32_weights=True）
- run：`C0-E2E-smoke-canon` = `/home/lyh/rsi-c0/runs/e2e_smoke_canon`（log `logs/e2e_canon_1015_0227.log`）；`C0-E2E-smoke-regroup` = `/home/lyh/rsi-c0/runs/e2e_smoke_regroup`（log `logs/e2e_regroup_1015_0228.log`）

## 结果

| 指标 | canonical | regroup (grid, sub2) |
|---|---|---|
| exit | 0 | 0 |
| steps / NaN / loss_decreased | 20 / false / true | 20 / false / true |
| final_loss | 7.2188（显示系 bf16 归约量化） | 7.2243（fp32 归约） |
| peak GPU mem | **25.34 GB** | **9.98 GB（−60.6%）** |
| 梯度覆盖 | — | **36/36 可训练张量**（逐组 backward 全覆盖，`gradient_coverage.json`） |
| ckpt | init/last/final 写出 ✓（验后已清，恢复=重跑） | 同 ✓ |

- **逐位 loss（20/20 步全量对比）**：|canon−regroup| ≤ 5.3e-3（最大 step18：7.4688 vs 7.4238，rel 6.1e-3；canonical 显示值被 bf16 归约量化为 1/64–1/32 网格）。差异落在 wave-2 已测"shipped 路径自身 bf16 波动带"（canonical-vs-ref ≤7.5e-3，`d1-002-c0`）内，方向符合预期（regroup 为更精确值）。
- 计门口径注意：本 smoke 的 wall（36.5s vs 62.5s 含分组形状编译）**不作性能主张**；性能以 wave-2 配对证据为准（+13–14% step / −57% peak）。
- 其他核对：`train.regroup enabled: mode=grid sub_batch=2 cost_weight=147.5 fp32_weights=True` 日志行 ✓；data stage digest 跨 run-dir 命中共用 ✓；provenance 记录外部 store 复用 ✓。

## 判定

- **集成产品路径 e2e：PASS**（功能性 + 数值级 + 内存级）。上述缺口关闭；残余未覆盖：online 模式（D4×D1 2×2 将覆盖）、100+ 步长轨迹、DDP（数学等价、多 allreduce 注记）。
- 性质：C0 自验 smoke（非门禁实验）；结论引用须带本文件路径与 run id。
