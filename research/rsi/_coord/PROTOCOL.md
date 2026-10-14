# RSI 实验协议 PROTOCOL v0.1.0

适用范围：D1–D5 全部 run；与 BASE_SHA `7eab136` 绑定。修改须升版本并在 REGISTRY 登记。

## 0. 原则

事实 → 可证伪假设 → 最小实验 → 检查证据 → 修正/实现 → 复核。每轮先写：假设、反例、关键指标、预期上限、预算、停止条件；一轮只改一个主要机制。允许证伪与转向；负结果、明确边界和可复现实验同样算完成。

## 1. 冻结清单（每个对照必须一致；同 seed 不能替代）

teacher/drafter revision 与初始化；数据与处理版本；样本访问计划（顺序 + 分组）；原始 anchor 表；随机噪声/掩码计划；逻辑 optimizer batch；各 loss 权重；LR 计划；dtype/精度路径。

- 物理重排不得改变 optimizer 更新顺序；先保持逻辑 batch 边界。
- 新 microbatch 不得直接重新平均而改变样本权重；若改全局加权目标，基线与候选同改，并单列这一协议变化。

## 2. 数值层次与容差

明确区分四层：数学目标等价 / 容差内数值一致 / 训练质量一致 / bit-exact。

- 用小例子高精度 oracle 对照：输出、分项 loss、相关输入梯度、全部 trainable 参数梯度、一次 update。teacher hidden 若 detach 不要求其梯度，但所有可训练融合/投影梯度必须正确。
- bf16 容差在候选结果出现前，用 baseline 自身数值波动设定；不得事后放宽。
- 每次 run 记录 `parity_pass` + 层级 + 证据路径。“打印 loss 相同”不等于 bit-exact。

## 3. 缓存与恢复语义（BASE 事实）

- `train.online_feature_cache_gb`：0=off；>0 为字节上限 LRU；运行时不落盘；**resume 一律冷启动**。
- hit = 复用同一数值（bitwise）；miss（冷启动/evict/超容量）按参考路径重算；partial hit 下 hit 行与重算行必须分开描述，不得合并宣称“整体 bit-exact”。
- 冷启动、partial hit、重新分组可能改变 bf16 teacher 数值（shuffling loader 每 epoch 分组不同，见 docs/online-training.md 与 REPORT §4.3–4.6）。
- 布局/缓存隔离实验先固定同一份特征；online 联合实验统一缓存模式或做 {off,on} 2×2 成对报告。禁止 warm 全量缓存 vs cold miss 只报机制收益。

## 4. Workloads

- W0：canonical 2K 切片（复用既有 recipe/asset）。
- W1：长视觉短回答——**必须单独冻结短回答协议**（现有 `min_response_tokens=128` 会滤掉 W1；不得只给优化版本换数据）。
- W2：短视觉长回答（高 anchor / 大词表投影负载）。
- W3：混合分辨率/多图/多轮、长度混合。

只做与假设相关的子集；先图像/多图，不把视频当已支持能力。长上下文纯张量 microbenchmark 与真实数据分开报告，不能用视觉特征重复伪造质量验证。

## 5. 规模与节奏

- 每方向先 128–256 样本 probe；过门槛才做 100–200 更新的配对实验。
- 稳定节点同机 A/B 交替 ≥3 次；首次加载/编译与 warm 稳态分开；正式计时关闭重 profiler；异步流水线用事件与端到端边界计时，不在每阶段强制 synchronize。
- 三 seeds 长训练只给 C0 批准晋级的 1–2 项；避免全模型 × 全算法 × 全形状笛卡尔积。
- 每个 run ≤2h wall（超出需在 REGISTRY 记录理由）；失败 run 同样入账。

## 6. 计量（必报）

train wall；需要时原始数据 → checkpoint pipeline wall（regen/capture 全计入）；分配 GPU-hours（含等待）；有效监督 / 有效 block 吞吐；GPU 与 host 内存峰值；逻辑读取 vs 物理磁盘读取；H2D/D2H 字节与耗时；多 GPU 时各卡忙闲、慢 rank、队列字节与等待。重叠阶段时间不能简单相加；各层收益不能相乘。

分阶段时间（按相关性取）：image decode/processor、vision encoder、target prefill/decode、capture、read/decode/collate、mask build、draft attention/MLP、head/loss、backward、optimizer、checkpoint。

## 7. 质量

held-out AL、per-slot acceptance、loss（同题配对统计）。最终 verifier 同答不能代替 drafter 质量验证；token-exact 只说明 verifier 路径正确。AL 门槛从独立 baseline/dev 冻结后再用，不能看优化版或测试集后挑。

## 8. 结果字段（EXPERIMENTS.jsonl 每行至少）

`run_id, role, git_sha, base_sha, protocol_version, deps_sha, hardware(device/contention), allocated_gpu_hours, command/config, seed, dataset_digest, anchor_digest, logical_batch_digest, cache_mode(cache_budget_bytes), execution_mode, effective_supervision_count, unique_teacher_positions, train_wall_s, pipeline_wall_s, step_p50_ms, step_p95_ms, gpu_peak_bytes, host_peak_bytes, logical_read_bytes, physical_read_bytes, h2d_bytes, d2h_bytes, context_padding_ratio, anchor_padding_ratio, loss_delta, gradient_relative_error, val_AL, final_AL, parity_pass, limitation, decision, raw_log_paths`。
不适用项填 null 并说明。

## 9. 决策格式

GO / HOLD / STOP / BLOCKED，附：证据（run_id + 原始路径）、相对强基线判断、预算消耗、下一轮条件与上限。连续两轮无新证据/收益或到达预算 → 必须给出决策；继续须新证据 + 更新预算。

方向专项停止条件：D3 未证明瓶颈不写完整 kernel；D5 缺静态强基线或真实多 GPU 不扩建大系统；D1/D2 优先研究资源；D4 提供实用 backend。

## 10. 禁止项

换指标保故事；偷改数据；调大 baseline 开销；只保留最快 run；把 microbenchmark 倍率当训练倍率；把常规缓存/packing/流水线包装成新算法；未核查就宣称 novelty；未获 C0 接收改共享入口或 cherry-pick 他人提交；动他人 worktree/进程/输出目录；用 sleep/mock 伪造加速。

## 11. 变更流程

协议变更：提案 → C0 起草修订 → REGISTRY 记录 → 版本 +1（minor=字段/规则；major=语义）→ 经用户转发通知 D1–D5。接口变更见 INTEGRATION.md §3。

## 12. iface v0.2 候选（草稿 · 2026-10-15 00:20）

D5 提交的跨层命名（REGISTRY §7 00:20 回合），C0 接受为候选。**未冻结**：最终冻结 = D1 `d1-001` 提交后与 layout contract v0 合并，由 C0 把各字段状态切为"已发布"（走 §11 流程）。所有字段**可选、默认行为不变**。

| 字段 | 状态 | 语义 / 备注 |
|---|---|---|
| `sample_id` | 候选（D5） | 数据集/manifest 行身份；取代 d5-001 §4 的 `row_key` |
| `epoch_index` | 候选（D5） | 0-based |
| `plan_position` | 候选（D5） | 该 epoch demand 序列内 0-based 位次；= D1 `logical_batch_index*micro_per_batch + micro_batch_index` 扁平化 |
| `regen_digest` | 候选（D5） | 行生成记录 digest；幂等键 = (`sample_id`,`epoch_index`,`plan_position`,`regen_digest`)；受 D5 REPORT §7 生成冻结协议约束 |
| `teacher_src_pos` | 已确认（D2 handoff §1；待 d1-001 合并冻结） | = D1 `slot_pos` − 1（`TeacherLogitsStream` 语义：slot k 读 position−1 hidden；D2 口径 = `label_indices−1` clamp，与 D1 contract v0 一致） |
| `host_ram_budget_gb` | **已确认（D4 00:46；默认 null；冻结走 §11）** | launcher 级 host RAM 总预算；D4 cache payload 与 D5 queue/staging 记入同一份额；product 侧只读；不进 `training_plan_digest`；接线需提案 |

补充（00:52，D1 `d1-001` 接收后）：v0.2 主体另含 D1 增量——可选 forward kwargs `anchor_table`/`sample_weights`、可选 config 块 `train.regroup.{mode,sub_batch,cost_weight,fp32_weights}`（默认全关、行为逐字节不变）、layout contract v0.2 字段组（D1 handoff "Layout contract v0.2"；C0 已阅）。冻结仍走 §11 流程。
