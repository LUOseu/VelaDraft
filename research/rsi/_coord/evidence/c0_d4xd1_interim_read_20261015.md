# C0 中期实读 — D4×D1 2×2（W3 cell；门冻结之下）

- 时间：2026-10-15 03:4x–03:5x CST（MIX cell 在跑）。来源：服务器 run tree 直读（未依赖 D1 harvest）。
- 上游：`d4xd1-001-c0`（预注册/门冻结）＋ D1 addendum（执行口径）；裁定 `d4xd1-002-c0`（update 轴读法）。

## 0. TL;DR

- W3 四腿 4/4 rc=0。**floor probe**（cache_on×bucket_off vs ref）：grads/updated/loss **全 bitwise 0.0**（缓存机制数值惰性 + 跨进程确定性成立）。
- 两条 bucket_on 腿（cache off/on）之间 **parity 数值与 loss 序列逐位一致** → 全部偏差 = 布局（regroup）足迹，与缓存无关。
- bucket_on vs ref：loss_rel 中位 **1.85e-3**、p90 4.29e-3、max **1.248e-2 @step 11**（steps 11–14 共 4 步 > 冻结门 7.5e-3）；有界、无 NaN、anchor 逐位一致、identity fp64 pass。
- **F1-as-registered = FAIL（bucket_on 两腿）**；bucket_off 两腿 PASS（0.0）。**不做事后放宽。**
- 初步归因：对比先天包含「regroup fp32 权重损失路径 vs 出厂 bf16」的已知差类（`train.py:340` `fp32_weights` 默认 True，注释 ~0.4% bias；wave-2 同 W3 行 canonical-vs-ref ≤5.96–6.60e-3）＋ 动力学放大；交互项（cache）为 0（bitwise）。
- 待办：MIX 收数 → C0 全量重算 → `d4xd1-003-c0` 收口（含“另立项”隔离对照提案）。

## 1. 事实（可复查路径）

| 项 | 结果 | 证据 |
|---|---|---|
| floor probe parity | grads {b0,b50,b100}=0.0、updated 同 | `runs/d4xd1_w3_cacheon_bucketoff/d1_capture/parity_vs_ref.json` |
| floor probe loss | 100/100 步 rel=0.0 | 两腿 `train/train_log.jsonl` |
| bucket_on parity（两腿） | grads 5.108e-3 / 0.8771 / 0.4872；updated 2.241e-4 / 5.754e-3 / 8.877e-3（两腿逐值相同） | 两腿 `d1_capture/parity_vs_ref.json` |
| bucket_on loss_rel（vs ref，2..101） | 中位 1.85e-3；p90 4.29e-3；max 1.248e-2@11；>3e-3：24 步；>7.5e-3：4 步（11–14） | 两腿 `train_log.jsonl` |
| anchors | 104 calls，0 mismatch（cell 内） | `d1_capture/anchors.jsonl` |
| identity（fp64） | pass：3.24e-8 / 5.69e-8（tol 1e-6） | `identity_grads000.json` |
| ref 张量范数 | grads full-vector：b0 89.05 / b50 0.2015 / b100 0.2239；b50 worst tensor(32) 范数 5.45e-4；b100 worst tensor(27) 范数 1.32e-3 | `/tmp/d4xd1_tensors/d4xd1_w3_cacheoff_bucketoff`（ref 保留） |
| 门定义 | F1=timed rows 2..101 max loss_rel ≤7.5e-3；grad_rel ≤6.0e-3（逐张量 worst L2-rel，addendum §4 声明） | `d4xd1-001-c0` §3 + addendum §4 |

## 2. 归因与读法

- `train.py:340`：`fp32_weights = bool(regroup_cfg.get("fp32_weights", True))`，注释：“bf16 quantisation of the preserved per-sample weights biases the loss by ~0.4%…”。regroup 腿因此使用 fp32 权重损失路径，而 ref 为出厂 bf16 路径——该差为**已知、文档化且 wave-2 已验证接受**的类别。
- wave-2 同 W3 行（离线、受控）：`EXPERIMENTS` rows 32–34 —— “canonical-vs-ref loss ≤5.96 / 6.53 / 6.60e-3（baseline bf16 loss_weight quantisation scale）”；“regroup-vs-ref ≤4.89e-5（对照臂=fp32 损失+canonical 分组）”。即：**损失路径差 ≈ 6e-3 级；纯分组足迹 ≤ 5e-5 级**。
- 本 2×2 的“bucket_on vs ref”= 损失路径差 + 分组 + 动力学；观测 max 1.248e-2 ≈ 已知差类（≤6.6e-3）的 ~1.9×（尾 + 训练动力学放大；online/offline 差异未定，隔离对照可界定）。**与 cache 及交互无关**（floor 0.0、两 bucket_on 腿逐位同）。
- grads 逐张量 worst 在 b50/b100 的 0.877/0.487 由小范数张量主导（绝对差 ~5–6.5e-4，≤0.3% 全向量范数）——该门读法需在报告中注语义；b0 5.108e-3@t23（范数 0.9487）为全向量范数可忽略级。

## 3. 不动作 / 边界

- **不放松 F1**；F1-as-registered FAIL 计入记录。
- 按 `d4xd1-001-c0` 判决映射：F1 败 → “组合（cache×regroup）数值相容性声明按原注册不成立；‘另立项’”。C0 将随收口给出**事前注册**的隔离对照提案（候选：W3(±MIX) `train.regroup.fp32_weights=false` 臂 vs ref，同门 7.5e-3；或 wave-2 式 fp32-ref 对照臂），不动原门。
- MIX cell 收数后追加（同读法：floor / bucket_on 双腿一致性 / F1 分布）。

## 4. 待办

1. MIX 4 腿收数（03:53:24 ref rc=0，其余在跑）→ C0 `d4xd1_recompute.py` 全量 8 腿重算。
2. D1 harvest 照跑（如实 F1 FAIL；IDENTITY 门照旧）。
3. `d4xd1-003-c0` 收口：F1–F5 + anchors + 三批位 parity + MIX 复现 + “另立项”提案。


