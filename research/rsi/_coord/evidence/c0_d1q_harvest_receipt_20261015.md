# C0 收讫：D1 d1-006 harvest 对账（L-W3-D1Q）— 2026-10-15 05:52

- harvest commits（lane `research/rsi-d1`）：`3bc72ba`（Stage-1 建带）、`12d3b5e`（s42 中间 FAIL as-registered）、`1b0328e`（完成，NEGATIVE）。
- 交付物核对：REPORT §14.1–14.3 存在；EXPERIMENTS +9 行（共 64 行）；`evidence/d1q_runs/<leg>/{train,eval}/`、`d1q_band_summary.json`、`d1q_lossdiff.json`、`d1q_timeline.json`、构建器 ×4。

## 对账结果（C0 独立重算 vs D1 汇总）

- band：AL `0.0018152127954278452` / acc `0.000279171630289541` —— **逐位一致**。
- 配对：s42 dAL `3.263316593647092e-3` / dacc `4.722927003200615e-4` → fail；s43 dAL `8.813420728333465e-3` / dacc `1.292476304990426e-3` → fail —— **逐位一致**；D1 记「NEGATIVE (regroup quality out of band)」≡ C0「FAIL」（同一冻结规则）。
- **raw 完整性重验**：服务器 4×`eval/eval_result.json` + 4×`train/train_summary.json` 的 sha256 **8/8 与 C0 /tmp/c0_d1q 副本一致**——harvest 未改动 raw。
- token_exact：4 腿 × 2 checkpoint = 8/8 true（双方一致）。

## 补充证据（表述性，不进判决）

- `d1q_lossdiff`：s42 max loss_rel 1.026e-2@195（36/200 步 >1 ulp；中位 0.462）；s43 max 3.207e-2@10（99/200 >1 ulp；中位 0.971）。披露：regroup（fp32_weights=true）记 fp32 值 vs canon auto bf16-grid——与 `d4xd1-004` 同类 logging-grid 不对称，仅作机制表述。
- `d1q_timeline`：四腿 guard→exit 全记账（含事故 #8 后恢复链）。

## 预算与披露

- 预算：**1.1536 / 2.4 GPU-h**（trains 0.0675/0.0692/0.0669/0.0694 + evals 0.2178/0.2158/0.2164/0.2175 + smoke 0.0131）——与 D1 报告一致；**L-W3-D1Q 关闭**。
- 披露（无整改）：ckpt_init/ckpt_last 已清（指标保留于 eval_result）；ckpt_final ×4 保留（487MB/腿）供可选重评；config.json 无 --set 回显（以日志 echo 为审计点）；2 seeds / N=200 / 低接受率（2.2–2.4%，slot-1 主导）为已知边界。

## C0 映射（采纳）

- d1-006 = **离线冻结特征布局轴**负向 → 与在线 `cache×regroup` 数值阻断（003/005）同向；**长训练 GO 需机制/数值修复 + 新预注册**，加 seed（N=200）不改变结论。
- 工程口径不受影响（测速/容量/内存结果保留）。
