# C0 终裁收据：d1-006（L-W3-D1Q）质量轴 — FAIL — 2026-10-15 05:47

四腿 = `d1q_canon_s42/s43`（Stage-1 建带）+ `d1q_regroup_s42/s43`（Stage-2 配对）；全部 200 步、loss_decreased=true、nan=false、**token_exact=True**；eval 相同 val64（64 prompts / max_new_tokens=128 / greedy / bf16 / sdpa，顺序 parity 验证）；本地镜像 `/tmp/c0_d1q/` 与服务器 **sha256 逐项一致**（16 项，下表）。

## 原始文件 sha256（本地=服务器）

| leg | eval/eval_result.json | train/train_summary.json | train/train_log.jsonl |
|---|---|---|---|
| canon_s42 | `c7ea30a0e96a4fa5669aaa1a5a71f98286f90f3023684f5b29b75608b068b089` | `355710bc849b3a24b6a5850ed4c3761329576b8f5cdd8e7c1ab5fe2b10a499d0` | `ef2fa5c5ad7db5e1dae9847b37531b50d779325ce25a22e9e43823aca4a171f8` |
| canon_s43 | `9071bf5741a150777f039e312cff746c372c2c7c6651b14aabfa46eb8961e9db` | `739c3b61c61a075587984496433ed02a8cf90d5dc0cb2719cf7f08f7e4692435` | `67923da9fe6c4bce537ec3a0b1e07fc61faed4f21512b318146d7976917e48ca` |
| regroup_s42 | `28f52c9a4f74f85a2ee0b296a304ffe5da654d90162d9988f4fbb3a6e004a459` | `b1f2842e90b34bbd06ed15d6d8a0e8d5da04293cb3a989a09102419bc869c4c8` | `ece84ff22cf519ecc93c2d1b597be993b077555a2ed7986e86eb7f2fa90ed565` |
| regroup_s43 | `f179797592db1625d9deb67f1913ee706bc66de414cc533d99e055cbe5e790a6` | `f7fbaf13dd810130fcbb5fe26284930a7824ff394ffe3294169828db19da9601` | `7a34b920fa9f5ee316c9acb4549dcc7b8f80a04d4ef618728d86d316abbb7931` |

## 数字（ckpt_final）

- c42: AL=1.1653666, acc=0.0239032 ｜ r42: AL=1.1621033, acc=0.0234309
- c43: AL=1.1635514, acc=0.0236240 ｜ r43: AL=1.1547380, acc=0.0223315
- band_AL = |c42−c43| = **0.0018152**；band_acc = **0.00027917**（canonical 42/43 ckpt_final 差值，冻结规则）

## 冻结判决（addendum §3；C0 未事后调整）

- s42: |ΔAL| = 3.2633e-3 > 0.0018152（**1.80×**）；|Δacc| = 4.7229e-4 > 0.00027917（**1.69×**）→ fail
- s43: |ΔAL| = 8.8134e-3 > 0.0018152（**4.86×**）；|Δacc| = 1.2925e-3 > 0.00027917（**4.63×**）→ fail
- **verdict = FAIL**（两 seed 同向、双指标同向；机器报告 `evidence/c0_d1q_band_report_20261015.json`，工具 `tools/d1q_band_recompute.py`）

## 含义

- 预注册映射：**regroup 训练质量有损 → 长训练 GO 的阻断项成立**。
- 与 `d4xd1-003/005` 同向（在线 cache×regroup 数值阻断）；机制不同（训练后 held-out 质量差异 vs loss-path 超带）。
- 工程口径保留：wave-1/2 测速（13.5%/13.3%）、peak（−57%）、容量（b24 OOM→14.41 GiB 可跑 5/5）仍为平台工程事实；「质量中性/免费」声明不成立。
- 观察（非判决口径）：s43 退化幅度大于 s42；绝对量级 AL 相对 −0.28%/−0.76%。

## 执行注记

- 执行链经事故 #8（守卫自碰撞）后由 D1 `d1q_run_leg.sh` 逐腿恢复（canon 43 → regroup 42 → regroup 43，05:10/05:27 起，末腿 done @05:44:52）；产物有效性不受事故影响（token_exact 全真、双端哈希一致）。
- L-W3-D1Q 预算（估）：四腿 train+eval ≈4×(3.6+13) min ≈ **1.1 GPU-h**（≤2.4；待 D1 harvest 对账）。
- 待办：D1 harvest 回填（归档四腿至 lane `evidence/d1q_runs/` + REPORT/EXPERIMENTS/STATE）→ C0 最终对账；D5 `staged_b32` 仍未启动。
