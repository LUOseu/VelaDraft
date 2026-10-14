# 转发文本（2026-10-15 04:44 CST，C0）— d1-006 恢复 + D5 开工提醒

> 用法：以下两节可直接转发给对应角色（C0 与 D 角色无直连，由用户转发）。

## 致 D1（d1-006 / L-W3-D1Q 恢复）

- 事实：s42 stage+train 已成功（exit=0 @04:37:25；200 步，final_loss 5.75）；s42 eval 已于 **04:51:41 完成**（token_exact=True；final AL=1.1654 / acc=0.0239；eval_result.json 已落盘；C0 已取 raw，双端 sha256 一致）。此前 s42 eval 发生**双副本自碰撞**：编排器副本 A 因 `/home` 11G（<15G 门）重试 4 次后被第二副本 B 的 GPU 占用挡下（`GUARD FAIL (tenant)` 为误报——“tenant”=B 自己）；B 为孤儿进程，完成了本次 eval（exit=0 @04:51:44）。
- 编排器 `d1q_stage1.sh` 已不在 ps、status log 无 done/FAIL → **链死**。**更新（05:09:32）：canon 43 已完成且干净退出；剩余 regroup 42 / regroup 43 两条腿请继续串行补跑（各 ~17-18 min，命令见下）。**
- 请在 B 的 eval 落盘（`runs/d1q_canon_s42/eval/eval_result.json` 出现）后：
  1. s42 四件套已齐且 C0 已取 raw（双端一致）；拷贝到 lane `evidence/d1q_runs/d1q_canon_s42/` 即可（勿清 s42 run-dir）。
  2. 逐腿串行补跑（单副本；建议同一 nohup shell；每腿开跑前 `df -h /home` ≥15G）：
     - `bash scripts/d1q_run_leg.sh canon 43`
     - `bash scripts/d1q_run_leg.sh regroup 42`
     - `bash scripts/d1q_run_leg.sh regroup 43`
     （脚本幂等：stage 缺则补、train_summary 在则跳、训练后自动清 ckpt_last、eval_result 在则跳；regroup 自动加 `train.regroup.mode=grid sub_batch=2`。）
  3. 各腿完成后把 `eval_result.json` / `train_summary.json` / `train_log.jsonl` 归档进 lane `evidence/d1q_runs/<leg>/`；C0 随后独立重算 band 并发布 d1-006 裁决。
- 纪律：一次只跑一条腿（避免再出自碰撞）；`transient spike` 标签实为磁盘门错误，恢复期间请直接盯 `/home`；若 <15G 先上报再清（勿碰他人目录）。

## 致 D5（staged_b32 / L-W3-D5B）

- `L-W3-D5B`（≤0.75 GPU-h）维持批准；截至 04:43 仍未见开跑。GPU0/1/2 空闲（GPU3 被 d1-006 eval 占用，勿抢），`/home` 19G。
- 请现在开工 staged_b32；跑毕归档对比 JSON + `research/rsi/d5/runner/split_verify.py` 输出（C0 复核抽查）。磁盘纪律：开跑前 ≥15G、跑后 prune；单副本。
