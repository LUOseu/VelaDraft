# Qwen3-VL-8B 主实验：准备包（未完成）

完整执行说明和 A/B/C/D 实验要求见 [新版实验 AGENTS.md](VelaDraft_Qwen3VL8B_8xA100_Main_Experiments_AGENTS.md)。

本目录为 `56fde8c563222ec3196a053cafb6f6e213bd97ac` 的准备工作，**不是可提交四张主表的一键训练实现**。B 实验四臂配置仅为草稿，不得直接启动；正式方法固定为 DFlash2 和 DFly。已有源配置来自公开原始 8B 训练归档，见 `original_configs/PROVENANCE.json`；原配方的路径只供溯源，必须在正式节点重新绑定并核验。

## 生成 B 实验草稿（单 seed，1–2 个方法）

```bash
python3 research/q3vl8b_main_campaign/prepare_configs.py \
  --dflash2 research/q3vl8b_main_campaign/original_configs/dflash2.json \
  --dfly research/q3vl8b_main_campaign/original_configs/dfly.json \
  --out /path/to/new-campaign/configs \
  --seed 17
```

该命令拒绝 256-anchor 简化版、非 BF16 / 4096 / block 8 / batch 8 / 3 epochs 配方，保留数据集 seed 42，并为实验 B 使用一个训练 seed（默认 17）。固定生成已选的两个方法 DFlash2 和 DFly 的 ref/opt 四臂。输出标为 `DRAFT_NOT_LAUNCH_READY`。当前末次 75k 必须以最终 checkpoint 验证；25k/50k 为中间快照。

## 上传进度

Hugging Face 目标为公开 model 仓库 `khazic/veladraft-qwen36-400k-dflash`，仅追加/更新 `training_system_eval/qwen3vl8b_main/<campaign-name>/`。`progress.py` 从 `work/<run>/train/` 提取训练记录到 `raw/updates/`；同步仅扫描明确列举的目录、文本扩展名，拒绝链接和若干敏感内容，不上传模型、optimizer、图片或节点脚本。**规则不是安全证明**：公开仓库上传前必须由节点负责人配置 `VELA_SANITIZE` 脱敏规则，检查数据发布权限和样例导出。规则 JSON 的 `replace` 是 `[正则,替换]` 列表，`leak` 是不得上传内容的非空正则（例如包含内部路径和主机名的组合）；规则文件放在运行目录外。Hub token 通过该机器已有的 Hugging Face 安全登录提供，切勿写进代码、命令行或运行目录。

```bash
export VELA_SANITIZE=/secure/path/node_sanitize_rules.json
export MONITOR_PYTHON=/path/to/python-with-huggingface-hub
bash research/q3vl8b_main_campaign/monitor.sh /path/to/new-campaign
```

前台循环默认每 300 秒导出和上传，失败继续重试；日志打印错误类型，成功后将文件哈希和最后 commit 记在运行目录的 `.hf_sync_state.json`。监控进程不会启动训练；停止监控需由启动者终止该进程。`hf_sync_safe.py CAMPAIGN --once` 可用于单次检查上传，未配置脱敏规则或登录时会报错。状态文件不是训练完成凭证。

## 仍需完成的关键工作

1. 在 **8×A100 80GB** 的真实目标节点核对可用 GPU UUID、完整正式训练数据/图片、原有 dev/test、目标模型和全部成熟权重，冻结 hash 与版本；执行 200 更新和 20 请求的预算校准。
2. 实现具备配对初始化、真实 anchor 流核对、每阶段状态、job hash / 有完成凭证才能跳过以及精确断点恢复的调度器；禁止以这个草稿生成器替代。
3. 实现与实际数据绑定的 A/B/C/D 采集矩阵、视觉区域/轨迹对齐、完整核验、质量评分与原始数据重建四表。C 的条件性续训必须先经过独立开发集门控。
4. 在目标节点运行 CPU/GPU 单元测试、真实小规模端到端 sanity 和公平性核验；然后才能启动长程作业。原训练单臂接口为 `scripts/q3vl8b_trainopt/launch_run.sh`；旧 `launch_campaign.sh` 会复制节点环境文件，不应上传其 `scripts/node_env.sh`。
5. 交付 `run_manifest.json`、`jobs.json`、真实完成凭证、原始记录、四表、报告与重算命令。当前均未生成。
