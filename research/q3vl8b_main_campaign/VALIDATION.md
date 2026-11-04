# 验证记录（2026-11-04）

准备节点：`bks@10.201.0.237:10010`，4×RTX 3090 24 GB，准备目录 `/data1/bks/veladraft_q3vl8b_main_preparation_20261104`；不是 8×A100 节点。仓库基线通过 `git archive` 导出自 `56fde8c563222ec3196a053cafb6f6e213bd97ac`；准备目录中的代码不是 Git worktree。

- `/home/bks/lyh/VelaDraft/.venv/bin/python` 可加载 `python -m vlm_dflash train --help` 和 `native-eval --help`。
- CPU：新工具测试、现有 trainer smoke、调度器和 training-plan 相关测试共 30 项通过；DFlash2、DFly、优化开关、projection/compaction 相关既有测试 78 项通过。pytest 单独安装在准备目录下的 `testdeps`，不更改原 venv。
- GPU：GPU 3（UUID `GPU-c2315efa-1eda-22b3-3fea-3afe285fa62a`）上小型合成 DFlash 的 BF16 forward/backward + AdamW 更新通过，非完整 8B，也非 DFlash2 或 DFly 实际多模态训练。
- 原始 DFlash2 / DFly 8B 配方生成 4 个明确标记为草稿的配置（两个方法 × ref/opt × 一个 seed），位于准备目录 `draft-configs/`；原始配方文件和 revision/hash 在 `original_configs/`。
- Hugging Face 目标仓库可公开读取；本机和准备节点没有检测到 Hub token。未向仓库上传；同步器通过本地模拟 commit、增量哈希及敏感内容过滤测试，真实授权上传、长期重试和节点脱敏规则尚未验证。

未执行：实际 200K 训练数据和 dev/test 血缘核验、真实的配对初态与 anchor 流一致性、真实 8B 图文更新、自动调度与断点恢复、A/B/C/D 矩阵、表格重算、A100 测量。不可由上述 smoke 结果推断这些工作已完成。
