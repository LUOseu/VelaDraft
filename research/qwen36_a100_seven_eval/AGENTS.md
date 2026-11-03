# 执行手册：8×A100，Qwen3.6 七方法，MMSpec 全量

## 已确认的任务，不要重新询问

- 机器是**一个节点、8 张 A100 80GB**。每张卡独立运行一个 **TP1、BF16、batch1** 的任务，不使用 TP8。
- Target 固定为 `Qwen/Qwen3.6-35B-A3B`，权重 revision 见配置文件。A800 80GB 上已经实测 TP1 可运行，优化后端的小规模速度约 1.3×；直接沿用这套适配作为起点，不再做“能否 TP1”的研究。
- 七方法：`dflash,dspark,domino,jetspec,dflash2,dflare,dfly`，全部 **K=3**。不扫其他 K、不重训、不测 MTP 或 ViSpec 方法。
- 唯一 benchmark 是 MMSpec 官方完整 `test`：**600 个 session、723 个 turn、六类各 100 个 session**。不跑之前 ViSpec 论文的数据集，不额外加 caption benchmark。MMSpec 本身包含 caption。
- 优先把正式速度评测跑完。逐 token 一致率是诊断项，不作为阻塞条件；不要把未评分的质量项当作速度失败。

## 代码已经有什么

仓库：<https://github.com/LUOseu/VelaDraft>。PR #39 提供七方法及优化后端；PR #41 提供真实存在的入口、数据准备、CPU 导出、八卡调度、断点恢复、汇总和 HF 发布器。

| 用途 | 文件 |
| --- | --- |
| 一键入口 | `scripts/run_qwen36_a100_benchmarks.sh` |
| 默认配置 | `configs/eval/qwen36_a100_seven.yaml` |
| 控制器、任务队列、实际推理 | 本目录 `cli.py`、`lane.py`、`stage.py` |
| 数据、权重准备 | 本目录 `data.py`、`prepare.py` |
| 汇总、持续上传 | 本目录 `summarize.py`、`publish.py` |
| 已测后端配置与 checkpoint pins | `research/qwen36_seven_20261102/INFERENCE_SETTINGS.json`、`checkpoint_manifest.json` |

**实现约束：**七方法通过已经验证的 DFlash proposer 接入各自模型。DSpark 使用现有 bridge 调用其 Markov 头，不能仅因方法名是 DSpark 就改用独立的 `method=dspark` proposer，否则会偏离已验证的 K+1 query/位置路径。DFlash 的 `JointWorker` 只提供 `backend_metadata`，不要求它实现其他方法的 `method_metadata`。

默认显存分配比例与 A800 实测一致为 `0.92`；保留 V2 runner、CUDA GDN、FlashAttention、CUDA Graph 和 SD 的 query-width MoE grouping。A100 tile 使用 vLLM 对应硬件的默认配置，已有实测 A100 表时才设置 `Q36_MOE_TUNED_DIR`。不要把 A800 表改名后宣称是 A100 已验证配置。此处不是承诺 A100 全量一定复现 1.3×。

## AR 和测量口径

主配对沿用 A800 口径：原生 vLLM V2 / CUDA GDN / FlashAttention **AR 不挂 drafter**，与 SD 在同一张物理 GPU 上交替测三轮。另测一个 stock-default AR，单独列出原始时间；当前代码不把这个独立任务当成与每个 SD 同卡的配对结果，也不宣称已找到全局最快 AR。不得刻意放慢 AR。

两种协议独立报告：

1. `natural`：greedy、自然停止、最多 1024 token。
2. `fixed512`：真实生成 512 token、`ignore_eos`，不拿 EOS 后续写计算自然回答质量。

每种协议三轮；主表为 `ΣAR 请求墙钟 / ΣSD 请求墙钟`，另外列每 token 耗时比、95% 配对 session bootstrap CI、输出长度及一致率。图像读取、处理、prefill、decode 和 detokenization 均计入请求时间；模型加载、编译、预热排除。多轮使用冻结的 AR 历史，输入 token 必须相同；不是各方法自由发展的对话质量测量。原生接受长度来自单独的 natural stats pass，不使用训练面板 AL。

## 在服务器上准备一次

以下操作均在 A100 节点执行。模型和数据不下载到用户 Mac。使用干净 checkout，先拉取包含本修复的公开分支；合并后可改用 main 并固定实际 commit。

```bash
git clone --branch codex/mmspec-a800-parity-fixes https://github.com/LUOseu/VelaDraft.git
cd VelaDraft

# 复用节点已有 vLLM 0.32.0 环境；如未准备，由执行 agent 在服务器创建独立环境。
# 不要更改宿主机驱动或覆盖他人的 Python 环境。
export Q36_PYTHON="$(command -v python)"
export CUDA_HOME=/usr/local/cuda-13.0

# /data/veladraft 可替换为节点上有写权限和足够空间的共享数据盘。
export Q36_STORAGE=/data/veladraft
export Q36_A100_ROOT="$Q36_STORAGE/qwen36_a100_seven"
export Q36_A100_EXPORTS="$Q36_STORAGE/exports"
export Q36_TARGET_PATH="$Q36_STORAGE/models/Qwen3.6-35B-A3B"
export Q36_MMSPEC_CLONE="$Q36_STORAGE/MMSpec"
export HF_HOME="$Q36_STORAGE/hf_cache"
export TMPDIR="$Q36_STORAGE/tmp"
export TRITON_CACHE_DIR="$Q36_STORAGE/triton_cache"
export VELADRAFT_GPU_LOCK_DIR="$Q36_STORAGE/gpu_locks"
mkdir -p "$Q36_A100_ROOT" "$TMPDIR" "$TRITON_CACHE_DIR" "$VELADRAFT_GPU_LOCK_DIR"

# 使用现有 HF 凭据；没有时交互登录，不在命令或日志里粘贴 token。
hf auth login
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir "$Q36_TARGET_PATH"
```

环境要求：vLLM **0.32.0** 及其兼容的 PyTorch/Transformers，PyYAML、huggingface_hub，完整 CUDA 13.0 toolkit。优先复用 A800 已验证环境；现场版本记录进入 manifest。完整 target 已在正确路径时不用重复下载。七个最终 checkpoint 会按固定 revision/hash 自动下载、校验并 CPU 导出；MMSpec 自动拉取固定 commit。

## 启动：短检查成功后自动进入全量

在同一个 shell、同一个环境运行下面整段 Bash。两步以退出码连接；短检查失败不会浪费整套正式评测。不要停在冒烟结果上当成正式表。

```bash
set -euo pipefail
BENCH_ARGS=(
  --config configs/eval/qwen36_a100_seven.yaml
  --gpus 0,1,2,3,4,5,6,7
  --methods dflash,dspark,domino,jetspec,dflash2,dflare,dfly
  --draft-length 3
  --publish
)
bash scripts/run_qwen36_a100_benchmarks.sh "${BENCH_ARGS[@]}" --preflight
bash scripts/run_qwen36_a100_benchmarks.sh "${BENCH_ARGS[@]}" --suite mmspec-smoke
bash scripts/run_qwen36_a100_benchmarks.sh "${BENCH_ARGS[@]}" --suite mmspec-full
```

需要断开 SSH 时，把以上已配置环境的命令放在 `tmux`/`screen` 中运行。初始 reference 只用一张卡生成，其余 lane 等待，这是现有队列行为；随后八卡独立领取 method/protocol 工作。模型加载期间 GPU util 低不等于退出，查看日志和状态。

## 查看进度与恢复

```bash
# 正式运行建立后：
cat "$Q36_A100_ROOT/latest_mmspec-full/status.json"
tail -f "$Q36_A100_ROOT/latest_mmspec-full/logs/lane_gpu0.log"
cat "$Q36_A100_ROOT/latest_mmspec-full/summary.csv"

# 中断后，保持同一代码 commit、环境、八张物理 GPU 和配置，继续已建立的正式运行：
bash scripts/run_qwen36_a100_benchmarks.sh \
  --config configs/eval/qwen36_a100_seven.yaml \
  --suite mmspec-full --gpus 0,1,2,3,4,5,6,7 \
  --draft-length 3 --publish --resume
```

首次启动不要加 `--resume`；它只用于已经存在的运行。需要恢复指定历史运行时，加 `--run-id <实际目录名>`。完整 topic shards 保留，缺失项重跑；不要编辑 manifest、旧原始数据或更改已有运行的物理 GPU。错误原因查看各 lane 日志和 `stages/`；其他可运行任务继续推进，不因一个方法失败全部停工。

## HF 与质量

所有七方法结果持续发布到：

`khazic/veladraft-qwen36-400k-dflash` 的 `main`，目录 `eval/qwen36_a100_seven_v1/runs/<run_id>/`。

使用具有目标仓库写权限的 HF 凭据；凭据缺失是唯一需要负责人提供的访问材料，不再询问 benchmark、K、卡数或 TP。发布器约每 300 秒更新状态和新增结果，保留其他运行并核验上传内容。七个 checkpoint 的 revision 在开始时已冻结，后续上传不会让模型跟随变化的 `main`。网络失败保留本地结果并重试，不覆盖旧原始 shard。

MMSpec 官方 JSONL 没有参考答案，也没有直接可用的任务评分器；当前 runner 保存所有自然输出并把质量标为 `pending`。先完成速度和接受长度；拿到明确的参考答案/评分器后再评分。输出一致率不能冒充任务分数，不能把质量 pending 写成质量已通过。

## 交付

执行完成后提供 `summary.csv`、`summary.json`、HF 结果链接、实际代码 commit、完成/失败方法清单。速度有效性、任务质量和输出一致率分别报告。当前代码经过 CPU 测试，不把它说成新的 A100 GPU 实测；A100 真正运行结束后再报告对应加速比。不要因追求固定 1.3× 而更换样本或降低 AR 性能。
