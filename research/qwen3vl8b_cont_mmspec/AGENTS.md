# Qwen3-VL-8B CONT 七方法 MMSpec 评测任务书

这份任务书对应 **Qwen3-VL-8B target + 七个 CONT drafter** 的 native-vLLM 端到端评测。七个 drafter 都是额外训练 2 个 epoch 后的 CONT 终点；不要换成原始 checkpoint，也不要把 smoke 数字写成正式结果。

## 1. 目标和统一结果位置

- Codebase：<https://github.com/LUOseu/VelaDraft>
- 入口：`vlm-dflash native-eval`
- 数据：MMSpec 完整 test，600 sessions / 723 turns
- 方法：`dflash`、`dspark`、`domino`、`jetspec`、`dflash2`、`dflare`、`dfly`
- 统一 HF 结果仓库：<https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-cont-dspark>
- 上传前缀：`eval/qwen3vl8b_mmspec600/<campaign_id>/`

七个方法的 raw report、evidence、命令、runtime metadata 和汇总 CSV 必须放在同一个 campaign 下。不要把结果上传到 checkpoint 根目录，也不要覆盖模型文件。不要上传 MMSpec 图片、target 权重或 drafter 权重；这些文件的路径、revision 和 SHA256 写入 `run_manifest.json`。

## 2. 当前使用的评测口径

正式速度沿用当前 A6000 上正在使用的 native-vLLM 路径：

```text
vlm_dflash/eval/native.py
vlm_dflash/eval/native_runtime.py
vlm_dflash/eval/native_resources.py
vlm_dflash/eval/native_bootstrap/
```

AR 使用原生 vLLM 默认高效路径，不挂 drafter；SD 使用方法自己的 native export、spatial adapter 和 runtime manifest。不要使用旧 `vlm-dflash eval`、prepared-input ratio、FlexAttention 或 F2/P2 作为主结果。

冻结设置：

```text
batch=1, greedy, temperature=0, do_sample=false
max_new_tokens=1024, natural EOS, repeats=3
gpu_memory_utilization=0.78（AR/SD 相同）
DFlash K=7；DSpark K=8；其余五种方法 K=7
```

计时包含 processor、图像解码、vision/prefill、完整 decode、draft、verification/rollback、同步和输出提取；模型加载、compile、warmup 另记，不进入请求 wall time。主指标为：

```text
speedup_sum_ar_over_sd = sum(AR e2e wall time) / sum(SD e2e wall time)
```

600 sessions 在三次重复下必须有 `723 * 3 = 2169` 个 paired turns。只有 `status=complete`、`resources_verified=true`、`timed_pairs=2169` 且没有资源污染的 formal report 才能进入正式汇总。

## 3. 当前服务器上的真实路径

默认路径就是当前 A6000 评测机使用的路径；换机器时通过环境变量覆盖：

```text
VLM_DFLASH_ROOT=/home/lyh/qwen3vl8b_eval_20261027
Q3VL8B_ROOT=/home/lyh/q3vl8b_cont_mmspec_20261103
TARGET=$VLM_DFLASH_ROOT/artifacts/target
SOURCE=$Q3VL8B_ROOT/mmspec_sharegpt_600.jsonl
IMAGES=/home/lyh/a6000_3b_v2_L300/shared/benchmarks/mmspec/images
PYTHON=$VLM_DFLASH_ROOT/runtime_frozen_v1/bin/python
```

七个 CONT export/manifest：

| 方法 | CONT native export | runtime manifest | K |
|---|---|---|---:|
| DFlash | `$Q3VL8B_ROOT/exports/dflash` | `$Q3VL8B_ROOT/dflash_cont_manifest.json` | 7 |
| DSpark | `$Q3VL8B_ROOT/exports/dspark` | `$Q3VL8B_ROOT/dspark_cont_manifest.json` | 8 |
| Domino | `$Q3VL8B_ROOT/domino_workspace/hf_cont_step125000` | `DOMINO_RUNTIME_MANIFEST` | 7 |
| JetSpec | `$Q3VL8B_ROOT/jetspec_workspace/jetspec_adapter/hf_export_step75000_v1` | `$Q3VL8B_ROOT/jetspec_cont_manifest.json` | 7 |
| DFlash2 | `$Q3VL8B_ROOT/dflash2_workspace/dflash2_8b_native_adapter/hf_step75000_v1` | `$Q3VL8B_ROOT/dflash2_cont_manifest.json` | 7 |
| DFlare | `$Q3VL8B_ROOT/dflare_workspace/dflare_8b_native_adapter/hf_step75000_v1` | `$Q3VL8B_ROOT/dflare_cont_manifest.json` | 7 |
| DFly | `$Q3VL8B_ROOT/dfly_workspace/dfly_native_adapter_v1/hf_step75000_v1` | `$Q3VL8B_ROOT/dfly_cont_manifest.json` | 7 |

当前目录中没有和其他方法同名的 `domino_cont_manifest.json`。脚本会明确阻止 Domino，不会偷偷套用通用 DFlash。要完成七方法全量，先提供经过 Domino adapter 审核的 manifest，并设置：

```bash
export DOMINO_RUNTIME_MANIFEST=/path/to/domino_cont_manifest.json
```

七个 CONT checkpoint 来源仓库分别是：

```text
khazic/veladraft-qwen3vl8b-200k-cont-dflash
khazic/veladraft-qwen3vl8b-200k-cont-dspark
khazic/veladraft-qwen3vl8b-200k-cont-domino
khazic/veladraft-qwen3vl8b-200k-cont-jetspec
khazic/veladraft-qwen3vl8b-200k-cont-dflash2
khazic/veladraft-qwen3vl8b-200k-cont-dflare
khazic/veladraft-qwen3vl8b-200k-cont-dfly
```

## 4. 一键启动

```bash
cd /path/to/VelaDraft/research/qwen3vl8b_cont_mmspec
chmod +x scripts/run_qwen3vl8b_cont_mmspec.sh
```

先做不占 GPU 的预检：

```bash
bash scripts/run_qwen3vl8b_cont_mmspec.sh --preflight --methods all
```

四张卡并行、每张卡上的方法串行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_qwen3vl8b_cont_mmspec.sh \
  --gpus 0,1,2,3 --methods all --suite all --publish
```

只有一张卡时七个方法自动串行：

```bash
bash scripts/run_qwen3vl8b_cont_mmspec.sh \
  --gpus 0 --methods all --suite all --publish
```

断点续跑：

```bash
bash scripts/run_qwen3vl8b_cont_mmspec.sh \
  --gpus 0,1,2,3 --methods all --suite all --resume --publish
```

默认本地输出为 `$HOME/qwen3vl8b_mmspec600/<campaign_id>/`，默认 HF 上传位置为：

```text
khazic/veladraft-qwen3vl8b-200k-cont-dspark:
eval/qwen3vl8b_mmspec600/<campaign_id>/
```

发布前确认：

```bash
hf auth whoami
```

## 5. 监控、验收和交付

```bash
tail -f "$HOME"/qwen3vl8b_mmspec600/*/logs/*.log
nvidia-smi
ps -fu "$USER" | grep -E 'native-eval|native_runtime' | grep -v grep
```

smoke 只证明入口、图像和 adapter 能启动，不能进正式主表。正式交付必须包含七方法的 `summary.csv`、每个方法的 formal report/evidence、`run_manifest.json` 和最终 HF commit。失败或缺少 manifest 的方法必须保留失败原因，不能用其他方法、旧 checkpoint 或 smoke 数字补齐。

如果没有 MMSpec gold/scorer，质量字段写 `N/A`；不要用 AR 输出伪造参考答案。
