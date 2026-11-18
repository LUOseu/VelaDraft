# Research and evaluation

Use [native-eval](../docs/benchmark.md#native-vllm-e2e) for a paired vLLM benchmark with your own prepared export. The following runners reproduce specific seven-drafter MMSpec campaigns. Each runbook gives the checkpoint pins, hardware configuration and launch commands.

| Target | Entry point | Runtime |
| --- | --- | --- |
| Qwen3-VL-8B | [CONT campaign](qwen3vl8b_cont_mmspec/README.md) | Export the continuation checkpoints, build the spatial adapter manifests and evaluate with native-eval. |
| Qwen3.6-35B-A3B | [Seven-method campaign](qwen36_a100_seven_eval/README.md) · [launcher](../scripts/run_qwen36_a100_benchmarks.sh) | Run resumable, single-GPU vLLM jobs with a pinned method table. |

## Qwen3.6 runtime sources

The launcher, saved worker class names and validation receipts use these directory names.

| Component | Sources |
| --- | --- |
| Checkpoint export and seven-method adapters | [Native exports](qwen36_native_vllm/) · [method workers and checkpoint manifest](qwen36_seven_20261102/) |
| Spatial positions and target execution | [vLLM 0.32 worker](qwen36_v032/) · [grouped MoE dispatch](qwen36_grouped_e2e_20261102/dispatch.py) · [tuning audit worker](qwen36_nextopt_20261102/worker.py) |

The [A800 validation receipt](qwen36_seven_20261102/VALIDATION.json) records the [original exporter](https://github.com/LUOseu/VelaDraft/blob/6ae909a9d52cfd36d6f4c3fd591c0665d0ba8848/research/qwen36_seven_20261102/prepare_custom.py). The A100 runner uses a later revision of that file.

The [research archive](archive/README.md) contains the earlier 8B snapshot, training studies, R3/R5 reports and Qwen3.6 diagnostic sweeps. Released model measurements and hardware validation are indexed in [Benchmarks](../BENCHMARKS.md) and the [support matrix](../SUPPORT_MATRIX.md).
