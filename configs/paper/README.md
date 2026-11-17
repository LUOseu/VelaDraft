# Paper training recipes

These 14 configurations were copied from the actual training runs' `run/config.json` files. Only the target, data, metadata, and output paths were replaced with configurable local paths. Model settings, objectives, optimizer parameters, seeds, and device placement are preserved.

| Target | Directory | Selected examples | Epochs | Effective batch |
| --- | --- | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | `qwen3vl8b/` | 200,000 | 3 | 8 |
| Qwen3.6-35B-A3B | `qwen36/` | 400,000 | 3 | 16 |

Each directory contains `dflash.json`, `dflash2.json`, `dspark.json`, `domino.json`, `jetspec.json`, `dflare.json`, and `dfly.json`.

```bash
export VELADRAFT_TARGET=/path/to/Qwen3-VL-8B-Instruct
export VELADRAFT_DATA_ROOT=/path/to/prepared-pool
export VELADRAFT_RUN_ROOT=runs/paper
vlm-dflash train --config configs/paper/qwen3vl8b/dflash2.json
```

The data directory must contain the prepared `pool.jsonl` and `pool_meta.jsonl`, with valid local image paths. The metadata supplies the category-based regeneration quotas. For Qwen3.6, the original candidate-pool limit is 660,000; the quotas select 400,000 examples. This limit is intentionally preserved.

The original Qwen3-VL runs place the target on `cuda:1` and the drafter on `cuda:0`. The Qwen3.6 recipes start the target on `cuda:0`, use `device_map: sequential` with memory budgets for two GPUs, and place the drafter on `cuda:1`. Preserve or explicitly adapt all device and memory-map settings to your hardware; changing them changes the execution configuration.

The formal recipes select `regen.backend: vllm`; target-answer regeneration therefore also requires a compatible vLLM installation. Prepare the regenerated data in the corresponding target runtime before comparing training execution. The small starter configurations use the simpler Transformers pipeline.

These are the original three-epoch recipes. The dense-target continuation evaluation uses later checkpoints. Prepared datasets, checkpoints, and optimizer snapshots are separate inputs and are not included in this source release.
