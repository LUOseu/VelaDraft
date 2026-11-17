# Paper-to-code guide

VelaDraft's implementation is organized around the following components. The Python package and CLI retain the names `vlm_dflash` and `vlm-dflash`.

| Paper component | Implementation |
| --- | --- |
| Target adapters and multimodal positions | `vlm_dflash/adapters/`, `vlm_dflash/contract.py`, `vlm_dflash/rope_layout.py` |
| Seven drafter plugins | `vlm_dflash/algorithms/` and the `dflash`, `dflash2`, `dspark`, `domino`, `jetspec`, `dflare`, and `dfly` implementation directories |
| Visual-aware feature reuse | `vlm_dflash/target/capture.py`, `vlm_dflash/target/online.py`, `vlm_dflash/target/cache_budget.py`, `vlm_dflash/dflash/features.py` |
| Context–Anchor batch execution | `vlm_dflash/dflash/train.py`, `vlm_dflash/dflash/regroup.py` |
| Objective-aware scoring | `vlm_dflash/dflash/chunked_ce.py`, `vlm_dflash/dflash/trainer_core.py`, `vlm_dflash/dflash2/trainer.py`, `vlm_dflash/dflash2/loss.py`, `vlm_dflash/dfly/trainer.py` |
| Native request evaluation | `vlm_dflash/eval/native.py`, `vlm_dflash/eval/native_runtime.py`, `vlm_dflash/export/` |
| Dense target integration | `research/qwen3vl8b_cont_mmspec/native/` |
| Hybrid MoE integration | `research/qwen36_seven_20261102/`, `research/qwen36_v032/` |
| Short-query expert grouping | `research/qwen36_grouped_e2e_20261102/dispatch.py` |

## Training

The actual three-epoch training configurations for all seven methods and both targets are in `configs/paper/qwen3vl8b/` and `configs/paper/qwen36/`. Their README explains local data paths and device placement. They preserve the original 200K/400K data-selection quotas and effective batch sizes of eight and sixteen. Other configurations under `configs/` are small development or compatibility recipes.

The main CLI supports both feature paths:

- `vlm-dflash train --mode online` extracts teacher-forced states during training; `--set train.online_feature_cache_gb=8` sets an 8 GiB host cache.
- `vlm-dflash train --features /path/to/store` reuses a compatible offline feature store.

Feature records must match the target, teacher layers, and multimodal positions. Compatibility checks remain enabled. Optimizations are selected by objective: `train.regroup` currently supports fixed-anchor DFlash, while objective-specific scoring paths serve DFlash2 and DFly. JetSpec uses the shared-block causal reference path rather than the upstream tree runtime.

## Native evaluation

Use separate environments for training, vLLM 0.22.1, and vLLM 0.32.0. The native adapters require their matching runtime and local trained checkpoints.

| Target | Entry point | Scope |
| --- | --- | --- |
| Qwen3-VL-8B | `research/qwen3vl8b_cont_mmspec/` | Seven-method continuation evaluation with vLLM 0.22.1, step-125,000 checkpoints, and 600 MMSpec sessions. This differs from the earlier step-75,000, 120-session campaign. |
| Qwen3.6-35B-A3B | `scripts/run_qwen36_a100_benchmarks.sh` | Portable seven-method vLLM 0.32.0 runner; configuration in `configs/eval/qwen36_a100_seven.yaml`. The paper's MoE timings used an A800, so the A100 runner name does not establish reproduction of those timings. |

`vlm-dflash native-eval` measures paired complete requests. `vlm-dflash eval` instead runs Hugging Face checkpoint and acceptance diagnostics. The latter is not the paper's native end-to-end timing path.

Short-query grouping selects vLLM's existing expert-alignment path for supported 3–8-token queries with top-8 routing and 256 experts. It preserves router selections and weights and falls back to native dispatch for other shapes, including one-token AR queries. An isolated stock/grouped comparison must hold the remaining configuration fixed. The retained A800 expert tile configuration is hardware-specific.

Model weights, prepared data, and optimizer snapshots are external inputs. The supplied tests check implementation behavior; paper-scale training and hardware performance must be evaluated in the corresponding runtime.

## Execution measurements

`examples/reuse_features_compare.py` captures features once and trains two methods
from the compatible store. `scripts/profile_online_extraction.py` profiles the
online extraction path. `scripts/benchmark_teacher_projection.py` measures
scoring and backward computation on controlled shapes.

`scripts/q3vl8b_trainopt/make_configs.py` defines the paired reference/optimized
arms for DFlash2 and DFly; it requires the corresponding original effective
configs and optimizer snapshots. `numcheck.py` checks paired updates and
`recompute_results.py` summarizes locally supplied campaign results. These tools
are not bundled historical results or a substitute for running the experiments.
