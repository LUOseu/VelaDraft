# VelaDraft documentation

Task-first entry points. New here? Start with the first line below.

## Install and first run
- [getting-started.md](getting-started.md) - install, prepare a local target model, `doctor`, the first demo, and what success looks like.
- [../examples/README.md](../examples/README.md) - runnable example scripts.

## Train on your own data
- [data.md](data.md) - JSONL shapes, train/val split, target-native regeneration, multi-turn supervision (default `last_turn_only`).
- [training.md](training.md) - full training runs, server long runs, logs, checkpoints, resume and outputs.
- [online-training.md](online-training.md) - live feature extraction, batching, optional RAM cache, device/limits.

## Choose a method / recipe
Per-method details: [TRAINING_SIGNALS.md](TRAINING_SIGNALS.md) (generated requirements), [algorithms/](algorithms/) (one page per drafter), [SUPPORT_MATRIX.md](../SUPPORT_MATRIX.md) (recorded evidence; generated), [../ALGORITHM_MATRIX.md](../ALGORITHM_MATRIX.md) (implementation progress).

| Method | Default objective | Selectable recipe | Config entry | Extra teacher signal | Recorded scope (highest) | Released weights |
| --- | --- | --- | --- | --- | --- | --- |
| [dflash](algorithms/dflash.md) | decayed block CE | `upstream_two_stage` | `configs/qwen2_5_vl_3b_2k.yaml` | final hidden only under the recipe | canonical 2K, 3B; full 2K, 8B | `zoo-v0.3.0` (3B), `zoo-v0.3.1` (8B) |
| [dspark](algorithms/dspark.md) | decayed block CE (+ greedy acceptance confidence) | `upstream_dspark` | `configs/dspark/qwen2_5_vl_3b_2k.yaml` | final hidden under the recipe | canonical 2K, 3B; 8B reduced | `zoo-v0.3.1` |
| [domino](algorithms/domino.md) | dual-logit CE + curriculum | none | `configs/domino/qwen2_5_vl_3b_2k.yaml` | none extra | canonical 2K, 3B; 8B reduced | `zoo-v0.3.0` |
| [jetspec](algorithms/jetspec.md) | forward-KL distillation | none | `configs/jetspec/qwen2_5_vl_3b_2k.yaml` | final hidden (`capture.store_final_hidden: true`) | canonical 2K, 3B; 8B reduced | `zoo-v0.3.1` |
| [dflash2](algorithms/dflash2.md) | experimental reconstruction (unary KL + selector CE) | none | `configs/dflash2/qwen2_5_vl_3b_2k.yaml` | final hidden | canonical 2K, 3B (experimental objective) | `zoo-v0.3.1` |
| [dflare](algorithms/dflare.md) | shared block CE | `upstream_two_stage` | `configs/dflare/qwen2_5_vl_3b_2k.yaml` | final hidden only under the recipe | canonical 2K, 3B; 8B reduced | `zoo-v0.3.1` |
| [dfly](algorithms/dfly.md) | shared block CE | `upstream_two_stage` | `configs/dfly/qwen2_5_vl_3b_2k.yaml` (recipe: `..._2k_recipe.yaml`) | final hidden under the recipe | canonical 2K: CE 1.682 / recipe 1.839; 8B reduced | `zoo-v0.3.1` (CE), `zoo-v0.3.2` (recipe) |

"none" in the recipe column means the method has a single default objective - it
does **not** mean the method lacks a training objective. Configs are recipes:
set your target/data paths first. Download commands and per-archive checks live
in [model-zoo.md](model-zoo.md).

## Evaluate and export
- [benchmark.md](benchmark.md) - benchmark harness usage; results live in [../BENCHMARKS.md](../BENCHMARKS.md).
- [verification-modes.md](verification-modes.md) - what token-exact verification means and which modes are diagnostics.
- [serving-bridge.md](serving-bridge.md) - export -> serving runtime status, current gates and what must not be claimed.

## Troubleshoot and extend
- [troubleshooting.md](troubleshooting.md) - common failures with the exact place to check.
- [adding_an_algorithm.md](adding_an_algorithm.md) - add a new drafter through the registry.
- [REFERENCE.md](REFERENCE.md) - implementation and numerical-semantics reference (ported code, MRoPE/capture details); not a beginner manual.
- [env-preflight.md](env-preflight.md) - historical development-host notes (paths, pinned versions).

## Research records
- [plan/](plan/) - experiment plans, run logs, claim-evidence tables (development history).
- [../research/](../research/) - research workspace (optimization lanes, negative results).
- [../CHANGELOG.md](../CHANGELOG.md) - release history.
