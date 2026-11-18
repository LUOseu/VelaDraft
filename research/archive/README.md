# Research archive

These studies retain their run reports, source snapshots and evidence. Use the [MMSpec runners](../README.md) for the current campaign entry points.

## Training studies

| Study | Start here |
| --- | --- |
| Feature reuse, cache policy and scoring costs | [System optimization report](training/system_optimization/REPORT.md) |
| Vocabulary projection and memory cost | [Training-cost report](training/train_cost_20261031/REPORT.md) |
| Context/anchor regrouping and convergence | [Experiment report](training/convergence/REPORT.md) · [A100 runbook](training/convergence/a100/RUNBOOK.md) |
| Earlier training-systems investigations | [Study summary](training/rsi/_coord/SUMMARY_D0_D5_20261015.md) |
| Three-layer and five-layer DFly comparison | [R3/R5 report](training/r3r5/REPORT_R3R5_NEXT_ROUND.md) · [artifacts](training/r3r5/artifacts/) |

## Native inference studies

| Study | Start here |
| --- | --- |
| Qwen3-VL-8B original A6000 campaign | [Adapters, scoring and audit snapshot](qwen3vl8b_native_vllm/README.md) |
| Qwen3.6 verification length | [K sweep protocol](qwen36/k_sweep/PROTOCOL.md) · [verification findings](qwen36/verifylen/FINDINGS_K1_K3.md) |
| Qwen3.6 backend and kernel attribution | [Fixed-K experiments](qwen36/backend/README.md) · [root-cause probes](qwen36/rootcause/) |
| Qwen3.6 generation-budget controls | [vLLM 0.32 budget sweep](qwen36/budget_v032/) · [fixed-length comparison](qwen36/fixedlength/) |

Recorded manifests and host logs refer to the checkout used for the original run. To follow an old path literally, open the [original research tree](https://github.com/LUOseu/VelaDraft/tree/6771a4fac6d9eda9a26eb9237fa6c39e5c3265f4/research) or the [original R3/R5 reports](https://github.com/LUOseu/VelaDraft/tree/6771a4fac6d9eda9a26eb9237fa6c39e5c3265f4/reports). This command creates a separate checkout of that layout.

```bash
git worktree add --detach ../VelaDraft-before-layout 6771a4fac6d9eda9a26eb9237fa6c39e5c3265f4
```
