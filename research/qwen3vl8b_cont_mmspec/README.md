# Qwen3-VL-8B CONT seven-method MMSpec evaluation (portable runner)

[AGENTS.md](AGENTS.md) is the task handoff written for the A6000 evaluation machine. That machine's
native exports, runtime manifests and frozen runtime are not part of the repository, so this
directory rebuilds them from the CONT checkpoints and runs the same `vlm-dflash native-eval`
protocol on the seven CONT finals (75,000 original + 50,000 continuation updates, step 125,000).

## Contents

| Path | Purpose |
| --- | --- |
| `native/` | Method adapters bound to the frozen Qwen3-VL-8B draft. Ported from `research/qwen36_native_vllm` (itself a port of `research/qwen3vl8b_native_vllm`), with the target contract in `native/contract.py`. |
| `native/export_checkpoint.py` | CPU export of one CONT final to the native layout; every trained tensor bitwise equal, receipt written next to it. |
| `native/make_runtime_manifest.py` | Runtime manifest pinning one method's bootstrap, adapter sources and checkpoint by SHA256. |
| `native/test_method_parity.py`, `native/test_mrope_parity.py` | CPU parity against the training implementation with the real CONT weights and target head. |
| `scripts/prepare_exports.sh` | Exports and manifests for all seven methods. |
| `scripts/run_qwen3vl8b_cont_mmspec.sh` | Campaign launcher from the handoff: per-GPU method queues, smoke then formal, summary CSV, optional HF publication. |

## Run

Machine paths live in a site env file outside the repository (`Q3VL8B_SITE`): `Q3VL8B_ROOT`
(holding `exports/`), `TARGET`, `SOURCE`, `IMAGES`, `PYTHON`, `RESULT_ROOT`, optionally `SHM`,
`HF_HOME` and `Q3VL8B_PRELAUNCH`.

```bash
PYTHON=<eval python> bash research/qwen3vl8b_cont_mmspec/scripts/prepare_exports.sh <cont>/runs <root>/exports
```

```bash
Q3VL8B_SITE=<site.env> bash research/qwen3vl8b_cont_mmspec/scripts/run_qwen3vl8b_cont_mmspec.sh --preflight --methods all
```

```bash
Q3VL8B_SITE=<site.env> bash research/qwen3vl8b_cont_mmspec/scripts/run_qwen3vl8b_cont_mmspec.sh --gpus 0,1,2,3 --methods all --suite all
```

## Differences from the A6000 handoff

| Item | Handoff | Here |
| --- | --- | --- |
| Hardware | A6000 | site GPU, recorded as `gpu_name` in `run_manifest.json`; speeds are only comparable within one GPU type |
| Runtime | `runtime_frozen_v1` (vLLM 0.22.1) | a separately installed vLLM 0.22.1 environment; the adapter checks the four reviewed vLLM source files by SHA256 |
| Exports and manifests | prepared on the A6000 machine | rebuilt from the CONT `ckpt_final.pt` by `prepare_exports.sh` |
| DSpark | K=8 in a dedicated environment | K=7: the adapter verifies the trained block of 8 (anchor + 7 draft queries) through the official DFlash proposer, as for the other six methods |
| Domino | blocked (no reviewed manifest) | runs through `native/head_adapters.py` (anchor-slot drafting, prefix GRU, correction head) |
| GPU ownership | PID mode | count mode by default, for container PID namespaces where nvidia-smi reports host PIDs: each card may hold the client context plus one engine core (`VELADRAFT_GPU_OWNERSHIP=pid` restores PID mode) |

Everything else follows the handoff: batch 1, greedy, `max_new_tokens` 1024 with natural EOS,
three repeats, `gpu_memory_utilization` 0.78 for AR and SD, PIECEWISE SD graphs, MMSpec full test
(600 sessions, 723 turns, 2169 paired turns), headline `sum(AR wall) / sum(SD wall)`.
Quality is `N/A`: this MMSpec input has no gold answers or scorer.
