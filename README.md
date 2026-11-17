# VelaDraft

### Adapting Parallel Block Drafters to Vision-Language Models

Anonymous code accompanying the paper. VelaDraft separates target-model adaptation,
drafter-specific logic, and shared execution so that parallel drafters can share
multimodal training and native inference infrastructure.

This review snapshot focuses on the paper's implementations, configurations, and
tests. Model weights and datasets are external inputs; historical run logs and
deployment infrastructure are not included.

## What is included

- **Seven drafters:** DFlash, DFlash2, DSpark, Domino, JetSpec, DFlare, and DFly.
- **Two principal targets:** Qwen3-VL-8B-Instruct (dense) and Qwen3.6-35B-A3B
  (hybrid MoE).
- **Training:** reusable multimodal teacher features, Context–Anchor-aware batch
  execution, and objective-aware vocabulary scoring.
- **Deployment:** native vLLM adapters and short-query expert grouping that
  preserves the router's expert choices and weights.

## Start with the paper's key code

| Component | Implementation |
| --- | --- |
| Target adapters and drafter plugins | [`vlm_dflash/adapters`](vlm_dflash/adapters), [`vlm_dflash/algorithms`](vlm_dflash/algorithms) |
| Visual-aware feature reuse | [`target/capture.py`](vlm_dflash/target/capture.py), [`target/online.py`](vlm_dflash/target/online.py) |
| Batch splitting and regrouping | [`dflash/train.py`](vlm_dflash/dflash/train.py), [`dflash/regroup.py`](vlm_dflash/dflash/regroup.py) |
| Objective-aware scoring | [`dflash/chunked_ce.py`](vlm_dflash/dflash/chunked_ce.py), [`dflash2/trainer.py`](vlm_dflash/dflash2/trainer.py), [`dfly/trainer.py`](vlm_dflash/dfly/trainer.py) |
| Short-query expert grouping | [`dispatch.py`](research/qwen36_grouped_e2e_20261102/dispatch.py) |
| Native evaluation | [`eval/native.py`](vlm_dflash/eval/native.py), [dense target](research/qwen3vl8b_cont_mmspec), [MoE target](research/qwen36_a100_seven_eval) |

[Paper-to-code and execution guide](docs/PAPER_GUIDE.md) ·
[Training configurations](configs) · [Tests](tests) · [Third-party notices](NOTICE)

## Installation

Use Python 3.10 or newer. The Python package remains `vlm_dflash` and its command
is `vlm-dflash`.

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -e '.[tested,dev]'
vlm-dflash --help
```

Training uses PyTorch and Transformers. Native inference uses **separate**
environments: vLLM **0.22.1** for the dense integration and **0.32.0** for the
hybrid MoE integration. These adapters are version-specific.

## Minimal training example

Provide a local target checkpoint and image/text data. First check the target
contract, then run a short end-to-end functionality check:

```bash
export VLM_DFLASH_TARGET=/path/to/Qwen3-VL-8B-Instruct
vlm-dflash doctor --target "$VLM_DFLASH_TARGET"
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" \
  --run-dir runs/quickstart --steps 20
```

For your own data, use a JSONL file, for example:

```json
{"id":"sample-1","image":"chart.png","question":"Which bar is tallest?","answer":"The blue bar."}
```

```bash
vlm-dflash train \
  --config configs/dflash2/qwen3_vl_8b_reduced.yaml \
  --target "$VLM_DFLASH_TARGET" \
  --dataset-source jsonl --dataset /path/to/rows.jsonl \
  --images-dir /path/to/images --run-dir runs/dflash2
```

This is a small starter configuration, not the paper-scale training run. The
pipeline prepares examples, regenerates supervision with the target, extracts
features, and trains the drafter. Use enough examples for the configured training
and validation splits. See the guide for online caching, offline feature sharing,
formal configurations, and native evaluation.

## Scope and checks

Execution optimizations follow each method's objective. In particular,
Context–Anchor regrouping currently supports the fixed-anchor DFlash path;
unsupported objective combinations are rejected. Native inference needs local
trained checkpoints and method-specific exports. Later continuation evaluations
use a different checkpoint and sample count from the original dense-target table;
the guide identifies these protocols separately.

```bash
python -m pytest tests
```

These tests check algorithms, feature contracts, batch execution, scoring,
exports, and evaluation bookkeeping. They do not reproduce GPU performance
measurements. See [validation](docs/VALIDATION.md) for checks performed on this
snapshot.

## License

Apache-2.0, with the third-party licenses and attribution retained in
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and the relevant source headers.
