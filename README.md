<p align="center">
  <img src="docs/assets/veladraft-hero.png" alt="VelaDraft — One capture. Many drafters." width="960" />
</p>

<p align="center">
  <strong>A shared toolkit for multimodal drafter training, systems analysis and benchmarking.</strong><br />
  VelaDraft brings parallel block-diffusion drafters such as DFlash and DSpark
  into a common multimodal workflow, and uses training-cost profiling to guide
  workload-aware system optimizations.
</p>

<p align="center">
  <sub>Primary verified targets: Qwen2.5-VL-3B and Qwen3-VL-8B &mdash; verification scope differs
  by method, recipe and run mode; SmolVLM2 so far has a DFlash single-image smoke.
  See the <a href="SUPPORT_MATRIX.md">support matrix</a>.</sub>
</p>

<p align="center">
  <a href="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml"><img src="https://github.com/LUOseu/VelaDraft/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-12263a" alt="Python 3.10+" /></a>
  <a href="https://github.com/LUOseu/VelaDraft/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-12263a" alt="Apache-2.0" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/model-zoo.md">Model Zoo</a> ·
  <a href="BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="SUPPORT_MATRIX.md">Support Matrix</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

---

## 🎉 News

- **[2026-10-29]** 🤗 Thanks to [@khazic](https://github.com/khazic) for training with **VelaDraft** and sharing **14 checkpoint repositories** for **Qwen3.6** and **Qwen3-VL-8B** on Hugging Face, covering all seven drafter methods!
  - **Qwen3.6:** [DFlash](https://huggingface.co/khazic/veladraft-qwen36-400k-dflash) · [DFlash2](https://huggingface.co/khazic/veladraft-qwen36-400k-dflash2) · [DSpark](https://huggingface.co/khazic/veladraft-qwen36-400k-dspark) · [DFly](https://huggingface.co/khazic/veladraft-qwen36-400k-dfly) · [DFlare](https://huggingface.co/khazic/veladraft-qwen36-400k-dflare) · [Domino](https://huggingface.co/khazic/veladraft-qwen36-400k-domino) · [JetSpec](https://huggingface.co/khazic/veladraft-qwen36-400k-jetspec)
  - **Qwen3-VL-8B:** [DFlash](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflash) · [DFlash2](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflash2) · [DSpark](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dspark) · [DFly](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dfly) · [DFlare](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflare) · [Domino](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-domino) · [JetSpec](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-jetspec)

- **[2026-10-21]** Added **Ascend NPU training support**. All seven registered drafter methods have BF16 offline-training, token-exact evaluation and export evidence on Ascend 910B3; DFlash also has additional online and multi-device-placement checks. These are engineering validation runs, not production-scale performance claims. See the [NPU setup and validation guide](docs/npu/README.md).

- **[2026-10-18]** 🔥 Added the **Qwen3.6-35B-A3B target adapter** with partial
  rotary support for regeneration, feature capture and dense drafter training
  ([#18](https://github.com/LUOseu/VelaDraft/pull/18), thanks to
  [@khazic](https://github.com/khazic)). See the [support matrix](SUPPORT_MATRIX.md#targets).

## Performance evaluation

Use [`vlm-dflash native-eval`](docs/benchmark.md#native-vllm-e2e) for native vLLM
paired request E2E: unchanged native AR defaults, Flash linear SD, separate
processes/warmup, ratio of total wall times and session-cluster intervals. It
requires a prepared complete export and an explicit pinned adapter for spatial
MRoPE/additional method modules. Existing `eval` and `scripts/benchmark.py` are
HF checkpoint diagnostics; their prepared-input timing is not the native E2E
headline. Training/NPU defaults are unchanged. Observed token agreement and real
quality are reported separately; this update publishes no new speedup claim.


## Why VelaDraft

Parallel block drafters combine an expensive frozen multimodal target with a
smaller trainable model. Their training cost is shaped by repeated target-feature
extraction, anchor-expanded batches and full-vocabulary scoring, not just the
size of the drafter. Comparing algorithms also requires keeping data, target
identity, decoding and measurement protocols explicit.

VelaDraft connects three pieces of that research workflow:

- **Train through shared interfaces.** Prepare image-text data, regenerate
  target answers, capture features, and train different drafter families without
  rebuilding the surrounding pipeline.
- **Profile and optimize the training workload.** Measure where time and memory
  go, then reduce redundant extraction and scoring work with explicit numerical
  and resource trade-offs.
- **Benchmark with traceable evidence.** Keep acceptance, task quality,
  output fidelity, latency and training cost distinct; link reported results to
  configurations, checkpoints and validation records.

<p align="center">
  <img src="docs/assets/how-it-works.svg" alt="VelaDraft workflow: prepare image-text data and target features, train a chosen drafter, then evaluate and export. A training-systems loop profiles extraction and scoring, optimizes reuse and wasted work, and validates numerical and resource trade-offs." width="1100" />
</p>

“Unified” means shared interfaces and protocol records, not identical objectives,
budgets or maturity across algorithms. `VelaDraft` is the project; the package
and CLI are still called `vlm-dflash`.

## Training systems: from profiling to optimization

The systems work follows the workload rather than applying every optimization
to every run. Current implementation and measured evidence include:

| Training characteristic | System response | Evidence and boundary |
|---|---|---|
| A frozen target can repeat feature extraction across epochs | Compatible offline-store reuse; optional byte-bounded host-RAM caching online, with adaptive budgets and pinned retention | Cache hits preserve stored tensors, but re-extraction with different batch padding can change bf16 numerics. [Modes, counters and numerical caveats](docs/online-training.md) |
| Anchor expansion sends zero-weight padding blocks through the LM head and loss | Compact padding rows before row-local scoring; skip unused CE scalars in recipe runs | Gradient-exact regression tests; row compaction is currently enabled automatically for DFlare/DFly, not all objectives. [Scoring optimization](docs/online-training.md#recipe-run-scoring-cost-drafterrecipe_row_compaction) |
| Some recipe stages spend substantial time in chunked TV/KL arithmetic | Opt-in compilation of the existing arithmetic, with eager fallback | Hardware- and stage-dependent; does not apply to every loss. [Execution optimization](docs/online-training.md#execution-optimization-compile-the-tvkl-chunk-arithmetic) |

Profiling also tells us what **not** to prioritize: on the recorded A6000
real-row sweep, larger teacher groups reduced extraction time by 13.3%, but the
estimated end-to-end ceiling was below 3%. Fewer forwards alone were not enough
to justify that direction. See the [measurement report](research/system_optimization/REPORT.md).

These are scoped engineering results, not universal speedup claims. For example,
pinned retention reduced paired production-CLI wall time by **3.6% / 6.9%** at
0.8 / 1.6 GiB cache caps on a 512-row, three-epoch workload (two pairs per cap).
The [full protocol](docs/online-training.md#static-pinned-retention-trainonline_feature_cache_policy-pinned)
separates this result from replay upper bounds. Training savings do not establish
inference speedup, and cache policies are opt-in rather than an assumed setting
of every benchmark.

## Supported methods

Seven drafter families share the `train` / `eval` / `export` interfaces and
checkpoint format. The statuses below summarize existing evidence, not validation
on every target or a completed large-scale ranking:

| Method | Idea | Status |
|---|---|---|
| [DFlash](docs/algorithms/dflash.md) | Block-parallel drafting | Validated |
| [DSpark](docs/algorithms/dspark.md) | Markov correction + confidence | Validated |
| [Domino](docs/algorithms/domino.md) | Causal GRU correction | Validated |
| [JetSpec](docs/algorithms/jetspec.md) | Causal block attention + forward KL | Validated with sequential serving (the parallel tree runtime is not ported) |
| [DFlash2](docs/algorithms/dflash2.md) | Local mixing + candidate selector | Experimental objective reconstruction (upstream objective undisclosed) |
| [DFlare](docs/algorithms/dflare.md) | Layer-wise target fusion | Validated |
| [DFly](docs/algorithms/dfly.md) | Fusion residual + hidden correction | Validated, including the ported upstream two-stage recipe |

Recipe selection (default objective, optional recipe, required teacher signal,
released weights) is one table in
[docs/README.md](docs/README.md#choose-a-method--recipe); per-target evidence
states live in [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) (generated).

## Current scope and limitations

- **Input:** images (one or many) + text, single- or multi-turn, greedy
  decoding. Default multi-turn supervision is `last_turn_only`: earlier
  assistant turns are teacher-forced context, only the last one is regenerated
  and supervised ([docs/data.md](docs/data.md)).
- **Hardware:** the validated reference is a single 48 GB GPU; the 24 GB preset
  remains a *suggested* configuration (no recorded run on a real 24 GB card).
- **Ascend NPU:** all seven methods have BF16 offline training, token-exact
  evaluation and export evidence on two Ascend 910B3 devices. DFlash also has
  online, split teacher/drafter, sharded-target and 200-step checks. These are
  small engineering runs; DDP remains unsupported. See the
  [NPU results and setup](docs/npu/README.md).
- **Training modes:** offline staged features (default), online live
  extraction from a frozen target, and `--features` store reuse. There is no
  DDP / multi-node launch path. "Online" is not on-policy rollout training.
- **Qwen3.6-35B-A3B:** the `qwen3_5_moe` adapter is implemented and CPU-tested;
  the method validation statuses above do not establish validation on this
  target. Set `drafter.intermediate_size` explicitly for the dense drafter.
  [Sharded target loading](docs/online-training.md#devices-what-is-supported-today) and
  [chat-template kwargs](docs/REFERENCE.md#chat-template-kwargs-render-protocol)
  are also available, including `enable_thinking: false`. End-to-end training
  remains unvalidated; speculative `eval` / `benchmark` / `spec_generate`
  are explicitly blocked until hybrid recurrent-cache rollback is implemented.
- **Sharded evaluation:** target sharding for capture/training does not imply
  sharded serving support. `eval` / `benchmark` currently reject a nonempty
  `target.device_map`.
- **Serving:** production vLLM/SGLang integration is future work;
  [serving-bridge.md](docs/serving-bridge.md) is an experimental export/load
  tool whose correctness and performance gates are not closed.
- **Reproducibility:** data and generation protocols reproduce; bf16 training
  is not bitwise-deterministic.

## Start here

| Path | What it gives you |
|---|---|
| **[Run the full pipeline](docs/getting-started.md)** | install, prepare a local target, `doctor`, a first demo, then `train` on your own data |
| **[Try a released drafter](docs/model-zoo.md)** | a pretrained checkpoint now &mdash; `demo --drafter` reports acceptance length, per-slot acceptance and token-exactness with no training |

```bash
curl -fL -O https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz
vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
```

Recorded on the reference host: 42 s, AL 1.313, 4/4 token-exact (the
random-init baseline demo: 45 s, AL 1.000). Both DFly variants are published
&mdash; the shared-CE default and the ported upstream recipe:

| Drafter | Target | Download |
|---|---|---|
| DFlash | Qwen2.5-VL-3B | [387 MB `zoo-v0.3.0`](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz) |
| DFly (shared block CE) | Qwen2.5-VL-3B | [420 MB `zoo-v0.3.1`](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.1/qwen25vl-3b-dfly.tar.gz) |
| DFly (upstream two-stage recipe, AL 1.839) | Qwen2.5-VL-3B | [420 MB `zoo-v0.3.2`](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.2/qwen25vl-3b-dfly-recipe.tar.gz) |
| DFlash | Qwen3-VL-8B | [1.00 GB `zoo-v0.3.1`](https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.1/qwen3vl-8b-dflash.tar.gz) |

All nine archives (Domino, DSpark, JetSpec, DFlash2 and DFlare included),
their checks and load commands: [docs/model-zoo.md](docs/model-zoo.md).

## Quick start

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft

python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"        # source install; there is no PyPI release yet

# Prepare the target once (configs run local_files_only and never download it).
export VLM_DFLASH_TARGET="$PWD/assets/models/Qwen2.5-VL-3B-Instruct"
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$VLM_DFLASH_TARGET"
# restricted network: prefix the same command with HF_ENDPOINT=https://hf-mirror.com

vlm-dflash doctor --target "$VLM_DFLASH_TARGET"
# success: exit 0, [ok] lines, and a resolved contract line
# (mask=151669 mrope=[16,24,24] layers=[1,17,33] ...)

vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

`train-demo` runs the whole staged pipeline (prepare → target-native regen →
capture → train → eval → export) on generated images. Recorded on the reference
host (single RTX A6000 48 GB, historical): 3 min 13 s, loss 13.0 → 0.013, eval
AL 1.138 vs random-init 1.000, 4/4 token-exact, export under
`runs/demo_dflash/export/`. It is a plumbing smoke, not a quality result.

Longer runs, logs, stop conditions, resume and hardware details:
[docs/getting-started.md](docs/getting-started.md) ·
[docs/training.md](docs/training.md).

## Three things you can do today

**1. Train on your own image-text data.** One JSON object per line; `image` is
resolved against `--images-dir`:

```jsonl
{"id":"row-001","image":"cat.jpg","question":"What is in this image?","answer":"A cat sleeps on a sofa."}
```

```bash
python examples/train_from_jsonl.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images \
  --limit 32 --steps 60
```

Runs prepare → regen → capture → train → eval and leaves checkpoints plus an
export in the run directory. Recorded (12 COCO rows, 10 steps): loss 7.69 →
7.06, visual check passed, token-exact 4/4. `--limit` is the number of
**training** rows &mdash; validation rows are drawn on top. Row shapes, split
rules and multi-turn supervision: [docs/data.md](docs/data.md).

**2. One capture, compare several drafters.** Regen + capture run once (the
expensive target work); each drafter then trains from the same store with the
target out of the loop:

```bash
python examples/reuse_features_compare.py \
  --target "$VLM_DFLASH_TARGET" \
  --jsonl data/rows.jsonl --images-dir data/images --steps 20
```

Recorded (12-row COCO pool, 5 steps, smoke scale): DFlash AL 1.153 / DSpark
AL 1.094, both token-exact. Reuse is verified, not assumed &mdash; the store's
target identity and the layers a method consumes must match, and mismatches
fail loudly ([docs/training.md](docs/training.md)). The CLI form is
`vlm-dflash train --features <store>`.

**3. Pick a training mode by resources.** Offline reads a staged feature store
(target off the GPU; ~9.4 GiB disk for the 2K capture); online keeps the frozen
target resident and extracts live; a multi-epoch online run can opt into a
host-RAM record cache (never persisted):

```bash
# offline (default)
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --run-dir runs/dfly_2k

# online, with the optional RAM cache
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml --mode online \
  --run-dir runs/dfly_online --set train.online_feature_cache_gb=8
```

Cache budgets, pinned/adaptive policies, resume behavior and the numerical
conditions for reuse are documented in [online training](docs/online-training.md).
Choose a mode for the available memory and storage, and record it in the run
protocol; do not silently change it inside a controlled comparison.

## Results and evidence

The public ledger contains recorded smokes, canonical runs and systems studies.
Large-scale experiments in progress are not included as completed results.

Everything measured &mdash; acceptance length, per-slot acceptance, token-exact
gates, training costs, ablations, runtime probes &mdash; is recorded with its
protocol in [BENCHMARKS.md](BENCHMARKS.md) ·
[RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) ·
[docs/measured-runs.md](docs/measured-runs.md). Machine-readable subset:
`benchmarks/results/zoo_canonical.json`.

What the numbers mean, precisely:

- **AL** is accepted tokens per target verification round &mdash; it is not a
  speedup multiplier. **Token-exact** means the speculative output reproduces
  the target's own greedy decoding; it is an output-consistency gate, not task
  accuracy. Task scores against ground truth (TextVQA/DocVQA/ChartQA answer
  accuracy) are **not reported yet**.
- The canonical benchmark runs are the **offline** path; online smokes validate
  plumbing (token-exactness, provenance, counters), not quality.
- Coverage is explicit: [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) states what is
  canonical-validated, reduced, smoke-only or unsupported per method × target ×
  mode, each row traced to recorded run ids.

## Documentation

[docs/README.md](docs/README.md) is the task index: install and first run, own
data, method/recipe selection, long training, online mode, evaluation and
troubleshooting. Direct links: [model-zoo.md](docs/model-zoo.md) ·
[data.md](docs/data.md) · [training.md](docs/training.md) ·
[online-training.md](docs/online-training.md) ·
[troubleshooting.md](docs/troubleshooting.md) ·
[npu/](docs/npu/README.md) ·
[algorithms/](docs/algorithms/) · [benchmark.md](docs/benchmark.md) ·
[env-preflight.md](docs/env-preflight.md) (historical host notes) ·
[REFERENCE.md](docs/REFERENCE.md) (implementation and numerical semantics).

## Contributing and citation

Reproductions, focused fixes and clear usage examples are welcome; the current
scope and release policy are in [CONTRIBUTING.md](CONTRIBUTING.md) and
[ROADMAP.md](ROADMAP.md). Algorithms plug in through `DrafterAlgorithm`, target
families through `VLMAdapter`.

Apache-2.0; ported components keep their upstream attribution
([NOTICE](NOTICE), [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md)).

```bibtex
@misc{veladraft2026,
  title = {VelaDraft: Multimodal Speculative Drafter Training},
  year  = {2026},
  url   = {https://github.com/LUOseu/VelaDraft}
}
```

---

<p align="center">
  <sub><strong>Why “Vela”?</strong> Vela means sail &mdash; and names the constellation.
  A lightweight drafter rides the visual context the target model provides.</sub>
</p>
