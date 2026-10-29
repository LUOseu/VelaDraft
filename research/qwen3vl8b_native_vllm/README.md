# Qwen3-VL-8B native vLLM evaluation snapshot

This directory preserves the evaluation and method-adapter sources used for the
A6000 Qwen3-VL-8B campaign. The supported, general CLI remains
[`vlm-dflash native-eval`](../../docs/benchmark.md#native-vllm-e2e). The
research adapters here are **not** native implementations of these algorithms
provided by vLLM, and are not automatically installed by the CLI.

## Contents

| Path | Purpose |
| --- | --- |
| `spatial_adapter/` | DFlash three-axis MRoPE and graph adapter. |
| `dfly_recovery_v2/dfly_native_adapter_v1/` | DFly fusion, correction and spatial adapter. |
| `dflash2_recovery_v2/dflash2_8b_native_adapter/` | DFlash2 selector, convolution and spatial adapter. |
| `dflare_8b_work_20261029/` | DFlare 8B adapter, CPU checks and queues. |
| `domino_8b_work_20261029/` | Domino 8B adapter, CPU checks and queues. |
| `jetspec_8b_work_20261029/` | JetSpec adapter and queues for the **local linear shared-block** implementation; not the upstream tree runtime. |
| `quality_eval_v1/`, `quality_dflash2_dynamic_v6/`, `quality_{dflare,domino,jetspec}_v1/` | Paired real-gold TextVQA, DocVQA and ChartQA scoring code. The vendored scorers retain their upstream licenses. |
| `audit_native_evidence.py`, `audit_attention_backends_v1.py`, `audit_stale_resources.py` | Independent request/time, actual backend and resource-history checks. |
| `collect_al_supplement_v1.py`, `prepare_al_harness.py`, `al_supplement_queue_v1.sh.reference` | Separate vLLM official acceptance-counter collection; never substitute its timing for the primary measurement. |
| `SAMPLE_IDENTIFIERS_20261029.json` | 120 performance sessions / 145 turns and three 256-question quality subsets, identified without prompts, images, outputs or gold answers. |
| `RESULTS_SNAPSHOT_20261029.md` | Timestamped results snapshot, not a live or independently reproducible result artifact. |

The six listed methods have 8B checkpoints at 75,000 training steps. A finished
8B DSpark checkpoint and an 8B DSpark adapter are **not** included in this
snapshot. A checkpoint filename alone does not establish convergence.

## What is and is not portable

The adapter and orchestration files preserve machine-specific code, including
absolute paths, source SHA pins, CUDA environment paths and input manifests from
the original server. **They are an evidence-oriented research snapshot, not a
turnkey cross-machine install.** The checkout contains no model weights, input
images, original prompts, gold answers, predictions, full result logs, current 8B runtime approval receipts or a complete independent audit. Some historical 3B review texts and a limited 8B CPU comparison are retained as references; they do not certify a new run. A new machine must supply those inputs,
verify their licenses and checkpoint hashes, adapt the paths, independently
regenerate all source/weight/CPU receipts and runtime manifests, and complete a
GPU smoke before running a formal measurement. Do not reuse the captured JSON
manifests as approval for different code, models, paths or hardware.

Historical shell launchers are retained as non-executable `*.sh.reference` files,
and machine-bound recovery writers are retained as `*.py.reference`. They
contain original fixed PIDs, log destinations and host paths and must not be
run as current job-control commands. The original quality `SOURCE_SHA256.txt`
files refer to the old server layout; do not treat them as checkout-relative
checksums. Rebuild manifests after preparing reviewed paths and a new run ID.
`restore_frozen_runtime_v1.py.reference` assumes the original shared vLLM patch
and binary layout and is not a general-purpose installer. The
captured DFlash2 quality launcher reference has its historical orphan-PID
adoption branch removed, since that PID is unsafe to reuse on another machine. The DFlare
quality `v1` launcher is the historical run; `v2` adds an explicit success gate
for the case where all smoke attempts fail. In the recorded run the `v1` smoke
passed, so the failed-exhaustion branch was not used.

## Measurement semantics

The campaign used A6000, Qwen3-VL-8B BF16, vLLM 0.22.1, batch one and linear
verification with seven speculative tokens (B8). AR and SD were timed fairly
within the same fixed request protocol. The headline is the sum of paired AR
request wall times divided by the sum of SD request wall times, excluding
startup and warmup but including image loading and preprocessing, prefill,
decode and output collection. Each full speed run uses 120 sessions, 145 turns
and three repeats; the 95% bootstrap interval resamples sessions and does not
cover independent runs, cards or host contention. Actual target and draft
attention backends must be checked from the worker records.

Quality uses 256 real-gold examples per task. Score full decoded text as the
primary metric; the historical first-line extraction is a separately reported
sensitivity analysis. This subset is not the entire benchmark. Record score
changes, token-exact outputs and timing separately. Floating-point differences
can change greedy outputs: neither a matching score nor this software alone
proves strict token-by-token losslessness or quality equivalence.

The optional `QWEN8B_DIGEST_CACHE` caches SHA-256 against file device, inode,
size and nanosecond mtime/ctime. It is an I/O optimization, **not** an
adversarial integrity boundary. Omit the variable to hash every observation.
GPU ownership is sampled, not continuously proven, and does not demonstrate
exclusive host CPU or I/O use. Treat foreign GPU use or incomplete logs as an
invalid run under the campaign's predeclared rule.

To make a **separate** stats-enabled copy of the CLI package for an AL run:

```bash
python research/qwen3vl8b_native_vllm/prepare_al_harness.py . /path/to/new-al-harness
```

Run the supplemental collection with the same workload and a freshly verified
runtime manifest. Report its accepted draft tokens, number of drafts and sample
count. The usual vLLM convention is `1 + accepted / drafts`; that convention
includes a bonus token. Those counters were disabled in the original primary
timing runs and cannot be reconstructed from their speedups.
