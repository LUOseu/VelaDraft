# Ascend hardware results — 2026-10-20

The supported single-process training pipeline was exercised on two Ascend
910B3 cards (64 GiB each), aarch64 Linux, driver 26.0.rc1, CANN 9.0.0 toolkit
and 910b ops, Python 3.10.14, torch/TorchNPU 2.10.0 and Transformers 5.12.1.
[Exact environment](evidence/2026-10-20/environment.json) · [install](INSTALL.md).

## Scope and results

The seven method runs below used `train_recipe: null`. They establish each
method's default objective only; they do not validate DSpark's
`upstream_dspark` or DFly's `upstream_two_stage` recipe. Their 32 query rows
and chunk size 32 execute one chunk, and all four-prompt evaluations accepted
zero draft tokens. Token-exact fallback therefore does not establish useful
draft acceptance or a multi-chunk memory benefit. The separate
[2026-10-21 reviewer follow-up](audit-2026-10-21/REPORT.md) records formal
recipe, natural-data, capacity and CUDA evidence with its own limits.

All seven methods used Qwen2.5-VL-3B-Instruct features from **real pretrained
teacher forwards on generated geometric images**. This is a plumbing/overfit
workload, not a natural-image task benchmark: 20 source rows, 16 train / 4 val,
image-grouped split; 48-token regeneration; 1024-token capture; two draft layers,
intermediate size 1024, block size 4, eight anchors, BF16 draft and FP32 masters.
The first DFlash run captured final hidden states so all methods could reuse
one compatible store. No feature tensors were invented in these real-target runs.

| Method | Optimizer steps | Final diagnostic loss | Peak training allocation (GB, decimal) | Sequential token-exact check |
|---|---:|---:|---:|---|
| dflare | 10 | 5.875 | 1.378 | 4/4 for init and trained |
| dflash | 10 | 6.65625 | 1.514 | 4/4 for init and trained |
| dflash2 | 10 | 9.68963 | 2.050 | 4/4 for init and trained |
| dfly | 10 | 7.28125 | 1.977 | 4/4 for init and trained |
| domino | 10 | 6.8125 | 1.760 | 4/4 for init and trained |
| dspark | 10 | 6.875 | 1.571 | 4/4 for init and trained |
| jetspec | 10 | 6.4068 | 1.514 | 4/4 for init and trained |

Loss objectives differ; this table is not a method ranking. Memory is the
training allocation with the target absent from offline training, not the total
memory required to load/capture the teacher. Evaluation used four held-out
prompts and at most 16 new tokens, checking both random-init and trained
checkpoints. Every method also exported successfully.
[Raw configurations, contracts, summaries, evaluations and export manifests](evidence/2026-10-20/real_runs.json).

Additional executed cases:

- DFlash: 20-step full prepare → regenerate → capture → train run, plus a
  **200-step** feature-reuse run. No NaN, no skipped supervision batches;
  mean loss over the first/last 20 steps **7.39375 → 2.328515625**. This small
  repeated-data result is not production convergence evidence.
- DFlash online: 10 steps with the frozen target resident on `npu:1`.
- Teacher/drafter split: target `npu:0`, draft `npu:1`, four train rows,
  10 online steps. Host cache: **6 hits / 4 misses / 0 evictions**.
- Target sharding: visual tower on `npu:0`, language model/head on `npu:1`,
  drafter on `npu:0`, four train rows and 10 online steps. The feature-source
  report records both target devices; cache counters are again 6 / 4 / 0.
  This is placement within one process, **not DDP or an HCCL benchmark**.
- Qwen3-VL-2B-Instruct and SmolVLM2-256M-Video-Instruct: DFlash BF16 offline
  training for 10 steps (four train rows), four-prompt init/final sequential
  token-exact evaluation, visual checks and export. This does not validate
  every method against these targets, nor Qwen3-VL-8B on NPU.
- A real NPU-trained DFlash export was copied to macOS and loaded/run on CPU
  with no `torch_npu` import. [Portable-export result](evidence/2026-10-20/portable_export.json).

## Operator, resume and continuity gates

- [H0](evidence/2026-10-20/H0.json): allocation, matrix multiplication/backward,
  RNG save/restore, memory accounting and SDPA.
- [H1](evidence/2026-10-20/H1.json): GQA, BF16 SDPA, mask equality and complete
  draft loss/parameter-gradient parity against CPU.
- [H2](evidence/2026-10-20/H2.json): seven real registered trainers on tiny
  deterministic synthetic features, ten updates, method-specific gradients,
  checkpoint continuation with bitwise model/master/optimizer/scheduler
  equality, and export/reload parity.
- [H3](evidence/2026-10-20/H3.json): 200 synthetic-feature steps, exact midpoint
  resume, per-checkpoint memory samples. Device allocation at checkpoints
  20–200 stayed at 425,472 bytes; host peak RSS rose about 1.5 MiB after the
  first sampled checkpoint. These tiny-shape observations do not establish
  memory stability for arbitrary workloads.
- [Optional fused attention](evidence/2026-10-20/fusion.json): real forward and
  backward, no fallback. Maximum absolute errors: output 1.19e-7, Q gradient
  7.45e-9, K/V gradient 2.24e-8. No fused-training speed claim.

The exact-resume harness enables deterministic algorithms. Without them,
DSpark/DFlash2 trainable lookup gradients showed bit-level differences on NPU;
ordinary CLI training retains PyTorch's default determinism policy. Do not
interpret this harness result as bitwise determinism for default BF16 runs.

## Hardware-discovered fixes

- Domino's native DynamicGRUV2 rejected FP32 weights. The NPU trainer now uses
  the existing differentiable GRU-cell equations over each block. Forward,
  input-gradient, parameter-gradient and complete trainer-loss parity are tested.
- SmolVLM's NPU bucketize rejected BF16. An instance-local vision embedding
  adaptation preserves BF16 coordinate rounding and performs bucketize in FP32;
  it preserves Accelerate hooks. Tile counts now come from the loaded model's
  geometry (64 for the tested 256M model, 81 for 2.2B).
- Fused attention passed an unsupported `return_softmax` keyword; removed it
  against the actual TorchNPU schema.
- Hardware acceptance scripts had CPU frozen-head/batch placement errors,
  incorrect GQA reference setup and incorrect KV-gradient reference aggregation.
- NPU presets could omit consumed target layers and truncate the image/prompt
  before any supervised response; layer routing is now model-derived and the
  context budget is 1024. Explicit `--images-dir` overrides preset defaults.
- Event timers now bind both boundaries to the selected device. Evaluation
  synchronizes target/speculative wall timers consistently.
- Step overrides now reach the effective manifest; requested unexecuted phases
  cannot produce a passing aggregate; exported smoke metadata uses actual steps,
  losses and histories instead of placeholder values.

## Reproduce

After [installing the tested environment](INSTALL.md), the generic hardware
commands are in [HARDWARE_BRINGUP.md](HARDWARE_BRINGUP.md). To reproduce the
image workload, generate the 20 source rows with `write_demo_source`, or supply
an existing JSONL/image directory. The initial capture must store final hidden
states for JetSpec and recipe losses:

```bash
python -c 'from vlm_dflash.demo import write_demo_source; write_demo_source("runs/npu/image-data", 20)'
vlm-dflash train --preset npu-dflash-smoke --target "$VLM_DFLASH_TARGET"   --dataset runs/npu/image-data/demo_source/pool.jsonl   --images-dir runs/npu/image-data/demo_source/images   --steps 20 --run-dir runs/npu/real-dflash --set capture.store_final_hidden=true
vlm-dflash train --preset npu-dspark-smoke --target "$VLM_DFLASH_TARGET"   --dataset runs/npu/image-data/demo_source/pool.jsonl   --images-dir runs/npu/image-data/demo_source/images   --features runs/npu/real-dflash/features --steps 10   --run-dir runs/npu/real-dspark --set capture.store_final_hidden=true
vlm-dflash eval runs/npu/real-dspark --num-prompts 4 --max-new-tokens 16
vlm-dflash export runs/npu/real-dspark
```

Use each method's NPU preset for the remaining methods. The
[executed command records](evidence/2026-10-20/commands.json) and full effective
configs in `real_runs.json` preserve device choices, limits and overrides.
Evidence sanitization replaces local absolute path prefixes only; numeric
results are unchanged. H0-H3 text logs and training JSONL histories accompany
these JSON files. [Deployed source hashes](evidence/2026-10-20/source_hashes.json)
were checked against all 93 corresponding local Python files.

Local validation: **738 passed, 6 model-dependent skips** in the full suite;
**137 passed without skips** in the fail-closed NPU CPU suite. Wheel build,
packaged resources and documentation are checked separately.
[Machine-readable CPU results](evidence/2026-10-20/cpu_validation.json).

Not validated: full-weight Qwen3.5/3.6-MoE, CUDA hardware regression in this
session, multi-node/DDP/HCCL, production-scale convergence and matched
GPU/NPU performance. The hybrid-MoE speculative-evaluation guard remains.
