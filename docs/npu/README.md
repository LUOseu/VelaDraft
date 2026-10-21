# NPU Training Support

Status on 2026-10-20:

| Axis | State |
|---|---|
| Implementation | COMPLETE |
| CPU evidence | CPU_VERIFIED |
| NPU evidence | NPU_SMOKE_PASSED on two Ascend 910B3 devices |

The seven registered drafters train through the same factory, trainer, loss,
AdamW/FP32-master path, resume and export code on CPU, CUDA and Ascend NPU.
All seven methods have real Qwen2.5-VL-3B BF16 offline training, token-exact
evaluation and export evidence. DFlash also has online, teacher/drafter split,
target device-map and 200-step evidence. These are small engineering validation
runs, not production convergence or speedup benchmarks.

See [hardware results](HARDWARE_RESULTS.md) for exact scopes and raw JSON,
and [installation](INSTALL.md) for the tested CANN/PyTorch/TorchNPU stack.

## Install And Preflight

CPU/CUDA installs do not require `torch_npu`, CANN, FLA or flash-attn:

```bash
python -m pip install -e ".[tested,dev]"
```

For an NPU host, first source the CANN environment and install the TorchNPU
wheel that matches the installed PyTorch, CANN and Python versions. Version
candidates and the upstream source are in `docs/npu/COMPATIBILITY.md`. Keep
`torch_npu` optional; it is intentionally not a core dependency.

```bash
vlm-dflash doctor --device npu:0 --json
```

An explicit `npu:0` request fails with an actionable error when the package,
CANN libraries or device are missing. It never falls back to CPU or CUDA.

## Evidence Commands

The CPU gate is stricter than a plain pytest run: zero tests, skips, failures,
missing methods, fewer than 10 optimizer steps, a NaN loss or a missing
resume/export case all fail closed.

```bash
python scripts/npu/run_cpu_suite.py --json runs/npu/cpu_suite.json
```

The staged NPU gate is opt-in by phase. `H0`-`H2` are the default bring-up
set; `H3` is a long 200-step continuity run; `H4` needs real extended-model and
multi-device resources and is reported as NOT_RUN rather than fabricated.

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H0,H1,H2 --work-dir runs/npu/smoke
```

The runner writes `runs/npu/smoke/manifest.json`, one log and JSON per phase,
and a final `runs/npu/smoke/run_npu_smoke.json`. Without an NPU it exits 2 and
writes `NPU_UNVERIFIED`; it does not execute a CPU substitute.

Phase details and first-device commands: `docs/npu/HARDWARE_BRINGUP.md`.

## Presets

Packaged presets (available after a wheel install through `--preset`):

- `npu-dflash-smoke`, `npu-dspark-smoke`, `npu-domino-smoke`,
  `npu-jetspec-smoke`, `npu-dflash2-smoke`, `npu-dflare-smoke`,
  `npu-dfly-smoke`: one tiny offline recipe per method.
- `npu-online-smoke`: DFlash with live frozen-target extraction.
- `npu-split-device`: teacher on `npu:0`, drafter on `npu:1` (model
  placement, not DDP).
- `npu-target-device-map`: Accelerate placement of Qwen2.5-VL's visual tower
  and language model across two NPUs, validated in a short online run.

All presets use the real schema (`train:`, `target:`, `drafter:`) and
`drafter.attention_backend: sdpa_chunked`. They assume local inputs from
`VLM_DFLASH_TARGET`, `VLM_DFLASH_NPU_DATA` and `VLM_DFLASH_NPU_IMAGES`; no data
or model is downloaded automatically.

## Current Boundaries

- `sdpa` and `sdpa_chunked` are the supported reference attention paths.
  `flex_attention` remains CUDA-only. `npu_fusion` is an opt-in candidate with
  explicit Ascend mask inversion and per-chunk fallback. Its operator
  forward/backward parity passed; full fused training and performance are not
  claimed.
- Qwen3.5/3.6-MoE loads its tracked PyTorch Gated DeltaNet reference path on
  NPU. It is a functional candidate, not a measured production path. The
  35B-A3B target still requires full-weight memory.
- `target.device_map` refuses CPU/disk offload. The tested placement is the
  Qwen2.5-VL preset, not every possible architecture or module map.
- DDP is not implemented in the training trunk. This work does not claim
  multi-NPU data parallelism or HCCL validation.
- Portability: exports record the serving backend (`sdpa`) separately from the
  training backend and load without importing `torch_npu`.
- H2/H3 enable deterministic operators to check **bitwise** resume, including
  DSpark/DFlash2 lookup gradients. Ordinary CLI training keeps PyTorch's default
  determinism policy; its BF16 checkpoints are resumable, but these tests do not
  establish bitwise continuation under nondeterministic kernels.

## Documents

- `docs/npu/COMPATIBILITY.md`: candidate software stack, operator and model limits.
- `docs/npu/STATUS.md`: implementation checklist and remaining work.
- `docs/npu/VALIDATION.md`: commands actually run and measured CPU errors.
- `docs/npu/HARDWARE_BRINGUP.md`: H0-H4 acceptance procedure.
- `docs/npu/support_matrix.json`: machine-readable method/adapter/stage states.
