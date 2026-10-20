# Ascend hardware validation plan

Date: 2026-10-20.

## Design

Build on the existing backend-neutral NPU implementation and its seven registered
trainers. Keep shared model, loss, feature, checkpoint and export contracts.
Use eager PyTorch and query-chunked SDPA as the baseline; enable fused attention
only after forward/backward checks. Global CUDA-to-NPU monkeypatching would hide
unsupported paths; rewriting trainers separately would duplicate objectives and
checkpoint semantics. Neither is needed for this migration.

The allocated host has two idle Ascend 910B3 devices, each with 64 GiB HBM,
an aarch64 CPU, driver 26.0.rc1 and an old CANN 8.0.RC2 image. Install an isolated
Python environment and CANN toolkit/operator packages under a project-specific
directory. Keep the host driver and existing environments intact.

## Execution and acceptance

- [x] Inspect source, branch ancestry, runtime, presets and existing acceptance
  scripts. Preserve the three existing NPU commits and integrate current main.
- [x] Run the local NPU-focused regression baseline: 48 tests pass.
- [x] Install matching torch/TorchNPU 2.10.0 and CANN 9.0.0. Record exact
  versions, visibility and real tensor operations on both devices.
- [x] Run `scripts/npu/run_npu_smoke.py --device npu:0 --phases H0,H1,H2`
  and repeat basic operator checks on `npu:1`. Preserve failures before fixing
  root causes; add regression coverage for each substantive code correction.
- [x] Run seven real method updates, checkpoint continuation, portable export
  and reload using `scripts/npu/seven_method_smoke.py`.
- [x] Run H3 continuity (200 steps) and measure finite losses, memory and resume.
- [x] Prepare local target weights and image/text data, then execute real
  regeneration, feature capture, offline and online training, evaluation and
  export through the CLI. Check teacher/drafter placement across both devices.
- [x] Exercise optional fused attention and extended targets where resources
  permit. Report unexecuted combinations explicitly. Split placement is not DDP.
- [x] Run CPU regressions and documentation/package checks. Attach compact,
  sanitized hardware evidence and reproducible environment/launch instructions.
Delivery: review the final diff and open a PR against `LUOseu/VelaDraft:main`
with measured results and remaining limitations.

Success claims must identify device, model, method, dtype, data and run length.
Synthetic feature tests establish execution correctness, not model convergence
or real-teacher coverage. Small image pipelines establish end-to-end plumbing,
not production training quality. Credentials and host-identifying details must
not appear in committed artifacts.
