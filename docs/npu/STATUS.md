# NPU Status

Updated: 2026-10-19. Branch: `feature/npu-training-preparation`.

# Implementation And Evidence

| Axis | State | Meaning |
|---|---|---|
| Implementation | COMPLETE | Device layer, attention fallback, seven methods, teacher data paths, resume/export and acceptance scripts are wired. |
| CPU evidence | CPU_VERIFIED | The CPU suite, seven-method synthetic training, exact resume and export round-trip executed locally. |
| NPU evidence | NPU_UNVERIFIED | No Ascend device was present. H0-H4 have not run on hardware. |


## Completed

- Device layer: CPU/CUDA/NPU resolution, explicit `torch_npu` initialization,
  actionable missing-device errors, synchronization, peak/current memory,
  device names, environment reports, accelerator timing and RNG save/restore.
- Attention: preserved dense mask semantics; added query-chunked SDPA with
  activation checkpointing and an opt-in `npu_fusion` candidate with explicit
  mask inversion and fallback reasons.
- Seven methods: DFlash, DSpark, Domino, JetSpec, DFlash2, DFlare and DFly use
  their real factory, trainer, loss and optimizer in the CPU suite. Each method
  has 10 real updates, method-specific nonzero gradients, a 6+6 exact-resume
  comparison against 12 continuous steps, and export/reload parity.
- Teacher data chain: offline feature stores, online live extraction,
  device-aware target routing, cache/regroup paths, Qwen3.5/3.6 PyTorch
  reference fallback and device-map offload refusal.
- State and export: NPU RNG in checkpoints, backend provenance, portable
  serving backend, and export loading without `torch_npu`.
- Acceptance package: CPU suite, operator smoke, seven-method smoke, staged
  runner, packaged presets, support matrix and bring-up documentation.

## Remaining Work

1. Run H0-H2 on the school Ascend host and attach the generated logs/JSON.
2. If the build exposes `npu_fusion_attention`, rerun H1 with
   `"include_fusion": true` and record forward/backward errors.
3. Run H3 for 200 steps with checkpoint/resume and memory sampling.
4. Run H4 for Qwen3-VL, SmolVLM and Qwen3.6-35B-A3B; test device maps and, if
   desired, implement DDP/HCCL separately.
5. Only after the corresponding evidence exists, change the affected NPU rows
   from NPU_UNVERIFIED to NPU_SMOKE_PASSED or NPU_TRAINING_VALIDATED.

The next command and phase acceptance criteria are in
`docs/npu/HARDWARE_BRINGUP.md`.
