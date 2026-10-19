# Compatibility And Boundaries

Last source check: 2026-10-19. No real Ascend device was available for this
checkout, so every NPU row below is a candidate, not a tested combination.

## Candidate Software Stacks

TorchNPU product releases and pip package versions are different namespaces.
Use the pip version, not the product marketing version, in `pip install`.

| Upstream product | PyTorch | `torch_npu` pip package | CANN | Python | State |
|---|---|---|---|---|---|
| TorchNPU 2.12.0 | 2.12.0 | matching 2.12.0 wheel | 9.1.0 | 3.10-3.14 | Candidate, recommended by upstream |
| TorchNPU 26.1.1 | 2.10.0 | `2.10.0.post6` | 9.1.X | check the wheel metadata | Candidate, closest to the repository's tested torch 2.10.0 |
| TorchNPU 26.1.0 | 2.10.0 | `2.10.0.post4` | 9.1.0 | check the wheel metadata | Candidate |

Sources:

- https://github.com/Ascend/pytorch
- https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md
- https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md

Still unknown for the target host: chip model, NPU count, HBM per device,
driver version, CANN patch level, container architecture and image tag. Do not
write a specific 910B/910C or link count into a run report until `npu-smi info`
and a real training run have recorded it.

## Runtime Rules

- `vlm_dflash.runtime` is the only device-resolution layer. Importing
  `vlm_dflash` does not import `torch_npu`.
- Explicit `cuda`/`npu` requests are checked before training writes artifacts.
  A missing backend raises `DeviceResolutionError`; there is no silent
  substitution.
- No global `transfer_to_npu` and no `torch.cuda = torch.npu` monkey patch.
- Timing uses accelerator events when available, otherwise synchronized wall
  clock. Peak/current memory returns null when the API is unavailable, never 0.
- Python, NumPy, torch and accelerator RNG states are saved in checkpoints.
  Old checkpoints without NPU RNG keys remain loadable.
- `pin_memory` stays CUDA-only. `non_blocking` is opt-in on CUDA/NPU.
  `optimizer_fused` stays CUDA-only; CPU/NPU use eager AdamW with FP32 masters.
  `torch.compile` is disabled on NPU; the eager objective path is retained.

## Attention

| Backend | Implementation | Evidence | Notes |
|---|---|---|---|
| `sdpa` | Complete | CPU_VERIFIED | Full additive mask; reference semantics. |
| `sdpa_chunked` | Complete | CPU_VERIFIED | Query-chunked mask construction, full K/V softmax per query, activation checkpointing under grad. |
| `flex_attention` | Complete | Existing CUDA evidence | Not routed on NPU. |
| `npu_fusion` | Complete, opt-in | NPU_UNVERIFIED | `torch_npu.npu_fusion_attention`, `True = masked`, per-chunk fallback when enabled. Never auto-selected. |

The chunked path keeps DFlash context-prefix visibility, document isolation,
padding-block safety, bidirectional DFlash blocks and JetSpec causal blocks.
Its mask is built per query chunk; it never materializes the dense mask first.
The attention memory plan reports dense versus chunked bytes and can reject a
configuration before allocation.

## Target Models

| Family | NPU routing | State |
|---|---|---|
| Qwen2.5-VL | Same adapter, device-neutral capture/online path | NPU_UNVERIFIED; first bring-up target |
| Qwen3-VL | Same adapter, untied head supported | NPU_UNVERIFIED |
| SmolVLM2 | 1-D positions and tile image path | NPU_UNVERIFIED |
| Qwen3.5/3.6-MoE | Forces upstream PyTorch Gated DeltaNet reference on NPU | NPU_UNVERIFIED; functional candidate only |

The Qwen3.5/3.6 adapter leaves `cache_crop_rollback=False` and does not remove
the speculative-inference guard. FLA/Triton fast kernels are not assumed on
NPU. The 35B-A3B checkpoint must be budgeted by full weight size, not by its
active-parameter label.

## Device Placement And Parallelism

- Same-device teacher/drafter is the default.
- `target.device` and `train.device` may differ; online extraction returns
  features on the drafter device while hooks accept per-layer devices.
- `target.device_map` is accepted by the existing loader for regen/capture/
  online. NPU map semantics through Accelerate are a candidate. The loader
  rejects CPU/disk offload; eval rejects a sharded target.
- DDP is not implemented. Teacher/drafter split or `device_map` must not be
  described as data parallelism. HCCL and multi-rank equivalence are NOT_RUN.

## Training State And Export

- BF16 drafter plus FP32 master parameters is preserved.
- Checkpoints record the training backend; a cross-backend resume requires
  `train.allow_cross_backend_resume=true` and is not an exact-parity claim.
- Exports record `drafter.attention_backend: sdpa` for portable serving and
  `drafter.training_attention_backend` for provenance. Loading an export does
  not require `torch_npu`.
- CPU and NPU floating-point results are not expected to be bitwise identical.
