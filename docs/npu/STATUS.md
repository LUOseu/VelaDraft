# NPU status

Updated: 2026-10-20. Hardware: two Ascend 910B3 devices.

| Scope | Evidence |
|---|---|
| Runtime, RNG, memory, SDPA/GQA/BF16 and draft gradients | NPU_SMOKE_PASSED |
| Seven methods: synthetic updates, deterministic resume, export/reload | NPU_SMOKE_PASSED |
| Qwen2.5-VL-3B: seven methods, 10 BF16 offline steps each | NPU_SMOKE_PASSED |
| Seven methods: four-prompt sequential token-exact eval, init and trained checkpoints | NPU_SMOKE_PASSED |
| DFlash: 200 synthetic steps with exact midpoint resume and memory samples | NPU_SMOKE_PASSED |
| DFlash: 200 real-target feature steps | NPU_SMOKE_PASSED |
| DFlash: online, cross-device teacher/drafter, sharded target | NPU_SMOKE_PASSED |
| Optional fused-attention operator forward/backward | NPU_SMOKE_PASSED |

Exact model/method/mode scopes, extended-model results, commands and measurements
are in [HARDWARE_RESULTS.md](HARDWARE_RESULTS.md). Machine-readable evidence is
linked from [support_matrix.json](support_matrix.json).

Remaining boundaries:

- No DDP launcher or multi-node training; two-device placement is not data
  parallelism. No HCCL collective validation is claimed.
- Qwen3.5/3.6-MoE has a PyTorch reference adaptation and CPU coverage, but no
  full-weight 35B-A3B NPU run. Its pre-existing speculative-evaluation guard
  remains because recurrent-cache rollback is unsupported.
- No CUDA hardware regression ran in this session.
- No production-scale convergence, matched GPU/NPU performance comparison, or
  end-to-end fused-attention training benchmark.
- Bitwise resume was checked with deterministic operators in the acceptance
  harness. Default CLI BF16 training is not claimed bitwise deterministic.
