# Validation Record

Date: 2026-10-19. Host: macOS arm64, no CUDA and no Ascend NPU. Python 3.12.13
(`.venv`), torch 2.10.0, torchvision 0.25.0, transformers 5.12.1,
accelerate 1.15.0. Base repository SHA:
`832edea25dd88ecbda03de96496d140e9fcca0a4`.

## Historical CPU preparation results (2026-10-19)

| Check | Command | Result |
|---|---|---|
| Full CPU suite | `pytest tests/ -q` | 719 passed, 6 skipped, 0 failed |
| NPU CPU suite | `python scripts/npu/run_cpu_suite.py --json runs/npu/cpu_suite.json` | CPU_VERIFIED, 132 tests, no skips/failures |
| Seven-method reference | `python scripts/npu/seven_method_smoke.py --device cpu --allow-cpu --steps 10 --out runs/npu/seven_methods_cpu_ref.json` | CPU_REFERENCE, 7/7 method records passed |
| Operator smoke without NPU | `python scripts/npu/operator_smoke.py --h1` | exit 2, NPU_UNVERIFIED, no CPU fallback |
| Staged runner without NPU | `python scripts/npu/run_npu_smoke.py --work-dir runs/npu/smoke` | exit 2, NPU_UNVERIFIED, manifest written |
| Doctor without NPU | `vlm-dflash doctor --device npu:0 --json` | exit 2, selected NPU unavailable, actionable import error |
| Documentation | `python scripts/check_docs.py` | 184 links, 375 paths, 80 CLI flags; docs OK |
| Build | `python -m build` | wheel and sdist built |
| Fresh wheel install | install wheel into a new venv; run CLI and load presets | CLI help OK, 10 NPU presets load, `torch_npu` absent from `sys.modules` |

The six full-suite skips are existing real-checkpoint contract/integration
cases gated by `VLM_DFLASH_TEST_MODEL`; they are not NPU tests and are not
counted as evidence.

The seven-method reference run executed the real registered factory, trainer,
loss and AdamW optimizer for every method with 10 optimizer steps, a 5+5 exact
resume, and an export/reload forward comparison. It used synthetic target
features and therefore does not validate a real Qwen teacher forward.

## CPU Numerical Parity

Measured from the same inputs and weights used by the attention tests:

| Comparison | Maximum absolute error |
|---|---|
| Dense additive mask vs concatenated chunk masks | 0 (exact tensor equality) |
| Chunked SDPA forward vs dense SDPA | 2.38e-07 |
| Q gradient, chunked vs dense | 4.66e-10 |
| K gradient, chunked vs dense | 9.31e-10 |
| V gradient, chunked vs dense | 9.31e-10 |
| Whole trainer loss, chunked vs dense | 0.0 |
| Whole trainer parameter gradients (25 compared) | 1.19e-07 |

The tests assert `rtol=1e-4, atol=1e-5` for SDPA/module parity. Mask equality
is exact, including padding blocks, document isolation, causal JetSpec blocks
and bidirectional DFlash blocks. `tests/test_npu_reporting.py` separately
proves the fail-closed aggregate rejects empty records, missing methods, fewer
than 10 steps, nonzero exits, skipped cases, all-skip runs, NaNs and missing
NaN evidence.

## Seven-Method CPU Evidence

Every method has:

- 10 real optimizer steps with finite loss and parameter updates;
- a method-specific nonzero-gradient check (DFlash2 conv/selector, DSpark
  heads, Domino GRU/projection, DFlare/DFly fusion and correction, JetSpec KD
  path, DFlash block path);
- 12 continuous steps versus 6 + checkpoint + 6 resumed steps with exact model,
  master, optimizer and scheduler state equality;
- export to the self-describing portable format, reload without `torch_npu`,
  and an identical small forward.

## Real hardware evidence

H0-H3, seven-method real-target BF16 training/evaluation/export, online and
multi-device placement have now executed. See [HARDWARE_RESULTS.md](HARDWARE_RESULTS.md)
for the measured scope and raw JSON. The CPU results above remain a separate
reference, not a substitute for those hardware runs.

Not validated: full-weight Qwen3.5/3.6-MoE, CUDA hardware regression in this
session, HCCL/DDP, full fused-attention training, production convergence or
matched GPU/NPU performance. H4 in the generic runner remains a command template
and returns a nonpassing result when requested; separately executed real-model
and placement runs have their own evidence.
