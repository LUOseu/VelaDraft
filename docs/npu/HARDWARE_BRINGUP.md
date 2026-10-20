# NPU Bring-Up Procedure

This document is the first-device checklist. Run the phases in order. A phase
is evidence only when its generated JSON says `NPU_SMOKE_PASSED`; a skipped or
disabled phase is NOT_RUN, not a pass.

## 0. Record The Host

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())"
vlm-dflash doctor --device npu:0 --json > runs/npu/doctor.json
```

Record the chip model, HBM per device, driver, CANN, image tag, Python,
PyTorch, TorchNPU and Transformers versions. The repository does not assume a
chip model or device count.

## H0: Environment And Basic Operators

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H0 --work-dir runs/npu/H0
```

Pass conditions: tensor allocation, matmul forward/backward, synchronization,
NPU RNG save/restore, peak-memory query and SDPA forward/backward all pass; all
checked tensors remain on `npu`; no CPU fallback or silent memory `0` report.

Artifacts: `runs/npu/H0/logs/H0.log` and `runs/npu/H0/phase_json/H0.json`.

## H1: Attention And Draft-Module Numeric Parity

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H1 --work-dir runs/npu/H1
```

Pass conditions: GQA SDPA, BF16 SDPA, chunked-mask equality and a complete
tiny DFlash draft module forward/backward match the CPU reference within the
recorded tolerances. RoPE is exercised through the draft module.

If the build exposes `torch_npu.npu_fusion_attention`, opt into the fused
check with a manifest override:

```bash
cat > runs/npu/h1_fusion_manifest.json <<'JSON'
{"format": "vlm-dflash-npu-smoke-manifest-v1", "phases": {"H1": {"include_fusion": true}}}
JSON
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H1 --manifest runs/npu/h1_fusion_manifest.json --work-dir runs/npu/H1-fusion
```

The fused path must pass forward and backward with the inverted Ascend mask. If
the API or a shape is unsupported, keep `sdpa_chunked` as the effective path
and record the exact limitation; do not claim fused support.

## H2: Seven-Method Short Run

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H2 --work-dir runs/npu/H2
```

Pass conditions for each of the seven registered methods: at least 10 real
optimizer steps, finite losses, method-specific gradients, parameter updates,
exact same-device resume with deterministic algorithms enabled, and export/reload parity. The default H2 body uses
deterministic synthetic target features, so it proves the device training path
but not a real image-text teacher forward.

For the real-image variant, run the preset recipe on a local JSONL corpus and
record a separate run directory:

```bash
VLM_DFLASH_TARGET=/models/Qwen2.5-VL-3B-Instruct \
VLM_DFLASH_NPU_DATA=/data/smoke.jsonl \
VLM_DFLASH_NPU_IMAGES=/data/smoke/images \
vlm-dflash train --preset npu-dflash-smoke --target /models/Qwen2.5-VL-3B-Instruct --dataset /data/smoke.jsonl --images-dir /data/smoke/images --steps 10 --run-dir runs/npu/H2-real-dflash
```

Repeat with `--preset npu-dspark-smoke`, `npu-domino-smoke`,
`npu-jetspec-smoke`, `npu-dflash2-smoke`, `npu-dflare-smoke` and
`npu-dfly-smoke`. Each method/mode gets its own status row.

## H3: Continuity And Memory (Opt-In, Long)

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H3 --h3-steps 200 --timeout 21600 --work-dir runs/npu/H3
```

Pass conditions: 200 finite optimizer steps, a midpoint checkpoint/resume,
no unbounded host or device memory growth, and no unexpected skipped data.
Record throughput but do not compare it to CUDA without matched hardware,
batch, sequence and dtype.

Increase anchors/context/batch only after the small recipe is stable. Keep the
effective configuration in the run JSON; do not silently change anchors or
loss weights to avoid an OOM.

## H4: Extended Models And Multi-Device

H4 is intentionally NOT_RUN in the staged runner unless real resources are
provided and the commands are executed separately. Generate the command list:

```bash
python scripts/npu/run_npu_smoke.py --device npu:0 --phases H4 --target /models/Qwen3-VL-8B-Instruct --dataset /data/smoke.jsonl --images-dir /data/smoke/images --work-dir runs/npu/H4
```

Then execute the generated commands and attach their run directories. Cover
Qwen3-VL, SmolVLM and Qwen3.6-35B-A3B one model × method × mode at a time.
Qwen3.5/3.6 uses the PyTorch Gated DeltaNet reference; it is a functional
candidate and needs full-weight memory accounting.

For teacher/drafter placement, use `npu-split-device`; this is not DDP. For the
validated Qwen2.5-VL placement, use `npu-target-device-map`; its
module-to-device map places the visual tower on `npu:0` and the language model
plus LM head on `npu:1`. DDP and HCCL are not implemented in this branch and
remain NOT_IMPLEMENTED.

## Promoting Evidence

Promote only the rows that actually passed:

1. Attach the phase JSON, log, commit and environment record.
2. Update the method/model/mode row in `docs/npu/support_matrix.json`.
3. Update `docs/npu/STATUS.md` and this document.
4. Use NPU_SMOKE_PASSED for short-run evidence; use
   NPU_TRAINING_VALIDATED only after sufficient long-run convergence and
   throughput evidence.

Never turn a CPU mock, a skipped phase or a synthetic-feature smoke into a real
NPU training claim.
