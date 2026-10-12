# Upstream audit — Domino (pending implementation)

Status: **audit pinned, implementation not started** (Stage 3 of the GOAL
execution order). Sources were downloaded and pinned so the audit can be
completed before any code is written; no Domino code exists in this repo yet.

```text
Paper:                     Domino speculative decoding (block drafter with a causal
                           low-rank correction; citation to be confirmed from the
                           upstream docs during the full audit)
Official repo:             to confirm
Reference implementations: NVIDIA-NeMo/Automodel
                             `nemo_automodel/components/speculative/dflash/domino_core.py`
                             `examples/speculative/dflash/qwen3_domino.yaml`
                           jacky566/Automodel_lao @ d21fb5aa also carries Domino benches
                           (read-only reference; bench JSONs only)
Commit inspected:          NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace
License:                   Apache-2.0
```

Known shape from the pinned sources (to be re-verified line by line when the
audit is completed, before implementation):

```text
Backbone:                  DFlash block backbone; Domino adds a causal in-block
                           correction head driven by a GRU over the block's previous
                           tokens (the trainer seam already supports the shifted label
                           window: `_build_block_targets(..., label_start=1)`)
Extra modules:             GRU (prefix_gru) + low-rank correction projection
                           (`embed_proj`), plus a base-anchor curriculum
                           (`lambda_base` ramp), per the example config
Training objective:        DFlash decayed CE (+ correction term; confirm exact
                           weights in `domino_core.py`)
Inference procedure:       block-parallel backbone + per-slot correction chained
                           over the block (the local draft model already carries
                           `projector_type="domino"` hooks from the frozen port)
Checkpoint format:         upstream NeMo checkpoints; ours will follow the unified
                           vlm-dflash container with `algorithm.json` semantics
```

The local `vlm_dflash/dflash/draft_model.py` already contains the frozen port's
Domino hooks (`projector_type`, `shift_label`, `pure_draft_prefix_len`,
`set_domino_training_stage`), which is why Domino is Stage 3: it validates that
the algorithm seam supports causal correction heads.
