# Upstream audit — JetSpec (pending implementation)

Status: **audit pinned, implementation not started** (Stage 4 of the GOAL
execution order).

```text
Paper:                     JetSpec (citation to be confirmed from the upstream docs
                           during the full audit)
Official repo:             to confirm
Reference implementations: NVIDIA-NeMo/Automodel
                             `nemo_automodel/components/speculative/dflash/jetspec_core.py`
                             `examples/speculative/jetspec/qwen3_jetspec.yaml`
Commit inspected:          NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace
License:                   Apache-2.0
```

Known shape from the pinned sources (to be re-verified before implementation):

```text
Backbone:                  DFlash block backbone with a *causal* in-block attention
                           mask (position i only attends to positions <= i), i.e.
                           JetSpec changes the attention semantics rather than
                           adding a head
Extra modules:             none (mask change); possibly proposal-tree semantics
Training objective:        block CE + forward-KL distillation term (confirm exact
                           form and weights in `jetspec_core.py`)
Inference procedure:       causal in-block sampling semantics (confirm from source)
Checkpoint format:         upstream NeMo; ours follows the unified container
```

The local trainer already accepts a `causal` flag when building the block
attention mask (`_prepare_block_inputs(..., causal=True)`), which is what Stage 4
must validate against the upstream mask construction.
