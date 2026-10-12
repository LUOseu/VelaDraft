# Upstream audit — DFlash2 (pending implementation)

Status: **audit pinned, implementation not started** (Stage 5 of the GOAL
execution order — the most complex variant, training objective still
experimental upstream).

```text
Paper:                     DFlash2 (citation to be confirmed; upstream docs marked
                           experimental)
Official repo:             to confirm
Reference implementations: vllm-project/speculators
                             `src/speculators/models/dflash2/{core,config,metrics,model_definitions}.py`
                             `docs/user_guide/algorithms/dflash2.md`
                           NVIDIA-NeMo/Automodel
                             `nemo_automodel/components/speculative/dflash/dflash2_core.py`
                             `nemo_automodel/components/speculative/dflash/draft_qwen3_dflash2.py`
                             `examples/speculative/dflash/qwen3_dflash2.yaml`
Commit inspected:          speculators @ 55ebbd5b9b49e9bd223f47949c986aa4abfffbc3
                           NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace
License:                   Apache-2.0
```

Known shape from the pinned sources (to be re-verified before implementation):

```text
Backbone:                  DFlash backbone + local/dynamic convolution over block
                           hidden states + a candidate-path selector
Extra modules:             conv module + path selector head
Training objective:        upstream objective is experimental / partially undisclosed;
                           will be reconstructed from the pinned sources and marked
                           Experimental in `ALGORITHM_MATRIX.md` + `SUPPORT_MATRIX.md`
Inference procedure:       selector requires the *full target vocabulary* per slot
                           (documented requirement; the `doctor` check for DFlash2 is
                           "Full vocab head required OK/FAIL")
Checkpoint format:         upstream NeMo/speculators; ours follows the unified container
```

No DFlash2 code exists in this repo yet; this file exists so the audit trail
starts from pinned sources rather than from memory.
