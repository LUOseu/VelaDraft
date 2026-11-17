# DFlash source attribution

The original DFlash core was adapted from NVIDIA NeMo Automodel / SpecForge,
via the Apache-2.0 reference `jacky566/Automodel_lao` at commit `d21fb5aa`.
The reference repository is not a runtime dependency.

| Local module | Original reference module |
| --- | --- |
| `vlm_dflash/dflash/draft_model.py` | `nemo_automodel/components/speculative/dflash/draft_qwen3.py` |
| `vlm_dflash/dflash/trainer_core.py` | `nemo_automodel/components/speculative/dflash/core.py` |
| `vlm_dflash/dflash/mask.py` | `nemo_automodel/components/attention/dflash_mask.py` |
| `vlm_dflash/dflash/decay_loss.py` | `nemo_automodel/components/loss/dllm_loss.py` |

VelaDraft adds target contracts, multimodal position handling, feature reuse,
batch execution, objective-specific scoring, and native deployment adapters.
See [the implementation guide](PAPER_GUIDE.md) for the current code and
[NOTICE](../NOTICE) for third-party attribution. Notes under `upstream/` record
architecture-port audits; the current source and selected recipe define behavior.
