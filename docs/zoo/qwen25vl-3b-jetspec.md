> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | jetspec (block drafter) |
| algorithm status | architecture: validated, multimodal: validated, runtime: experimental, training: validated |
| target support | qwen25_vl: validated, qwen3_vl: untested |
| target | Qwen2.5-VL-3B-Instruct (qwen2_5_vl; path redacted here) |
| target fingerprint | `e044f48898fec43d…` |
| adapter | qwen2_5_vl |
| block size | 8 |
| anchors / layers | 256 / 3 |
| trainable parameters | 243811072 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [16, 24, 24] (replace) |
| code commit | `9f4d56f` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 6.061732829371585 → 3.2565143772318392 |
| GPU | recorded in RELEASE_VALIDATION.md |
| peak VRAM / wall | 29.654835712 GB / 1233.612478017807 s |

## Notes

Trained with a causal in-block mask and forward-KL distillation. The paper's parallel-tree runtime is not ported: the shared block-parallel decoder serves this checkpoint with the same causal in-block mask (per-slot greedy proposals, no tree conditioning), which is the closest factorization to training this repo supports.

Upstream references:

- algorithm_doc: docs/algorithms/jetspec.md
- port_notes: docs/upstream/jetspec.md
- upstream_reference: NVIDIA-NeMo/Automodel (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Exported for evaluation/training interop; serving integration is separate roadmap.
