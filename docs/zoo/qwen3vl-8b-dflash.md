> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen3-VL-8B-Instruct

| field | value |
| --- | --- |
| method | dflash (block drafter) |
| algorithm status | architecture: stable, multimodal: validated, runtime: experimental, training: validated |
| target support | qwen25_vl: validated, qwen3_vl: validated |
| target | Qwen3-VL-8B-Instruct (qwen3_vl; path redacted here) |
| target fingerprint | `3925a38557fb5f5f…` |
| adapter | qwen3_vl |
| block size | 8 |
| anchors / layers | 256 / 3 |
| trainable parameters | 629179136 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [24, 20, 20] (replace) |
| code commit | `9f4d56f` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 1500 |
| block CE (first10 → last10) | 7.660625 → 2.5214583333333334 |
| GPU | recorded in RELEASE_VALIDATION.md |
| peak VRAM / wall | 27.165554688 GB / 3311.1519384384155 s |

## Notes

Upstream references:

- algorithm_doc: docs/algorithms/dflash.md
- port_notes: docs/REFERENCE.md
- upstream_reference: jacky566/Automodel_lao@d21fb5aa (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Exported for evaluation/training interop; serving integration is separate roadmap.
