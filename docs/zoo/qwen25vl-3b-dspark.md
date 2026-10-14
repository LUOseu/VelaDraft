> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | dspark (block drafter) |
| algorithm status | architecture: validated, multimodal: validated, runtime: experimental, training: validated |
| target support | qwen25_vl: validated, qwen3_vl: experimental |
| target | Qwen2.5-VL-3B-Instruct (qwen2_5_vl; path redacted here) |
| target fingerprint | `e044f48898fec43d…` |
| adapter | qwen2_5_vl |
| block size | 8 |
| anchors / layers | 256 / 3 |
| trainable parameters | 321604609 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [16, 24, 24] (replace) |
| code commit | `9f4d56f` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 7.589003164556962 → 3.4450158227848102 |
| GPU | recorded in RELEASE_VALIDATION.md |
| peak VRAM / wall | 38.089121792 GB / 709.5355050563812 s |

## Verification

- token-exact vs target-only: **True** over 128 prompts

## Notes

Upstream references:

- algorithm_doc: docs/algorithms/dspark.md
- port_notes: docs/upstream/dspark.md
- upstream_reference: vllm-project/speculators@55ebbd5b, NVIDIA-NeMo/Automodel@bb5b34ce (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Exported for evaluation/training interop; serving integration is separate roadmap.
