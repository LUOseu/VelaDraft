> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | dflare (block drafter) |
| algorithm status | architecture: validated, multimodal: validated, runtime: experimental, training: validated |
| target support | qwen25_vl: validated, qwen3_vl: reduced_validation |
| target | Qwen2.5-VL-3B-Instruct (qwen2_5_vl; path redacted here) |
| target fingerprint | `e044f48898fec43d…` |
| adapter | qwen2_5_vl |
| block size | 8 |
| features | offline (pre-captured feature store) |
| anchors / layers | 256 / 3 |
| trainable parameters | 234373897 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [16, 24, 24] (replace) |
| code commit | `af61557` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 7.3888449367088604 → 3.7349683544303796 |
| GPU | NVIDIA RTX A6000 |
| peak VRAM / wall | 30.2482944 GB / 1444.4913008213043 s |

## Verification

- token-exact vs target-only: **True** over 128 prompts

## Notes

Upstream references:

- algorithm_doc: docs/algorithms/dflare.md
- paper: arXiv:2606.02091 (DFlare)
- port_notes: docs/upstream/dflare.md
- upstream_reference: Tencent/AngelSpec@d3412bed231025ea85d35dbf5e0af44f1ac62a5b angelspec/models/draft/dflare.py (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Exported for evaluation/training interop; serving integration is separate roadmap.
