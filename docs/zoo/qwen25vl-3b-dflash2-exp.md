> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | dflash2 (block drafter) |
| algorithm status | architecture: validated, multimodal: validated, runtime: experimental, training: experimental |
| target support | qwen25_vl: experimental, qwen3_vl: untested |
| target | Qwen2.5-VL-3B-Instruct (qwen2_5_vl; path redacted here) |
| target fingerprint | `e044f48898fec43d…` |
| adapter | qwen2_5_vl |
| block size | 8 |
| anchors / layers | 256 / 3 |
| trainable parameters | 328467200 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [16, 24, 24] (replace) |
| code commit | `9f4d56f` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 5.611857993693292 → 1.4780075203014325 |
| GPU | recorded in RELEASE_VALIDATION.md |
| peak VRAM / wall | 31.11320832 GB / 1359.0401673316956 s |

## Verification

- token-exact vs target-only: **True** over 128 prompts

## Notes

Experimental DFlash2 port: the conv+selector architecture and checkpoint key names follow z-lab/dflash via vllm-project/speculators, but upstream does not publish its training objective, so this checkpoint was trained with speculators' experimental split objective (unary forward-KL + K-way selector CE) and served by this repo's block decoder (one parallel draft forward + a sequential top-K candidate walk).

Upstream references:

- algorithm_doc: docs/algorithms/dflash2.md
- paper: no paper; Z Lab DFlash2 blog (https://inco.ai/blog/dflash2/)
- port_notes: docs/upstream/dflash2.md
- upstream_reference: z-lab/dflash@07ebd93d via vllm-project/speculators@55ebbd5b (MIT model code / Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Exported for evaluation/training interop; serving integration is separate roadmap.
