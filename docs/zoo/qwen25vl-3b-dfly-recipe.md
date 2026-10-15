> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | dfly (block drafter) |
| algorithm status | architecture: validated, multimodal: validated, runtime: experimental, training: experimental |
| target support | qwen25_vl: validated, qwen3_vl: reduced_validation |
| target | Qwen2.5-VL-3B-Instruct (qwen2_5_vl; path redacted here) |
| target fingerprint | `e044f48898fec43d…` |
| adapter | qwen2_5_vl |
| block size | 8 |
| features | offline (pre-captured feature store) |
| anchors / layers | 256 / 3 |
| trainable parameters | 264786697 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [16, 24, 24] (replace) |
| code commit | `6ad8500` |
| objective profile | upstream_two_stage |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 4.358583860759493 → 1.7792721518987342 |
| GPU | NVIDIA RTX A6000 |
| peak VRAM / wall | 46.529032704 GB / 2646.2188243865967 s |

## Quality (greedy, exactness-checked)

- token-exact vs target-only: **True** over 128 prompts
- `ckpt_final.pt`: AL 1.8392623438429507, acceptance rate 0.12225568884168742, per-slot [0.4396314466913964, 0.2074653822998194, 0.10314769975786925, 0.050572751645137706, 0.023564064801178203, 0.01345014807502468, 0.007074593521161723] (trained drafter)
- `ckpt_init.pt`: AL 1.0029844936092909, acceptance rate 0.0, per-slot [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (acceptance baseline (random-init drafter))

## Notes

Trained with this repo's port of upstream AngelSpec's two-stage recipe (objective_profile=upstream_two_stage: LK / D-PACE cold start -> end-to-end TV; see docs/upstream/dfly.md). At inference the block backbone runs in parallel, then the hidden-states correction and the LM-head projection run per slot sequentially, conditioned on the previously proposed tokens -- the documented local adaptation of upstream's train-time-only correction transformation.

This is the recipe variant of the `zoo-v0.3.1` DFly archive (same architecture,
different `train_recipe`: `upstream_two_stage` instead of the shared decayed
block CE); see `BENCHMARKS.md` for the controlled comparison.

Upstream references:

- algorithm_doc: docs/algorithms/dfly.md
- paper: arXiv:2607.25852 (AngelSpec / DFly)
- port_notes: docs/upstream/dfly.md
- upstream_reference: Tencent/AngelSpec@d3412bed231025ea85d35dbf5e0af44f1ac62a5b angelspec/models/draft/dfly.py (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Acceptance numbers depend on the exact target revision and processor recorded above.
- Exported for evaluation/training interop; serving integration is separate roadmap.
