> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it
> verbatim, including the authoring host's local target path). Target-support fields are
> the values recorded at export time; current machine-readable states:
> [`SUPPORT_MATRIX.md`](../SUPPORT_MATRIX.md).

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
| code commit | `6cf8141` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 6.286787974683544 → 1.6027492088607596 |
| GPU | NVIDIA RTX A6000 |
| peak VRAM / wall | 31.61120256 GB / 965.7326419353485 s |

## Quality (greedy, exactness-checked)

- token-exact vs target-only: **True** over 128 prompts
- `ckpt_final.pt`: AL 1.681971493852682, acceptance rate 0.09910296784044278, per-slot [0.3836980306345733, 0.16013647369579573, 0.07271721084670725, 0.03704527756146401, 0.018707292483477092, 0.008226279017354067, 0.004420265215912955] (trained drafter)
- `ckpt_init.pt`: AL 1.0029844936092909, acceptance rate 0.0, per-slot [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (acceptance baseline (random-init drafter))

## Notes

Trained with the shared DFlash block-CE objective (objective_profile=controlled_ce); upstream AngelSpec's two-stage LK/D-PACE -> end-to-end TV recipe is selectable in this repo via algorithm.train_recipe=upstream_two_stage (see docs/upstream/dfly.md). At inference the block backbone runs in parallel, then the hidden-states correction and the LM-head projection run per slot sequentially, conditioned on the previously proposed tokens -- the documented local adaptation of upstream's train-time-only correction transformation.

Upstream references:

- algorithm_doc: docs/algorithms/dfly.md
- paper: arXiv:2607.25852 (AngelSpec / DFly)
- port_notes: docs/upstream/dfly.md
- upstream_reference: Tencent/AngelSpec@d3412bed231025ea85d35dbf5e0af44f1ac62a5b angelspec/models/draft/dfly.py (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Acceptance numbers depend on the exact target revision and processor recorded above.
- Exported for evaluation/training interop; serving integration is separate roadmap.
