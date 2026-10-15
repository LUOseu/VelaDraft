> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it verbatim, including the authoring host's local target path). Target-support states inside the archive predate the v0.3.1 `reduced_validation` retag; current machine-readable states: [`SUPPORT_MATRIX.md`](../../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | dflash (block drafter) |
| algorithm status | architecture: stable, multimodal: validated, runtime: experimental, training: validated |
| target support | qwen25_vl: validated, qwen3_vl: validated |
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
| block CE (first10 → last10) | 7.4984177215189876 → 3.7852056962025316 |
| GPU | recorded in RELEASE_VALIDATION.md |
| peak VRAM / wall | 29.546386944 GB / 585.9702818393707 s |

## Quality (greedy, exactness-checked)

- token-exact vs target-only: **True** over 128 prompts
- `ckpt_final.pt`: AL 1.6298365840801265, acceptance rate 0.09146679819964239, per-slot [0.4126126126126126, 0.14905640260155667, 0.04844068159897117, 0.014777262431237191, 0.004128639721860061, 0.0014196789341487386, 0.0003297065611605671] (trained drafter)
- `ckpt_init.pt`: AL 1.0029844936092909, acceptance rate 0.0, per-slot [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (acceptance baseline (random-init drafter))

## Notes

Upstream references:

- algorithm_doc: docs/algorithms/dflash.md
- port_notes: docs/REFERENCE.md
- upstream_reference: jacky566/Automodel_lao@d21fb5aa (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Acceptance numbers depend on the exact target revision and processor recorded above.
- Exported for evaluation/training interop; serving integration is separate roadmap.
