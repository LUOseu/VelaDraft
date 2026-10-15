> Copy of the released export's generated `model_card.md` (the downloaded archive keeps it verbatim, including the authoring host's local target path). Target-support states inside the archive predate the v0.3.1 `reduced_validation` retag; current machine-readable states: [`SUPPORT_MATRIX.md`](../../SUPPORT_MATRIX.md).

# vlm-dflash drafter — Qwen2.5-VL-3B-Instruct

| field | value |
| --- | --- |
| method | domino (block drafter) |
| algorithm status | architecture: validated, multimodal: validated, runtime: experimental, training: validated |
| target support | qwen25_vl: validated, qwen3_vl: experimental |
| target | Qwen2.5-VL-3B-Instruct (qwen2_5_vl; path redacted here) |
| target fingerprint | `e044f48898fec43d…` |
| adapter | qwen2_5_vl |
| block size | 8 |
| anchors / layers | 256 / 3 |
| trainable parameters | 292930304 |
| mask token | 151669 |
| routed target layers | [1, 17, 33] |
| MRoPE sections | [16, 24, 24] (replace) |
| code commit | `9f4d56f` |
| training data | 2000 train / 128 val from `llava_instruct` |
| data digest | `5dfc5d7e9aa52c27…` |
| training steps | 792 |
| block CE (first10 → last10) | 7.2515822784810124 → 2.199367088607595 |
| GPU | recorded in RELEASE_VALIDATION.md |
| peak VRAM / wall | 38.367236608 GB / 775.3165581226349 s |

## Quality (greedy, exactness-checked)

- token-exact vs target-only: **True** over 128 prompts
- `ckpt_final.pt`: AL 1.763719338277239, acceptance rate 0.111153788346888, per-slot [0.442570281124498, 0.18027485852869846, 0.07802159526297457, 0.036223416686141624, 0.016362566215420836, 0.008762581409117821, 0.005121486422105765] (trained drafter)
- `ckpt_init.pt`: AL 1.0029844936092909, acceptance rate 0.0, per-slot [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (acceptance baseline (random-init drafter))

## Notes

Upstream references:

- algorithm_doc: docs/algorithms/domino.md
- port_notes: docs/upstream/domino.md
- upstream_reference: SpecForge#571 via NVIDIA-NeMo/Automodel (Apache-2.0)

## Scope / limitations

- Single-image image-text prompts; greedy decoding only.
- Acceptance numbers depend on the exact target revision and processor recorded above.
- Exported for evaluation/training interop; serving integration is separate roadmap.
