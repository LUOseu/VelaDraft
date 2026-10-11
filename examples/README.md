# Examples

The canonical examples live in the top-level `scripts/` directory:

- `bash scripts/quickstart.sh --config configs/smoke_coco.yaml --limit 64` -
  fastest end-to-end pipeline smoke.
- `bash scripts/quickstart.sh --config configs/qwen2_5_vl_3b.yaml --limit 512` -
  the reference long-answer run (see README for measured numbers).

Programmatic (library) examples are roadmap for v0.2; v0.1 intentionally keeps
the supported surface to the five stage scripts.
