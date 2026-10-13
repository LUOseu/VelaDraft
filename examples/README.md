# Examples

Small, stable examples that import the installed package (no `sys.path`
hacks). They need a local Qwen2.5-VL-3B-Instruct checkout and one CUDA GPU.

## Score a released drafter (no training)

```bash
curl -L -O https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar xzf qwen25vl-3b-dflash.tar.gz
python examples/score_pretrained.py \
    --target /path/to/Qwen2.5-VL-3B-Instruct --drafter qwen25vl-3b-dflash
```

Same code path as `vlm-dflash demo --drafter <export>`: prints the generation,
the token-exactness check and acceptance length (vs. the random-init
baseline).

## Train from a small JSONL pool

```bash
python examples/train_from_jsonl.py \
    --target /path/to/Qwen2.5-VL-3B-Instruct \
    --jsonl data/rows.jsonl --images-dir data/images --limit 32 --steps 60
```

Runs the full staged pipeline (prepare → target-native regen → capture →
train → eval) on ≤32 rows; the target is loaded for regen/capture only.

## Reuse one capture across algorithms

```bash
python examples/reuse_features_compare.py \
    --target /path/to/Qwen2.5-VL-3B-Instruct \
    --jsonl data/rows.jsonl --images-dir data/images --steps 20
```

Captures features once, then trains DFlash and DSpark from that store via
`train_dflash(..., features_dir=...)` and prints both acceptance lengths. The
store's target identity and saved head are re-verified before each training,
and the reuse is recorded in each run's `provenance.json`.

## Shell-level entry points

The CLI and stage scripts remain the primary interface:

```bash
bash scripts/quickstart.sh --config configs/smoke_coco.yaml --limit 64   # pipeline smoke
bash scripts/quickstart.sh --config configs/qwen2_5_vl_3b.yaml --limit 512  # reference run
vlm-dflash train --dataset data/rows.jsonl --target "$VLM_DFLASH_TARGET" \
    --algorithm dspark --features runs/shared_2k/features --run-dir runs/dspark_2k
```
