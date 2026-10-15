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
train → eval); the target is loaded for regen/capture only.

`--limit` is the number of **training** rows — the 4 validation rows are drawn
on top, so this example needs **at least 36 usable rows** in the pool
(32 train plus 4 val); a pool of 32 rows with `--limit 32` leaves an empty val
split. Rows can also shrink during the pipeline: target-native regen drops rows whose
regenerated answer is shorter than `regen.min_response_tokens` (16 in this
example; the default long-answer recipes use 128), so check the counts in
`data/manifest.json` (`num_train` / `num_val`) and `regen/regen_stats.json`
(`num_kept`) before training. JSONL input is not length-filtered — the
`min_answer_chars` filter belongs to the LLaVA-Instruct source, not to JSONL.

One JSON object per line (JSONL — not a pretty-printed JSON array); `image` is
resolved against `--images-dir`:

```jsonl
{"id":"row-001","image":"cat.jpg","question":"What is in this image?","answer":"A cat sleeps on a sofa."}
{"id":"row-002","image":"street.png","question":"Describe the sign in the photo.","answer":"A blue rectangle street sign reads 'Main St'."}
```

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
