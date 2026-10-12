# Environment preflight (development host findings)

Recorded 2026-10-11 on the development hosts. Personal paths are kept out of
the repository on purpose; substitute your own checkout and dataset roots.

## GPU host layout

- Repo checkout: `~/vlm-dflash` on the GPU host (rsync mirror of the working
  copy; `runs/`, `.venv/`, `assets/` are excluded from the mirror).
- Python: `.venv` created with `uv`, Python 3.11.16.
- GPUs: 3x RTX A6000 48GB. Convention: `CUDA_VISIBLE_DEVICES=0` (other cards
  belong to other users; never run multi-GPU).
- Sync command (from the working copy root):
  `rsync -az --exclude runs/ --exclude .venv/ --exclude assets/ --exclude __pycache__/ ./ <gpu-host>:~/vlm-dflash/`

## Pinned version combination (tested)

| package | version | note |
| --- | --- | --- |
| torch | 2.10.0 (cu128) | cluster wheel |
| torchvision | 0.25.0 | required by `Qwen2VLVideoProcessor` import, even for images |
| transformers | 5.12.1 | MRoPE sections live at `text_config.rope_parameters`; VL forwards need `mm_token_type_ids` |
| datasets | 4.2.0 | data prep |
| accelerate | 1.15.0 | device placement |
| huggingface-hub | 1.33.0 | |
| modelscope | 1.40.2 | model download path |
| tokenizers | 0.22.2 | |
| sentencepiece | (n/a) | not needed |
| pillow | 12.3.0 | image decode |
| numpy | 2.4.6 | |
| safetensors | 0.8.0 | |

## Model availability

- Target: `Qwen2.5-VL-3B-Instruct`, downloaded via **ModelScope** to
  `assets/models/Qwen2.5-VL-3B-Instruct` (~7.1 GB). Verified local load with
  `local_files_only: true`.
- `hf-mirror.com` direct download was not usable from this host (xet bridge
  capped around 21 KB/s); ModelScope worked. HF direct connectivity is
  restricted on this cluster.
- Model fingerprint (sha256 over config/generation/index + weight shard sizes)
  and per-file processor hashes are recorded in `runs/<run>/contract.json`.
  Qwen2.5-VL-3B @ download: `e044f48898fec43dce11810610f9b11ea048cbb7e99057423b79619f67d426d5`.

## Data availability (read-only cluster mirrors)

- LLaVA-Instruct-150K conversations: point `VLM_DFLASH_LLAVA_JSON` at the
  local mirror of `llava_instruct_150k.json`. With `min_answer_chars: 600` the
  pool is 64,969 rows.
- COCO train2017 images + captions: point `VLM_DFLASH_COCO_CAPTIONS` /
  `VLM_DFLASH_COCO_IMAGES` at the local mirrors. Used only for the pipeline
  smoke config (`configs/smoke_coco.yaml`).
- Mirrors are read-only; the pipeline only ever writes under `runs/`.

## Verified smoke facts (Qwen2.5-VL-3B, bf16, sdpa)

- Model load: ~4.8 s; single-image greedy generate works; peak reserved memory
  ~7.7 GB for a single 3B target forward/generate.
- Image token count sanity: 345 image tokens == `grid_thw` product / merge^2
  for the default pixel bounds (min 200704 / max 1003520).
- Determinism: replaying the same image/prompt gives bit-identical captured
  hidden states (`max_diff == 0`).

## Test/run commands

- Unit + tiny-CPU tests (no GPU):
  `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/ -q`
- Pipeline stages (single GPU):
  `CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/<stage>.py --config configs/<cfg>.yaml [--limit N] [--steps N] [--set key=value]`
- `scripts/quickstart.sh` uses `$PYTHON` (default `python3`); on the cluster run
  it with `PYTHON=.venv/bin/python bash scripts/quickstart.sh ...`.
