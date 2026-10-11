# vlm-dflash

**The simplest reproducible path to train a multimodal DFlash block drafter for Qwen-VL from image-text data.**

v0.2 supports two target families through one adapter interface: a local
**Qwen2.5-VL-3B-Instruct** or **Qwen3-VL-8B-Instruct** checkpoint, single-image
+ text samples, a staged DFlash-style block drafter (block size 8), and a
minimal acceptance evaluation that checks speculative greedy decoding is
**token-exact** against the target's own greedy output. Everything family
specific (MRoPE layout, mask token, pixel bounds, layer routing) is resolved
from the target at runtime and validated against the config.

```
image-text data -> target-native regen -> fused hidden-state capture
                -> DFlash drafter training (target off GPU) -> acceptance eval
```

This is not a new speculative-decoding algorithm and not a framework: it is a
small, complete, runnable loop around the existing DFlash method, with the
multimodal plumbing (Qwen-VL processor, MRoPE, image spans, provenance) done
correctly and testably.

## What you get

- Staged pipeline: every stage writes to `runs/<run>/{data,regen,features,train,eval}`;
  training never keeps the 3B target in GPU memory (it reads captured features).
- A runtime-resolved **model contract** (`contract.json`): mask token id, MRoPE
  sections, draft layer routing, image pixel bounds, special tokens - parsed from
  the target config/tokenizer/processor and validated against the YAML defaults.
- **Provenance gates**: target identity / processor hashes are recorded per stage;
  mismatched identity fails loudly (no silent teacher-target mismatch).
- Minimal eval with a **token-exact equivalence gate** (speculative greedy ==
  target greedy, per token, including EOS handling) plus AL / per-slot
  acceptance / tokens-per-target-forward, and a random-init-drafter acceptance
  baseline (target-only greedy is reported only as an output/speed reference).

## Unified CLI (v0.2)

```bash
vlm-dflash doctor --target /path/to/Qwen3-VL-8B-Instruct
vlm-dflash train  --target /path/to/Qwen3-VL-8B-Instruct \
                  --config configs/qwen3_vl_8b.yaml --preset single-gpu-48g
vlm-dflash eval   runs/<run>                          # token-exact gate + acceptance
vlm-dflash export runs/<run>                          # self-describing HF-style checkpoint
vlm-dflash train  --config ... --resume               # continue from train/ckpt_last.pt
```

`train` runs the whole staged pipeline (prepare -> target-native regen ->
capture -> train); `--limit/--steps` shrink any stage for smoke runs, `--force`
rebuilds finished stages, `--resume` restores model, optimizer, scheduler and
RNG state (a plan mismatch fails loudly). The five stage scripts below remain
the advanced interface. Presets: `smoke` (COCO captions, tiny), `single-gpu-48g`
(measured reference recipe), `single-gpu-24g` (starting point, **not**
hardware-tested).

## Quickstart (4 steps)

```bash
# 0) environment: torch + transformers + torchvision (Qwen2-VL processor)
python -m venv .venv && . .venv/bin/activate
pip install -e . && pip install torchvision

# 1) point the config at your local model + dataset
export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct
export VLM_DFLASH_LLAVA_JSON=/path/to/llava_instruct_150k.json
export VLM_DFLASH_COCO_IMAGES=/path/to/coco/train2017

# 2) one-shot smoke (COCO captions, needs only a few GPU-minutes)
PYTHON=python bash scripts/quickstart.sh --config configs/smoke_coco.yaml --limit 64

# 3) full pipeline (long-answer LLaVA subset, ~1 GPU-hour on an A6000)
PYTHON=python bash scripts/quickstart.sh --config configs/qwen2_5_vl_3b.yaml --limit 512
```

Stages can also be run individually (all take `--config`, `--limit`, `--set k=v`):

```bash
python scripts/prepare_data.py      --config configs/qwen2_5_vl_3b.yaml
python scripts/generate_target_data.py --config configs/qwen2_5_vl_3b.yaml
python scripts/capture_features.py  --config configs/qwen2_5_vl_3b.yaml
python scripts/train_dflash.py      --config configs/qwen2_5_vl_3b.yaml
python scripts/eval_dflash.py       --config configs/qwen2_5_vl_3b.yaml
```

## Why not just patch an existing stack?

Existing spec-decoding stacks target text-only drafters; the multimodal gaps
that bite you are Qwen-VL's MRoPE position ids, image-token spans, the
processor's pixel bounds, and the target's extra reserved mask embedding row.
This repo pins the whole loop around those details, keeps training staged
(no co-resident target), and ships the exact provenance/equivalence checks that
catch teacher-target mismatches and silent text-only regressions.

## Known limitations

- Two target families (Qwen2.5-VL, Qwen3-VL); one drafter family: DFlash block
  drafter. Other families need a small adapter (`vlm_dflash/adapters/`).
- Single image per sample; no video; no visual adaptor stages.
- Acceptance eval is small-scale (64 prompts) and greedy-only; the token-exact
  gate runs in parity mode (sequential verification + generate()-equivalent
  SDPA numerics - see `docs/REFERENCE.md`).
- Not a serving runtime: this repo trains and evaluates a drafter; serving
  integration is roadmap.

## Measured reference run (v0.1, one RTX A6000 48GB)

Full pipeline with `configs/qwen2_5_vl_3b.yaml` (Qwen2.5-VL-3B target, bf16,
greedy everything), measured end to end:

| stage | wall | outcome |
| --- | --- | --- |
| prepare (512 train + 64 val from a 64,969-row LLaVA-Instruct answer>=600-char pool) | < 1 min | `rows_sha256` recorded (stable across reruns) |
| target-native regen (160 max new tokens) | ~6 min | 274/512 kept (regenerated answers must be >= 128 tokens) |
| feature capture (274 samples, seq 2048) | < 1 min | + visual-path check passed |
| drafter training | ~6 min | 210 steps; block CE (first-10% mean) 9.11 -> 6.11 (last-10% mean); no NaN/Inf; overfit sanity passed; peak GPU memory ~30.6 GB (includes the 16-sample sanity phase at batch 16) |
| acceptance eval (64 held-out prompts, <= 128 new tokens) | ~16 min | token-exact gate: 64/64 (trained and random-init); see below |

Results (numbers only, no thresholding):

- AL (speculation): trained drafter **1.220** vs random-init drafter **1.003**
  (random-init drafter is the acceptance baseline; target-only greedy is only an
  output/speed reference - it has no draft proposals, so no AL).
- Per-slot acceptance (trained): `[0.203, 0.013, 0.001, 0, 0, 0, 0]`; draft
  acceptance rate 3.2%.
- Equivalence: speculative greedy == target-only greedy token-exact on all 64
  prompts for both checkpoints in parity mode (`sequential_target_verification:
  true`, `verify_sdpa_backend: parity`). Parallel block verification with the
  reference SDPA narrowing flips bf16 near-tie argmax decisions on longer
  generations (diagnostic: 3/16 token-exact); see `docs/REFERENCE.md`.
- The eval is correctness-first: sequential verification makes wall-clock
  comparable to (not faster than) plain target greedy. Speed is a serving-time
  concern and out of scope for v0.1.

Pipeline smoke validation: `bash scripts/quickstart.sh --config
configs/smoke_coco.yaml --limit 16 --steps 30 --eval-prompts 4` runs all five
stages end to end in ~3.5 min on the same card and finishes with
`token_exact=True` (trained AL 1.113 vs random-init 1.000).

## Measured v0.2 run (Qwen3-VL-8B, same card)

Full pipeline with `configs/qwen3_vl_8b.yaml` (Qwen3-VL-8B-Instruct target,
bf16, 512 long-answer LLaVA-Instruct samples, seq 2048, 6 epochs, block 8,
3 draft layers, interleaved MRoPE), measured end to end on one A6000 48GB:

| stage | wall | outcome |
| --- | --- | --- |
| prepare (512 train + 64 val) | < 1 min | `rows_sha256` recorded |
| target-native regen (max 160 new tokens) | ~14 min | 512/512 kept (>= 128 tokens) |
| feature capture (512 samples, seq 2048) | ~2.5 min | visual-path check passed |
| drafter training (384 steps, incl. the 16-sample overfit sanity) | ~20 min | block CE 10.29 (first 10%) -> 4.78 (last 10%); no NaN/Inf; overfit sanity passed; peak GPU memory 42.9 GB |
| acceptance eval (64 held-out prompts, <= 128 new tokens) | ~19 min | token-exact gate: 64/64 (trained and random-init) |

Results (numbers only, no thresholding):

- AL (speculation): trained drafter **1.322** vs random-init drafter **1.000**.
- Per-slot acceptance (trained): `[0.244, 0.058, 0.018, 0.005, 0, 0, 0]`;
  draft acceptance rate 4.8%.

**Untied target heads.** Qwen3-VL-8B does not tie `lm_head` to the input
embeddings (`tie_word_embeddings: false`). The pipeline handles both layouts:
`capture_features` stores the target's own output projection in
`features/target_head.pt`, `contract.json` records `head.tie_word_embeddings`,
every checkpoint records the head it was trained against, and `train`/`eval`
hard-fail on any disagreement. This matters because training against the
embedding-tied head while `spec_generate` samples through the real `lm_head` is
*silent*: training-side acceptance looks normal and end-to-end AL collapses to
exactly 1.000. The numbers above are a head-consistent retrain; see
`docs/REFERENCE.md` (Training semantics).

CLI smoke (`vlm-dflash train --preset smoke --target .../Qwen3-VL-8B-Instruct
--run-dir runs/cli_smoke_8b --limit 8 --steps 3`, COCO captions) proves the CLI
path end to end against an untied target: capture visual-path check passed,
`vlm-dflash eval` reports `token_exact: true` on 8/8 prompts for both
checkpoints (trained AL 1.076 vs random-init 1.000), and `vlm-dflash export`
writes the portable HF-style checkpoint.

Resume was verified on the same run: re-running with a different `--steps`
fails loudly ("Resume plan mismatch: checkpoint planned 3 total steps, this
config computes 5"), and re-running with the matching plan restores model +
optimizer + scheduler + RNG state so the resumed step is bit-identical (the two
step-3 rows in `train/train_log.jsonl` are equal).

Hardware note: all numbers above are from a single 48 GB card. 24 GB cards were
**not** tested with these defaults; if you are memory-constrained, lower
`train.micro_batch_size`, `capture.seq_length`, or run a smaller slice.
