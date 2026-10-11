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
  `eval.verification` selects `sequential` (the parity gate) / `parallel` /
  `strict_parallel`; see the runtime probe below.

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
  gate runs in `verification: sequential` (generate()-equivalent numerics -
  see `docs/REFERENCE.md`). Parallel / strict-parallel verification is
  available but is not bitwise-exact in bf16 and was not faster at the
  drafter quality measured so far.
- Not a serving runtime: this repo trains and evaluates a drafter; serving
  integration is roadmap. `sequential` verification is correctness-first, not
  a speed path.

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
  prompts for both checkpoints in parity mode (`verification: sequential`,
  `verify_sdpa_backend: parity`). Parallel block verification flips bf16
  near-tie argmax decisions on longer generations; see the runtime probe below
  and `docs/REFERENCE.md`.
- Rerun on the current code (`runs/qwen25vl3b_refresh`, same config and seed):
  token-exact 64/64 again, AL 1.198 (vs 1.220 first run) - see the
  non-determinism note in the runtime-probe section.
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

### Verification modes and a runtime probe (v0.2, Qwen3-VL-8B)

`eval.verification` selects how the target verifies drafted blocks (see
`docs/REFERENCE.md` for the mechanism):

| mode | token-exact vs target-only greedy | wall / sample | speed vs target-only |
| --- | --- | --- | --- |
| `sequential` (default; the gate) | **32/32 prompts** | 5.25 s | 0.84x |
| `parallel` (one forward per block) | 6/32 | 4.31 s | 1.02x |
| `strict_parallel`, margin 0.25 | 11/32 (4.6% of rounds re-verified) | 4.71 s | 0.97x |
| `strict_parallel`, margin 2.0 | 10/32 (45.8% re-verified) | 6.61 s | 0.67x |
| `strict_parallel`, margin 1e9 (control) | **32/32** | 8.66 s | 0.51x |

Protocol: `ckpt_final.pt`, 32 held-out prompts, greedy, bf16, one A6000,
`verify_sdpa_backend: parity`; raw numbers in `eval/eval_result_*.json`, drift
probe in `scripts/diag_verify_drift.py`. The 8B rows above are the
512-sample drafter (`runs/qwen3vl8b_dflash`, AL ~1.33). The same probe on the
2K-scale 3B drafter (`runs/qwen25vl3b_2k`, AL ~1.68) shows the speed crossing
over while exactness does not:

| mode | token-exact | AL | wall / sample | speed vs target-only |
| --- | --- | --- | --- | --- |
| `sequential` (gate) | **32/32** | 1.692 | 4.35 s | 0.86x |
| `parallel` | 8/32 | 1.677 | 2.97 s | **1.25x** |
| `strict_parallel`, margin 0.25 | 7/32 (14.0% re-verified) | 1.690 | 3.66 s | 1.03x |

Reading:

- Block-parallel verification is not bitwise-equivalent to step mode in bf16:
  the same position with identical inputs moves by up to ~0.31 logits (prefill
  deltas are 0). Accepted tokens carry that drift into the KV cache, so a
  per-round near-tie guard reduces but cannot eliminate divergence - the
  control run re-verifies every round and lands exactly on 32/32 with the same
  AL, which pins the residual divergence on numerics rather than bookkeeping.
- Speed scales with drafter quality, exactness does not follow: the parallel
  path measured 1.02x at AL 1.33 (512-sample 8B drafter) and 1.25x at AL 1.68
  (2K-scale 3B drafter), while the sequential gate - the only bitwise-exact
  mode - was 0.84x and 0.86x on the same settings.
- `sequential` therefore remains the correctness gate; parallel modes are
  throughput experiments whose agreement rate is reported per run.

### Scaling (Phase 2, in progress)

Same recipe, larger LLaVA-Instruct slices, Qwen2.5-VL-3B target,
`verification: sequential` (128 val prompts; the 512 point keeps the original
64-prompt val set for comparison with v0.1):

| slice | kept | train steps | AL (trained) | AL (random-init) | per-slot (trained) | token-exact |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 274 | 210 | 1.198 | 1.003 | [0.184, 0.011, 0, 0, 0, 0, 0] | 64/64 |
| 2000 | 1050 | 792 | **1.630** | 1.003 | [0.413, 0.149, 0.048, 0.015, 0.004, 0.001, 0] | 128/128 |
| 5000 | running | | | | | |

(512 row is the post-v0.2 rerun; the original v0.1 run measured 1.220 - see the
non-determinism note below. 2000-slice training: block CE 7.50 -> 3.79
first/last-10% means, overfit sanity passed, peak 29.5 GB, ~10 min on one
A6000.) Slots 2-5 become non-zero between the 512 and 2K points, and the
runtime probe above shows the parallel verification path crossing 1x on the
same checkpoint; the 5K and 8B-2K points are produced by
`scripts/run_scaling_suite.sh`.

Same-seed reruns: data, regen and the protocol reproduce (`rows_sha256`
identical, 274/512 regen keeps identical), but the bf16 capture and training
passes are not bitwise deterministic across runs (feature shard hashes and
step-1 loss differ between identical reruns; AL moved 1.220 -> 1.198 on the
3B config). Reproduce the protocol, not the bits; treat sub-2% AL differences
as run-to-run noise.

Hardware note: all numbers above are from a single 48 GB card. 24 GB cards were
**not** tested with these defaults; if you are memory-constrained, lower
`train.micro_batch_size`, `capture.seq_length`, or run a smaller slice.
