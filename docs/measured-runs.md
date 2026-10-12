# Measured end-to-end runs

Reference runs recorded during development of v0.1-v0.3; the canonical
algorithm-zoo scoreboard lives in `BENCHMARKS.md`. All numbers below: greedy
decoding, bf16, one NVIDIA RTX A6000 48 GB, `verification: sequential`
(token-exact gate).

## v0.1 - Qwen2.5-VL-3B, 512-row LLaVA-Instruct slice (`configs/qwen2_5_vl_3b.yaml`)

| stage | wall | outcome |
| --- | --- | --- |
| prepare (512 train + 64 val from an answer>=600-char LLaVA-Instruct pool) | < 1 min | `rows_sha256` recorded (stable across reruns) |
| target-native regen (160 max new tokens) | ~6 min | 274/512 kept (regenerated answers must be >= 128 tokens) |
| feature capture (274 samples, seq 2048) | < 1 min | visual-path check passed |
| drafter training | ~6 min | 210 steps; block CE (first-10% mean) 9.11 -> 6.11 (last-10% mean); no NaN/Inf; overfit sanity passed; peak ~30.6 GB |
| acceptance eval (64 held-out prompts, <= 128 new tokens) | ~16 min | token-exact 64/64 (trained and random-init) |

- AL: trained drafter **1.220** vs random-init drafter **1.003**.
- Per-slot acceptance (trained): `[0.203, 0.013, 0.001, 0, 0, 0, 0]`; draft
  acceptance rate 3.2%.
- Rerun on the current code (`runs/qwen25vl3b_refresh`, same config and seed):
  token-exact 64/64 again, AL 1.198 (vs 1.220 first run).

Pipeline smoke: `bash scripts/quickstart.sh --config configs/smoke_coco.yaml
--limit 16 --steps 30 --eval-prompts 4` runs all five stages in ~3.5 min and
finishes with `token_exact=True` (trained AL 1.113 vs random-init 1.000).

## v0.2 - Qwen3-VL-8B-Instruct, 512-row slice (`configs/qwen3_vl_8b.yaml`)

| stage | wall | outcome |
| --- | --- | --- |
| prepare (512 train + 64 val) | < 1 min | `rows_sha256` recorded |
| target-native regen (max 160 new tokens) | ~14 min | 512/512 kept |
| feature capture (512 samples, seq 2048) | ~2.5 min | visual-path check passed |
| drafter training (384 steps incl. overfit sanity) | ~20 min | block CE 10.29 -> 4.78 (first/last-10% means); peak 42.9 GB |
| acceptance eval (64 held-out prompts) | ~19 min | token-exact 64/64 (trained and random-init) |

- AL: trained drafter **1.322** vs random-init **1.000**; per-slot
  `[0.244, 0.058, 0.018, 0.005, 0, 0, 0]`.

**Untied target heads.** Qwen3-VL-8B does not tie `lm_head` to the input
embeddings. `capture_features` stores the target's own output projection in
`features/target_head.pt`, `contract.json` records `head.tie_word_embeddings`,
every checkpoint records the head it was trained against, and `train` / `eval`
hard-fail on disagreement - training against the tied head while the runtime
samples through the real `lm_head` is *silent* (training-side acceptance looks
normal, end-to-end AL collapses to 1.000).

CLI smoke against the untied target (`--preset smoke --limit 8 --steps 3`):
capture visual-path check passed, `vlm-dflash eval` reports
`token_exact: true` on 8/8 prompts (trained AL 1.076 vs random-init 1.000),
export writes the portable checkpoint. Resume was verified on the same run:
a mismatched plan fails loudly ("Resume plan mismatch: checkpoint planned
3 total steps, this config computes 5"); a matching plan restores model +
optimizer + scheduler + RNG state (the two step-3 rows in `train_log.jsonl`
are equal).

## Scaling (Qwen2.5-VL-3B, `verification: sequential`)

| slice | kept | train steps | AL (trained) | AL (random-init) | per-slot (trained) | token-exact |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 274 | 210 | 1.198 | 1.003 | [0.184, 0.011, 0, 0, 0, 0, 0] | 64/64 |
| 2000 | 1050 | 792 | **1.630** | 1.003 | [0.413, 0.149, 0.048, 0.015, 0.004, 0.001, 0] | 128/128 |
| 5000 | 2629 | 1974 | **1.817** | 1.003 | [0.482, 0.210, 0.080, 0.030, 0.011, 0.005, 0.001] | 128/128 |

(2000-slice: block CE 7.50 -> 3.79, peak 29.5 GB, ~10 min; 5000-slice: block CE
6.84 -> 2.58, peak 30.0 GB, ~40 min. Acceptance is still length-limited by the
128-prompt / 128-token eval budget - "more data still helps", not a plateau.
The Qwen3-VL-8B 2K point is 1.665 at 32 prompts / 128 tokens.)

## Reproducibility statement

Same-seed reruns: data, regen and the protocol reproduce (`rows_sha256`
identical, regen keeps identical), but the bf16 capture and training passes are
**not bitwise deterministic** across runs (feature shard hashes and step-1
loss differ between identical reruns; AL moved 1.220 -> 1.198 once on the same
config). Reproduce the protocol, not the bits; treat sub-2% AL differences as
run-to-run noise. The token-exact gate itself is deterministic given a
checkpoint (the frozen 2K checkpoint re-evaluated bit-identically after the
full algorithm zoo landed - see `tests/reference_dflash/manifest.json`).
