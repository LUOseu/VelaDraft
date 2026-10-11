# Reference implementation notes

This repo ports the DFlash block-drafter core from a read-only reference so the
training/eval semantics stay identical (and comparable) to the original work:

- Reference: `jacky566/Automodel_lao` @ `d21fb5aa` ("feat(speculative): improve
  Qwen2.5-VL visual drafting"), Apache-2.0. Runbook at commit `85f5dd9c` (`AGENT.md`).
- We only read that repository; this repo never imports it at runtime and never
  depends on its (unpushed) work-in-progress files.

## Ported modules

| Local file | Reference source (at d21fb5aa) | Local modifications |
| --- | --- | --- |
| `vlm_dflash/dflash/draft_model.py` | `nemo_automodel/components/speculative/dflash/draft_qwen3.py` (block-attention draft + speculative `generate`) | import paths; `_DFlashAcceptanceEvent` per-round counters; `set_verify_sdpa_backend_switch` (see below) |
| `vlm_dflash/dflash/trainer_core.py` | `nemo_automodel/components/speculative/dflash/core.py` (training forward, anchor sampling, block targets) | import paths; additive per-slot metrics (`pos_correct`, `pos_count`); plain-torch optim path |
| `vlm_dflash/dflash/mask.py` | `nemo_automodel/components/attention/dflash_mask.py` | import paths only |
| `vlm_dflash/dflash/decay_loss.py` | `nemo_automodel/components/loss/dllm_loss.py` | import paths only |

Success reference configuration (interpreted for our contract):
`examples/speculative/dflash/qwen2_5_vl_dflash_hard_mrope_full.yaml`
(spatial-rope `replace` with sections `[16,24,24]`, mask token 151669,
block size 8, 256 anchors, loss gamma 7, lr 6e-4, warmup 0.04).

## Training semantics (kept identical)

- Block layout: each anchor has `block_size` slots; slot 0 is the clean anchor
  and is **not** supervised; slots `1..block_size-1` are supervised with decay
  weight `exp(-(k-1)/gamma)`.
- Context features: `extract_context_feature` concatenates
  `hidden_states[layer_id + 1]` for every routed target layer (so
  `draft_layer_ids = [1, 17, 33]` means the trained fuse reads target layers
  2/18/34 outputs).
- Speculative decode: per verification round the target verifies `block_size-1`
  draft tokens; on a partial accept the target's own next token (the "bonus"
  token) is appended, so `new_tokens = accepted + 1` per round.
- Acceptance length (recorded in every `eval_result.json`):
  `AL = new_tokens / target_verification_steps` (bonus-inclusive, matches the
  reference `_acceptance_lengths`).
- Per-slot acceptance: per-offset rate `accepted(k) / opportunities(k)` where
  `opportunities(k)` counts rounds that drafted at least `k` tokens (matches the
  reference `_aggregate_proposal_offset_acceptance`; rounds clipped by the
  output budget are excluded from denominators they never had).
- Stopping: sequences are cut at the first stop token (EOS/EOT), both in the
  speculative loop and before any equivalence comparison.
- Output head: the draft logits always go through the **target's own output
  projection** - the tied embedding matrix when `tie_word_embeddings=true`
  (Qwen2.5-VL-3B), the separate `lm_head` when false (Qwen3-VL-8B). The trainer
  therefore consumes the head captured in `features/target_head.pt` (untied
  targets store their own weight), `contract.json` records
  `head.tie_word_embeddings`, every checkpoint stores the `head` tag it was
  trained with, and `train`/`eval` hard-fail when the two disagree. Tying an
  untied target's head to the embeddings trains a projection that inference
  never samples through: training-side acceptance looks healthy while
  end-to-end acceptance collapses to zero.

## MRoPE (Qwen2.5-VL)

- Sections are resolved at runtime from `text_config.rope_parameters.mrope_section`
  (transformers 5.12 no longer exposes a top-level `rope_scaling` on
  Qwen2-VL configs). Resolved value for Qwen2.5-VL-3B: `[16, 24, 24]`.
  The YAML value is a default that must match, otherwise the contract check fails.
- `spatial_rope_mode: replace` feeds the target's 3-axis (t/h/w) position ids
  into the draft rope, replacing the shared text-only positions.
- In transformers 5.12 the Qwen2.5-VL model requires explicit
  `mm_token_type_ids` (`0` = text, `1` = image) for multimodal forwards;
  the processor emits them and capture/eval append `0` for answer tokens.

## MRoPE layouts (Qwen2.5-VL chunked vs Qwen3-VL interleaved)

Both layouts are resolved from the target config by the contract and are
reproduced by the draft's `_merge_multiaxis_rotary_embeddings`; unit tests
(`tests/test_mrope_layouts.py`) compare the constructed tables and the rotated
q/k tensors against the installed transformers implementation.

- Chunked (Qwen2.5-VL, `mrope_section: [16, 24, 24]`): HF's
  `apply_multimodal_rotary_pos_emb` does `mrope_section * 2` on a *list*, so the
  doubled table is split into `[16, 24, 24, 16, 24, 24]` and the six chunks are
  assigned to axes `t, h, w, t, h, w`. The draft merge mirrors that exactly.
- Interleaved (Qwen3-VL, `mrope_interleaved: true`, `mrope_section: [24, 20, 20]`):
  the first `3*max(sections[1:])` half-head channels cycle `t, h, w` and the
  remaining channels stay temporal, reproducing
  `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`. The draft selects this
  layout with `drafter.spatial_rope_interleaved: true`, which the contract
  validates against the target (`rope.interleaved` in `contract.json`).
- Qwen3-VL also requires `mm_token_type_ids` in multimodal forwards (it raises
  otherwise), and the capture stage now feeds it the processor map extended with
  text (`0`) rows for the regenerated answer; Qwen2.5-VL uses the same path so
  capture and eval build identical MRoPE positions.

## Verification modes and token-exactness (important)

`eval.verification` selects how the target verifies drafted blocks
(`spec_generate(..., sequential_target_verification=..., strict_parallel_margin=...)`):

| mode | mechanism | token-exact vs target-only greedy | wall / sample | speed vs target-only |
| --- | --- | --- | --- | --- |
| `sequential` (default, the gate) | one target step per draft token, `generate()`-backend numerics | **32/32 prompts** | 5.25 s | 0.84x |
| `parallel` | one target forward over the whole drafted block | 6/32 | 4.31 s | 1.02x |
| `strict_parallel`, margin 0.25 | parallel + sequential parity re-verify of rounds whose emitted token had a top-1/top-2 logit gap below the margin | 11/32 (4.6% of rounds re-verified) | 4.71 s | 0.97x |
| `strict_parallel`, margin 2.0 | as above with an aggressive threshold | 10/32 (45.8% re-verified) | 6.61 s | 0.67x |
| `strict_parallel`, margin 1e9 (control) | every round re-verified sequentially | **32/32** | 8.66 s | 0.51x |

Measured on the Qwen3-VL-8B 512-sample run (`runs/qwen3vl8b_dflash`,
`ckpt_final.pt`, 32 held-out prompts, greedy, bf16, one A6000,
`verify_sdpa_backend: parity`; raw numbers in `eval/eval_result_*.json`).

Why a block-parallel verify cannot be bitwise-exact on this stack: a forward
over the whole block and a forward over one token evaluate the same function
with different kernel shapes, so their logits differ by up to ~0.3 at the *same
position with identical inputs* (`scripts/diag_verify_drift.py`: prefill max
|delta| = 0, first verify position max |delta| = 0.31, mean 0.07). Those deltas
flip the target argmax whenever the top-1/top-2 gap is smaller than the
drift. Accepted draft tokens then enter the target KV cache *as computed by the
block forward*, so the drift propagates through the cache: a per-round margin
guard can refuse a round's own near-tie, but cannot repair drift already in the
cache. The control run (margin 1e9, i.e. no parallel-computed KV ever
survives) reproduces sequential mode exactly - 32/32 prompts *and* the same AL
(1.329) - which shows the fallback mechanics are correct and the residual
divergence is numerics, not bookkeeping.

Consequences:

* `sequential` is the only mode that is bitwise equivalent to
  `model.generate()` greedy; keep it as the correctness gate (the token-exact
  results above the table, e.g. 64/64 on the 64-prompt gate, are all
  `sequential`).
* `parallel` / `strict_parallel` are throughput experiments. At the current
  drafter quality (AL ~1.33) they are also *not* faster: one block verify costs
  about what the ~1.3 tokens it commits save, and every fallback round adds one
  or more sequential target forwards. In this harness the parallel path breaks
  even at AL ~1.3 and reaches ~1.5x around AL ~2.0; the sequential gate needs
  AL ~1.6 to break even.
* The reference SDPA backend narrowing (`.verify_sdpa_backend: reference`) is a
  second, smaller near-tie source; `parity` disables it. The numbers above are
  `parity`.

## Visual-path checks (our addition)

Two checks prove the drafter conditions on the captured visual context:

1. Capture stage (`features/visual_path_check.json`): replay determinism (same
   image -> bit-identical hidden states), image-swap span changes (all image-span
   rows differ), and downstream propagation (at least K text positions change).
2. Train stage (`train/visual_path_check.json`): with identical token ids,
   swapping image A/B features must change block logits (and block CE), and
   perturbing the image-span feature rows must also move the block CE above a
   1e-5 floor. The perturbation is scale-relative (sigma = 0.5 * std of the span
   features); a fixed 0.05 absolute sigma was too small to clear the floor on a
   trained drafter (CE delta ~9.5e-6), while still being far from a no-op.
   Anchors are resampled under a fixed seed (1234) before every comparison, and
   block CE is recomputed in fp32 (bf16 loss resolution ~1.3e-2 would hide the
   effect).
