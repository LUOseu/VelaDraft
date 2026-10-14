# DSpark

DSpark extends the frozen DFlash drafter with two heads on top of the same
block-parallel backbone (upstream audit: `docs/upstream/dspark.md`):

- a **low-rank Markov logit-bias head** — `bias = W2(W1(prev_token))` added to
  each slot's draft logits, restoring an in-block conditional dependency that
  pure block-parallel drafting lacks (semi-autoregressive generation), and
- a **per-slot confidence head** predicting acceptance probability (the signal
  for future confidence-scheduled draft lengths).

The training objective is selectable: plain runs train the DFlash-style
decayed block CE plus the greedy confidence BCE (the original port), while
`algorithm.train_recipe: upstream_dspark` reproduces the upstream speculators
default objective — verifier-distribution KL distillation plus the analytic
confidence target — by reconstructing the teacher distribution on demand from
the target's final hidden states (see the recipe section below).

## Status

`validated` on Qwen2.5-VL-3B (see `ALGORITHM_MATRIX.md`, `SUPPORT_MATRIX.md`):
architecture, training, unit tests, COCO smoke and the canonical 2K run all
pass; the checkpoint exports and reloads. Qwen3-VL passed reduced-scale
validation (256-row slice, 16-prompt sequential eval, export + reload, all
token-exact -- see `RELEASE_VALIDATION.md`). Runtime integration is future
work (sequential verification only).

Measured (3B / 2K corpus / 128 prompts, sequential): AL 1.735 vs DFlash 1.630
at the same protocol; at the frozen DFlash round width the Markov head alone
accounts for +0.089 and the anchor-slot drafting adds +0.016
(`BENCHMARKS.md`, zoo scoreboard).

## Config

Algorithm knobs are namespaced under `algorithm.dspark` (unknown keys are
rejected by validation; the plain `algorithm: dspark` string form uses the
defaults):

```yaml
algorithm:
  name: dspark
  dspark:
    markov_rank: 256              # 0 disables the Markov head
    markov_head_type: vanilla     # vanilla | gated | rnn
    enable_confidence_head: true
    confidence_head_with_markov: true   # feed W1(prev) into the confidence head
    confidence_head_alpha: 1.0    # BCE weight
    sample_from_anchor: true      # upstream default: block_size proposals/round
```

All DFlash knobs (`drafter.block_size`, `drafter.num_anchors`,
`drafter.loss_decay_gamma`, ...) apply unchanged; the Markov and confidence
heads train on the same cached features (no extra capture, no target residency).

## Train / eval / export

```bash
vlm-dflash train  --config configs/dspark/qwen2_5_vl_3b_2k.yaml
vlm-dflash eval   runs/qwen25vl3b_dspark_2k --config configs/dspark/qwen2_5_vl_3b_2k.yaml
vlm-dflash export runs/qwen25vl3b_dspark_2k
```

`train` writes the standard artifacts plus two DSpark-specific ones:

- `train/confidence_calibration.json` — BCE / ECE (10-bin) / AUROC of the
  confidence signal against greedy slot acceptance, with per-slot predicted vs
  observed acceptance (proves the signal is not constant before any scheduling
  use), and
- `train/gradient_coverage.json` — the GOAL §13 audit (per-group trainable
  params, params with gradient, grad norm), so a head that silently receives no
  gradient cannot pass as trained.

Training-log rows additionally carry `confidence_bce`, `confidence_pred_mean`,
`confidence_target_mean`, `confidence_abs_error` and `markov_bias_abs_mean`.

## Slot conventions

- `sample_from_anchor: true` (upstream DSpark default): slot `k` predicts
  `anchor + k + 1`; every slot is supervised; a verification round proposes
  `block_size` tokens (round window `block_size + 1`).
- `sample_from_anchor: false`: DFlash convention — slot `k` predicts
  `anchor + k`, slot 0 (the clean anchor copy) is not supervised, a round
  proposes `block_size - 1` tokens. This is the DFlash-compatible ablation
  (same round accounting as the frozen baseline, so the two are directly
  comparable at the same `block_size`).

At inference the head chains over *sampled* tokens: slot 0 conditions on the
anchor token, slot `k > 0` on the token sampled at slot `k - 1`. Training
teacher-forces the same chain with ground-truth previous tokens.

## Faithful reference recipe (`algorithm.train_recipe: upstream_dspark`)

The speculators default objective for DSpark — forward
`KL(p_target || p_draft)` distillation with the decayed per-position weight,
plus a confidence BCE against the analytic acceptance rate — is implemented as
the single-stage recipe `upstream_dspark` (config:
`configs/dspark/qwen2_5_vl_3b_2k_recipe.yaml`):

```yaml
algorithm:
  name: dspark
  train_recipe: upstream_dspark    # or {name: upstream_dspark, confidence_coef: 1.0}
capture:
  store_final_hidden: true         # the teacher signal; required and gated
```

Mechanics (all covered by CPU tests in `tests/test_objectives.py` and
`tests/test_dspark.py`):

- The teacher next-token distribution for each supervised slot is projected
  **on demand** from the stored target final hidden states through the frozen
  output head (`TeacherLogitsStream`, position `label - 1`, bounded chunks,
  activation-checkpointed) — full-vocab semantics, no top-k truncation, no
  stored verifier logits.
- The KL term and the confidence target share that one teacher chunk stream
  per step, so the recipe costs a single teacher projection per forward (the
  confidence target is not a second pass).
- The confidence target is
  `sum_v min(p_draft, p_target) = 1 - TV`, computed per step from the **live**
  Markov-corrected drafter logits (detached); it is never cached as a fixed
  feature, so it tracks the current policy. The BCE keeps the position decay
  of the distribution term.
- `train_recipe.confidence_coef` (default `1.0`, the upstream
  `confidence_head_alpha`) is recorded in the training-plan digest. Under the
  recipe `algorithm.dspark.confidence_head_alpha` must stay at `1.0` (fail
  loud otherwise), the confidence head must be enabled, and the capture/store
  must carry final hidden states; all three are checked at config-validation
  time.

Runs without `train_recipe` keep the original port objective: DFlash-style
decayed block CE plus the greedy per-slot acceptance indicator as the
confidence target (the temperature-0 verification event the calibration
report measures). That mode remains the default; its plan digests are
byte-identical to the pre-recipe scheme.

## Known differences from upstream

- Objective: covered by the recipe above. Without it, plain runs use the
  documented greedy-confidence adaptation instead of the analytic target.
- Normalization convention: the distribution term (and every other decayed
  term in this repo) is a decay-weighted mean (divisor `sum(mask * decay)`),
  while speculators divides by the plain mask count (their `loss_function`).
  Gradient direction is identical; the effective step scale differs by the
  batch's mean decay factor. Within-repo comparisons are consistent because
  all objectives share this convention.
- Decay gamma: our dev configs run `loss_decay_gamma: 7.0`; the upstream
  default is `4.0` (`dflash_decay_gamma` in their train schema). The recipe
  uses the run's value — set it to `4.0` to match upstream numerics exactly.
- Draft vocabulary: upstream's example pipeline supports a truncated
  draft-vocab mapping (e.g. 32k of a 151k verifier vocab); this repo trains
  and distils over the full target vocabulary (the untruncated semantics).
- `markov_head_type: rnn` follows NeMo's unrolled-state semantics for training
  and stateful chaining for sampling.

## Tests

`tests/test_dspark.py` covers: head factorization vs manual W2(W1(prev)),
small-init bias bound, gradient flow through gated/rnn heads, the sampling
chain (a bias favoring token X after Y picks X first), tiny-batch overfit for
both slot conventions with both heads receiving gradient, gradient-coverage
flagging of an unused head, the calibration report, and the registry/config
plumbing (namespaced config, checkpoint metadata, inference-module rebuild).
