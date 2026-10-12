# Domino

Domino extends the frozen DFlash drafter with a **causal correction head**
(upstream audit: `docs/upstream/domino.md`):

- a small **GRU** (`prefix_gru`) encodes the block's *previous* tokens into a
  causal state, and a **low-rank projection** (`embed_proj`) of
  `[backbone hidden | GRU state]` produces a logit delta that is added to the
  block-parallel base logits on the corrected suffix — restoring in-block
  causality that pure block-parallel drafting lacks;
- training uses a **base-anchor curriculum**:
  `loss = (1 - lambda_base) * final_loss + lambda_base * base_loss`, with
  `lambda_base` decaying linearly from `lambda_base_start` to 0 over the first
  `lambda_base_decay_ratio` fraction of optimizer steps (`final_loss` is the
  CE on the corrected logits, `base_loss` the CE on the backbone-only logits).

## Status

`experimental` (see `ALGORITHM_MATRIX.md` and `SUPPORT_MATRIX.md`): architecture,
training, unit tests and docs exist; real-VLM smoke / canonical numbers are
recorded as they land. Nothing here is a performance claim yet.

## Config

Algorithm knobs are namespaced under `algorithm.domino` (defaults follow the
maintainer's multimodal recipe; DFlash knobs apply unchanged):

```yaml
algorithm:
  name: domino
  domino:
    emb_dim: 256                # low-rank bottleneck of the correction head
    gru_hidden_dim: 1024        # GRU state size (GRU input = target hidden_size)
    pure_draft_prefix_len: 1    # leading block positions kept base-only
    shift_label: true           # slot k predicts anchor + 1 + k (Domino default)
    lambda_base_start: 1.0      # initial base-CE weight of the curriculum
    lambda_base_decay_ratio: 1.0  # decay window (fraction of total steps)
    enable_correction_head: true  # ablation-only switch (see below)
```

No extra capture is needed: the correction head consumes the same cached
features as DFlash (the GRU embeds through the frozen target embeddings), and
the training loop never loads the target.

The **ablation switch** `enable_correction_head: false` is not an upstream knob:
it trains the exact DFlash objective on the same architecture (the GRU/proj are
constructed but never used), which isolates the correction head's contribution
from training randomness for the required enabled/disabled comparison.

## Train / eval / export

```bash
vlm-dflash train  --config configs/domino/qwen2_5_vl_3b_2k.yaml
vlm-dflash eval   runs/qwen25vl3b_domino_2k --config configs/domino/qwen2_5_vl_3b_2k.yaml
vlm-dflash export runs/qwen25vl3b_domino_2k
```

Training writes the standard artifacts; the training log additionally carries
the dual-head metrics `lambda_base`, `final_loss`, `base_loss`,
`base_accuracy`, `base_accept_len` and (with the head enabled)
`correction_abs_mean`. `train/gradient_coverage.json` audits `prefix_gru` /
`embed_proj` as their own parameter groups; the audit runs at 10% of the run
because λ=1 at step 0 intentionally gives the head no gradient (the loss is
pure base CE on the very first step).

## Slot conventions

`shift_label: true` (upstream default): slot `k` predicts `anchor + 1 + k`; the
GRU state for slot `k` is built from the ground-truth token at `anchor + k`; a
round still proposes `block_size - 1` tokens. `shift_label: false` keeps the
DFlash label window and conditions each proposal on the supervised token
itself. Corrections apply from block position `pure_draft_prefix_len`
(shifted) / `1 + pure_draft_prefix_len` (unshifted); leading positions stay on
backbone-only logits, matching upstream `_suffix_start`.

At inference the GRU chains over *sampled* in-block tokens; training
teacher-forces the same chain with ground-truth previous tokens (both paths are
the ported frozen-backbone implementation).

## Known differences from upstream

- Staged/offline feature capture instead of the maintainer's online target
  forward (same frozen target hidden states, different plumbing).
- `enable_correction_head` is an ablation-only addition (documented above).
- Curriculum progress is reported per optimizer step by the shared training
  loop (`set_training_progress`), including inside the overfit-sanity run.
