# Upstream audit — Domino

Status: **audited; implementation landed** (Stage 3 of the GOAL execution
order). This audit was completed by reading the pinned sources line by line
before the trainer was written; the local implementation ports the trainer
semantics and reuses the correction head already carried by the frozen draft
backbone.

```text
Paper / design:            Domino speculative decoding — specforge/core/domino.py from
                           sgl-project/SpecForge#571 (as described by the maintainer
                           implementation below; no standalone paper pinned)
Official repo:             NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace
Reference implementation:  nemo_automodel/components/speculative/dflash/domino_core.py
                             (DominoTrainerModule, get_lambda_base)
                           nemo_automodel/components/speculative/dflash/draft_qwen3.py
                             (prefix_gru / embed_proj head, suffix_start inference path)
                           examples/speculative/dflash/qwen3_domino.yaml
                           examples/speculative/dflash/qwen2_5_vl_domino.yaml  (VL recipe)
Read-only local mirror:    jacky566/Automodel_lao @ d21fb5aa (carries the same
                           maintainer files; inspected for this port)
License:                   Apache-2.0 (NVIDIA copyright header in domino_core.py)
```

## Semantics recorded from the sources

```text
Backbone:                  DFlash block-parallel draft (frozen port already carries
                           projector_type='domino', pure_draft_prefix_len, shift_label,
                           prefix_gru, embed_proj, set_domino_training_stage)
Extra modules:             prefix_gru: nn.GRU(hidden_size -> gru_hidden_dim, bias=False)
                           embed_proj: Linear(hidden + gru_hidden, emb_dim, bias=False)
                                       -> SiLU -> Linear(emb_dim, vocab, bias=False)
Training objective:        loss = (1 - lambda_base) * final_loss + lambda_base * base_loss,
                           both decayed block CE with the shared block mask / gamma;
                           lambda_base = start * (1 - step / (decay_ratio * total)),
                           clamped to [0, 1] (get_lambda_base)
Inference procedure:       in-block sequential GRU chain over sampled tokens; correction
                           added at head_position >= suffix_start, where
                           suffix_start = pure_draft_prefix_len (shift_label)
                           else 1 + pure_draft_prefix_len
Checkpoint format:         upstream NeMo checkpoints; ours uses the unified
                           vlm-dflash container with the domino fields in
                           draft_config.dflash_config + algorithm='domino'
Defaults (VL recipe):      emb_dim 256, gru_hidden_dim 1024, pure_draft_prefix_len 1,
                           shift_label true, lambda_base_start 1.0,
                           lambda_base_decay_ratio 1.0
```

## What we port / change / why

```text
What we port:   the correction head (already present in the frozen backbone port),
                the dual-logit objective and mixing weight, the curriculum schedule
                (get_lambda_base semantics, same clamps), the suffix-start convention,
                both shift_label branches, and the base-head diagnostics
                (base_loss / base_accuracy / base_accept_len).
What we change: (1) staged offline features instead of the online target forward
                (identical target hidden states; the staged pipeline keeps the target
                off the training GPU); (2) the GRU embeds through the frozen
                target embedding table exactly as upstream (shared embed_tokens);
                (3) curriculum progress is injected by the shared training loop via
                set_training_progress(step, total_steps) instead of a recipe-local
                step counter; (4) an ablation-only enable_correction_head switch
                (not upstream) for the required enabled/disabled comparison;
                (5) metrics are recorded through the repo's extra_metrics channel.
Why we change:  the repo's single contract (staged capture, algorithm registry,
                unified checkpoints) must hold for every algorithm; nothing here
                alters the training math.
Parity level:   Maintainer implementation (ported semantics + same defaults;
                not verified against upstream checkpoints token-for-token).
```

Local code: `vlm_dflash/domino/trainer.py` (dual-logit loss + curriculum),
`vlm_dflash/domino/factory.py` (config plumbing), `vlm_dflash/algorithms/domino.py`
(registry plugin). Tests: `tests/test_domino.py`. User doc:
`docs/algorithms/domino.md`.
