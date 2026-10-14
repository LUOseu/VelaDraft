# Upstream audit — DFly

Status: **audited; architecture + hidden-states correction + trainer port landed.
The upstream two-stage objective (LK / D-PACE, e2e TV) is now ported as the
config-selectable ``algorithm.train_recipe: upstream_two_stage`` (exact loss math
in ``vlm_dflash/objectives.py``; see item (5) below). DFly runs without
``train_recipe`` keep the shared DFlash decayed block CE, and every reported
number states which objective profile produced it.**

```text
Paper / design:            AngelSpec: Towards Real-World High Performance Inference
                           with Speculative Decoding — arXiv:2607.25852
                           (DFly = DFlash backbone + TreeFlash hidden-states
                           correction, formula (1), plus a two-stage loss recipe)
Official repo:             Tencent/AngelSpec @ d3412bed231025ea85d35dbf5e0af44f1ac62a5b
Reference implementation:  angelspec/models/draft/dfly.py
                             (DFlyConfig, HiddenStatesCorrection, DFlyDraftModel)
                           angelspec/models/dfly.py (DFlyModel wrapper; applies the
                             correction in _compute_draft_logits, TRAIN TIME)
                           angelspec/training/dfly_trainer.py
                           angelspec/config/dfly_qwen3_8b_draft_config.json
                           configs/vllm_qwen3_8b_dfly.yaml
                           angelspec/models/dflash.py (_compute_lk_loss,
                             _compute_e2e_tv_loss, _dpace_position_weights)
License:                   Apache-2.0 (AngelSpec repo LICENSE; the same file
                           lists its third-party MIT components: specforge,
                           deepspec, torchspec)
```

## Semantics recorded from the sources

```text
Draft model (DFlyDraftModel subclasses DFlareDraftModel, then restores
DFlashDecoderLayer so context and draft once again share k_proj/v_proj):

  layer_context_i = context_norm( FC(flatten(stacked [B,S,T,D]))
                                  + einsum(softmax(fusion_w_i), stacked) )

  i.e. DFlash's single FC context PLUS a DFlare-style per-layer fusion residual,
  RMS-normalized once. FC: Linear(num_target_layers * target_hidden_size, hidden_size).
  Constraint enforced in __init__: target_hidden_size == hidden_size.

Hidden-states correction (TreeFlash formula (1)), optional, default ON:
  h'_{t+i} = h_{t+i} + SwiGLU( norm(h_{t+i}) :: norm(e_{t+i-1}) )
  SwiGLU(x) = down_proj(silu(gate_proj(x)) * up_proj(x));
  hidden_norm / embed_norm are independent RMSNorms; down_proj is ZERO-INIT,
  so the correction starts at exactly 0 (identity) and the model degenerates to
  its uncorrected form until training moves it.
  Applied by the DFlyModel wrapper in _compute_draft_logits BEFORE the LM head,
  with prev_token_ids [B, n_blocks, block_size] teacher-forced previous tokens.
  So upstream defines the correction only as a TRAIN-TIME transformation of the
  logits; the inference-time semantics are not specified in the sources.

Config knobs:              enable_hidden_correction (default true),
                           hidden_correction_intermediate_size (default null ->
                           hidden_size).
Recipe (vllm_qwen3_8b_dfly.yaml): lr 5e-4, min_lr 5e-5, weight_decay 0.1,
                           warmup_ratio 0.04, max_grad_norm 1.0, block_size 16,
                           num_anchors 128, num_target_layers 5, max_seq 2048,
                           4 GPUs, ttt_length 7.
Draft config (dfly_qwen3_8b_draft_config.json): hidden 4096, 5 draft layers,
                           target_layer_ids [1, 9, 17, 25, 33] (= build_target_layer_ids(36, 5)),
                           target_hidden_size 4096, mask_token_id 151669,
                           tie_word_embeddings false, correction enabled.
Two-stage objective (config comments): cold start = pure LK loss
                           (loss_objective "dpace" with alpha 0.5 weighting the LK
                           positions; lk_loss_weight 1.0; lk_loss_type "hybrid",
                           eta 3.0; CE absent), final stage = lk_loss_weight 0.0
                           and e2e_tv_loss_weight 1.0 (multi-step accepted-length
                           objective; gamma = block_size).
  LK loss (hybrid):  lambda*KL(p||q) + (1-lambda)*TV(p,q),
                     lambda = exp(-eta * sg[alpha]),  alpha = sum_v min(p_v, q_v)
  e2e TV loss:       L = 1 - (1/gamma) * sum_{j<=gamma} prod_{i<=j} alpha_i
```

## What we port / change / why

```text
What we port:  the draft architecture (shared K/V decoder layers + DFlash FC base
               + per-layer fusion residual + post-mix RMSNorm), the exact
               HiddenStatesCorrection module (independent RMSNorms, SwiGLU,
               zero-init down_proj), the config knob names/defaults including
               enable_hidden_correction=True, and the enforced
               target_hidden_size == hidden_size constraint. The fusion
               distance-prior init is shared with DFlare.
What we change:
  (1) Correction application surface. Upstream patches the model wrapper's
      _compute_draft_logits (pre-head, teacher-forced previous tokens); this
      repo applies the same transformation through the trainer module's pre-head
      hook with teacher-forced prev tokens taken from input_ids[label-1], so the
      training math matches (zero-init => exact identity at step 0, verified by
      a unit test).
  (2) Inference semantics are a DOCUMENTED ADAPTATION: upstream does not define a
      serving rule for the correction, so this repo defines one explicitly —
      per-slot sequential correction using the real prefix tokens (the anchor,
      then the tokens proposed earlier in the same block) before that slot's
      head projection. With a zero-init down_proj this path is provably identical
      to uncorrected parallel drafting (unit-tested); after training it makes the
      drafter's slot distribution condition on the actual accepted prefix, which
      is the natural serving counterpart of the train-time rule. The
      tree/speculative parallel form of the correction is not ported.
  (3) Feature plumbing, algorithm registration, VL/MRoPE inheritance: identical to
      the DFlare port (see docs/upstream/dflare.md).
  (4) Ablation-only switches (not upstream): enable_hidden_correction=false
      (isolates the correction; upstream default is true), fusion_trainable=false,
      fusion_init="uniform".
  (5) The upstream loss recipe is ported as a config-selectable objective:
      ``algorithm.train_recipe: upstream_two_stage`` runs the LK / D-PACE cold
      start (no CE) and then the D-PACE-weighted CE + end-to-end multi-step TV
      stage — an exact port of angelspec/models/dflash.py's _compute_lk_loss /
      _compute_e2e_tv_loss / _dpace_position_weights, chunked in fp32 (see
      vlm_dflash/objectives.py; test oracles in tests/test_objectives.py). Both
      stages consume the target's final hidden states, so a recipe run requires
      capture.store_final_hidden=true, exactly like JetSpec / DFlash2. The
      stage boundary is this repo's ``cold_start_fraction`` knob (fraction of
      the planned optimizer steps in stage 1); the stage is part of the plan
      digest and of the resume gate. Without ``train_recipe`` the run keeps the
      shared decayed block CE (objective_profile=controlled_ce).
What we do NOT port:
  (6) The tree/speculative parallel form of the inference correction (item (2)
      defines this repo's sequential serving rule), upstream checkpoints, and
      upstream's raw recipe YAML spellings — loss_objective / lk_loss_weight /
      tv_loss / tv_loss_weight / two_stage / d_pace / d_pace_weight stay
      rejected under algorithm.dfly so one config cannot describe one run two
      conflicting ways; select the ported recipe through algorithm.train_recipe
      instead (item (5)).
Why we change: the repo's single contract (registered algorithms, staged/online
               features, unified checkpoints, one shared CE objective pipeline)
               must hold for every algorithm; the ported architecture and
               correction math are unchanged.
Parity level:  architecture + correction = file-level port against the pinned
               commit; objective = file-level port of the LK / D-PACE / e2e-TV
               math, selectable via algorithm.train_recipe, with the shared CE
               profile as the default; inference = documented adaptation. Not
               verified against upstream checkpoints; the canonical 2K recipe
               run is recorded in-house (`R-20261014-11`: AL 1.839 vs
               controlled_ce 1.682, 128/128 token-exact) - no upstream-run
               equivalence is claimed.
```

Local code: `vlm_dflash/dfly/model.py` (FC + fusion residual + correction),
`vlm_dflash/dfly/factory.py` (config plumbing), `vlm_dflash/dfly/trainer.py`
(correction metrics, sequential sampling rule), `vlm_dflash/algorithms/dfly.py`
(registry plugin). Tests: `tests/test_dfly.py`. User doc: `docs/algorithms/dfly.md`.
