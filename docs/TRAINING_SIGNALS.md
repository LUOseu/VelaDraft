# Training signals

What each method's training objective consumes, per method: hard-label
supervision, context target layers, whether the target's final hidden states
are required (and under which selectable objective), distribution and
confidence objectives, stage structure, and the block supervision convention.
This is the requirement declaration behind the recipe machinery
(`algorithm.train_recipe`) and the capture gates
(`capture.store_final_hidden`).

Generated from `AlgorithmMetadata.training_signals`
(`vlm_dflash/algorithms/<name>.py`) and the recipe supports-matrix
(`RECIPE_SUPPORT` in `vlm_dflash/objectives.py`). Regenerate with
`python scripts/render_training_signals.py`. Signal text is deliberate: a
requirement stated here must match the trainer's behavior, and any change is
reviewed together with its tests.

## Recipes accepted per algorithm

- `dflare`: `controlled_ce`, `upstream_two_stage`
- `dflash`: `controlled_ce`, `upstream_two_stage`
- `dflash2`: none
- `dfly`: `controlled_ce`, `upstream_two_stage`
- `domino`: none
- `dspark`: `upstream_dspark`
- `jetspec`: none

## `dflare`

| Signal | Requirement |
| --- | --- |
| Hard label | Yes - decayed block CE on regenerated targets (the shared DFlash objective). |
| Context layers | Per-draft-layer fusion over the captured target layers (context_layer_ids; null = every captured layer) plus dedicated target-context K/V projections. |
| Teacher final hidden states | Not used by the default objective; required by train_recipe: upstream_two_stage. |
| Distribution objective | None by default; LK + end-to-end TV under the recipe. |
| Confidence objective | None. |
| Stage | Single stage by default; two-stage recipe as for DFlash. |
| Block convention | DFlash slot convention (slot 0 anchor clean, unsupervised; S-1 proposals). |

Docs: [`docs/algorithms/dflare.md`](algorithms/dflare.md)

## `dflash`

| Signal | Requirement |
| --- | --- |
| Hard label | Yes - decayed block CE on regenerated targets; slots 1..block_size-1 (slot 0 is the clean anchor, unsupervised). |
| Context layers | Contract capture layers (drafter.capture_layer_ids; fused [T*H] context projection shared by every draft layer). |
| Teacher final hidden states | Not used by the default objective; required by train_recipe: upstream_two_stage. |
| Distribution objective | None by default; LK (hybrid KL/TV) + end-to-end multi-step TV under the recipe. |
| Confidence objective | None. |
| Stage | Single stage by default; the recipe switches LK cold start -> e2e TV at cold_start_fraction. |
| Block convention | DFlash convention: slot 0 = clean anchor (unsupervised), slots 1..S-1 supervised; S-1 proposals per round. |

Docs: [`docs/algorithms/dflash.md`](algorithms/dflash.md)

## `dflash2`

| Signal | Requirement |
| --- | --- |
| Hard label | Selector term: hard K-way CE over the unary top-K candidates (the target id is forced in when absent). The unary term is a distribution objective, not hard-label CE. |
| Context layers | Contract capture layers (shared DFlash backbone). |
| Teacher final hidden states | Required (capture.store_final_hidden=true): the unary term distils the target's next-token distribution reconstructed through the frozen output head. |
| Distribution objective | Yes - forward KL(target \|\| draft) on the unary logits with the fixed exponential decay (gamma 4.0); an experimental reconstruction - upstream does not publish the training objective. |
| Confidence objective | None (the selector reranks candidates; it is not an acceptance head). |
| Stage | Single stage. |
| Block convention | DFlash slot convention (slot 0 anchor clean, unsupervised). |
| Notes | Objective is a reconstruction, not a reproduction; full vocabulary required. |

Docs: [`docs/algorithms/dflash2.md`](algorithms/dflash2.md)

## `dfly`

| Signal | Requirement |
| --- | --- |
| Hard label | Yes - decayed block CE by default; under the recipe, D-PACE-weighted block CE in stage 2 (stage 1 is LK-only, no CE). |
| Context layers | context_layer_ids with per-layer fusion + the hidden-states correction (upstream recipe: [1,9,17,25,33]; matched 3-layer variant: [1,17,33]). |
| Teacher final hidden states | Recipe only - train_recipe: upstream_two_stage consumes the target's final hidden states in both stages (LK and e2e TV). |
| Distribution objective | Under the recipe: LK hybrid (KL/TV blend) cold start, then the end-to-end multi-step TV term. |
| Confidence objective | None as a trained head; D-PACE confidences are detached position weights derived from the CE. |
| Stage | Single stage by default; the recipe switches cold_start_lk_dpace -> e2e_tv. |
| Block convention | Forced DFlash convention (sample_from_anchor=false regardless of the YAML value). |

Docs: [`docs/algorithms/dfly.md`](algorithms/dfly.md)

## `domino`

| Signal | Requirement |
| --- | --- |
| Hard label | Yes - decayed block CE on regenerated targets; the base-anchor curriculum mixes CE on the corrected logits with CE on the backbone-only logits (lambda_base decays linearly to 0). |
| Context layers | Contract capture layers (shared DFlash context). |
| Teacher final hidden states | Not used (the curriculum trains on hard regenerated targets). |
| Distribution objective | None (the port source defines the curriculum CE only). |
| Confidence objective | None. |
| Stage | Single stage; the curriculum is a config schedule (lambda_base_start / lambda_base_decay_ratio). |
| Block convention | shift_label=true (default): anchor included, all slots supervised (slot k predicts anchor+k+1); false: DFlash convention (slots 1..S-1). |

Docs: [`docs/algorithms/domino.md`](algorithms/domino.md)

## `dspark`

| Signal | Requirement |
| --- | --- |
| Hard label | Default: decayed block CE on regenerated targets, every supervised slot. sample_from_anchor=true (upstream default) supervises all S slots; false keeps the DFlash convention (slots 1..S-1). |
| Context layers | Contract capture layers (same fused context as DFlash). |
| Teacher final hidden states | Recipe only - train_recipe: upstream_dspark streams the target's final hidden states through the frozen output head to reconstruct the teacher distribution. |
| Distribution objective | None by default; under the recipe, per-slot forward KL(p_target \|\| p_draft) over the full vocabulary, decay-weighted (the speculators default loss_fn=kl_div). |
| Confidence objective | Greedy per-slot acceptance indicator by default (the temperature-0 verification event); under the recipe, the analytic acceptance rate sum_v min(p_draft, p_target) = 1 - TV, computed from the live drafter distribution each step (detached) - never cached. |
| Stage | Single stage. |
| Block convention | sample_from_anchor=true: slot k predicts anchor+k+1, all slots supervised, S proposals per round (round window S+1); false: DFlash convention, S-1 proposals. |

Docs: [`docs/algorithms/dspark.md`](algorithms/dspark.md)

## `jetspec`

| Signal | Requirement |
| --- | --- |
| Hard label | No hard-label CE - the trainer replaces it with the teacher-distribution KL (it never degenerates to CE). |
| Context layers | Contract capture layers (unmodified DFlash backbone; no new parameters). |
| Teacher final hidden states | Required (capture.store_final_hidden=true): teacher logits are the frozen output-head projection of the stored final hidden states at the gathered positions (bit-exact reproduction of the target's own logits). |
| Distribution objective | Yes - temperature-scaled forward KL(P_target \|\| Q_draft), T^2 scaling, uniform weighting over slots 1..S-1 (no depth decay). |
| Confidence objective | None. |
| Stage | Single stage. |
| Block convention | DFlash slot convention (slot 0 anchor clean, unsupervised; S-1 proposals) with the block-causal in-block attention mask. |

Docs: [`docs/algorithms/jetspec.md`](algorithms/jetspec.md)
