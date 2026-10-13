# DFly

DFly keeps the **DFlash decoder stack and its single FC context**, adds
DFlare's **per-draft-layer fusion as a residual** on top of that FC, and adds
the optional **TreeFlash hidden-states correction** (upstream audit:
`docs/upstream/dfly.md`):

- every draft layer's context is
  `context_norm( FC(concat(target layers)) + fusion_i(target layers) )` — the
  DFlash FC gives a shared base, the DFlare-style softmax mix (same
  relative-depth prior init) is added per layer before the shared RMSNorm;
- context and draft K/V share the DFlash projections (DFly restores
  `DFlashDecoderLayer`; there is no `k_proj_target`/`v_proj_target`);
- `HiddenStatesCorrection` conditions the **pre-head hidden state** on the
  previous token:
  `h' = h + down(silu(gate(cat(norm(h), norm(embed(prev))))) * up(...))`,
  with `down_proj` zero-initialized so the module starts as an exact identity.

## Status

Architecture + trainer ported, unit-tested on CPU, and recorded on
Qwen2.5-VL-3B: COCO 64-row smoke (token-exact 8/8, AL 1.438 vs init 1.000),
the canonical 2K matched run — **AL 1.682, token-exact 128/128** (+0.052 over
the frozen DFlash 2K baseline; deep slots carry the gain, slot 1 gives some
back) — the layer-rich variant (**AL 1.709**), the no-correction ablation
(**AL 1.569** — the zero-init hidden-states correction is worth +0.113, i.e.
DFly's gain at this scale) and the Qwen3-VL-8B reduced validation (**AL
1.226**, token-exact 16/16). The correction trains from its zero init
(|delta| 0 → 1.81 at step 792; the 2K step-0 coverage audit shows only
`down_proj` with gradient at that instant and the gate/up path lights up
from step 1 — unit-tested and re-checked on `ckpt_final.pt` with
`scripts/diag_gradient_coverage.py`, 42/42); the fusion stays near its prior
(drift 0.0039).
**The upstream two-stage objective (LK / D-PACE, e2e TV) is not ported** — see
"Objective" and `docs/upstream/dfly.md`; DFly runs here train the ported
architecture with the shared DFlash decayed block CE, and any reported number
must say so.

## Objective (read before citing numbers)

Upstream `vllm_qwen3_8b_dfly.yaml` trains two stages: a cold start on the LK
(acceptance-rate) distillation loss with D-PACE position weighting
(`lk_loss_weight: 1.0`, `dpace_alpha: 0.5`, `lk_loss_type: hybrid`, no CE), then
a final stage on the end-to-end multi-step TV objective
(`lk_loss_weight: 0.0`, `e2e_tv_loss_weight: 1.0`). **None of these losses are
implemented in this repo yet**; the trainer uses the same decayed block CE as
DFlash/DFlare (anchors, block mask, slot conventions unchanged). The gap is
tracked in `ROADMAP.md`; until it lands, DFly here is an architecture +
correction port, not an objective reproduction.

So a config cannot silently claim the upstream recipe: every DFly run is
stamped with a machine-readable `objective_profile: controlled_ce` (draft
config, checkpoint metadata, export and model card), and the upstream recipe
knobs (`loss_objective`, `lk_loss_weight`, `tv_loss`, `tv_loss_weight`,
`two_stage`, `d_pace`, `d_pace_weight`) are **rejected at build time** with a
message pointing back at this profile instead of being ignored.

## Config

```yaml
drafter:
  capture_layer_ids: [1, 9, 17, 25, 33]   # ordered superset actually captured
algorithm:
  name: dfly
  dfly:
    context_layer_ids: [1, 17, 33]   # subset the FC + fusion residual consume
    fusion_trainable: true
    fusion_init: distance
    enable_hidden_correction: true            # upstream default
    hidden_correction_intermediate_size: null # null -> hidden_size
```

Rules enforced at build time: the same capture-subset contract as DFlare
(`context_layer_ids ⊆ drafter.capture_layer_ids ⊇ draft routing`),
`target_hidden_size == hidden_size`, `sample_from_anchor: false` and shared
context K/V (`separate_context_kv: true` is rejected with a clear error).

Canonical configs mirror the DFlare set: `configs/dfly/smoke_coco.yaml`,
`configs/dfly/qwen2_5_vl_3b_2k.yaml` (matched, `[1, 17, 33]`),
`configs/dfly/qwen2_5_vl_3b_2k_layers.yaml` (upstream 5-layer context
`[1, 9, 17, 25, 33]` = `build_target_layer_ids(36, 5)`) and
`configs/dfly/qwen3_vl_8b_reduced.yaml`. Upstream recipe reference
(`vllm_qwen3_8b_dfly.yaml`): lr 5e-4, min_lr 5e-5, weight decay 0.1, block 16,
128 anchors, 5 draft layers / 5 target layers of a 4096-wide target.

## Train / eval / export

```bash
vlm-dflash train  --config configs/dfly/qwen2_5_vl_3b_2k.yaml
vlm-dflash eval   runs/qwen25vl3b_dfly_2k --config configs/dfly/qwen2_5_vl_3b_2k.yaml
vlm-dflash export runs/qwen25vl3b_dfly_2k
```

Training logs the correction diagnostics: `correction_enabled` (1.0/0.0),
`correction_delta_abs_mean` (mean |correction| before the head; starts at 0
with the zero-init down_proj and grows as training moves it) and
`context_base_abs_mean` / `context_fusion_residual_abs_mean` (FC base vs fusion
residual scale). The fusion metrics are the DFlare ones; they read the
optimizer's fp32 masters and also log `fusion_prob_l1_drift` from the prior
(see `docs/algorithms/dflare.md` for why the bf16 live parameter alone would
under-report movement). The correction trains through a pre-head hook with
teacher-forced previous tokens (`input_ids[label-1]`), matching upstream's
train-time application.

## Inference semantics (documented adaptation)

Upstream defines the correction only at train time (the model wrapper's
`_compute_draft_logits`). This repo defines the serving rule explicitly and
documented:

- proposals are generated **per slot in order**; slot `k` is corrected with the
  real prefix tokens — the anchor for slot 1, then the token proposed at slot
  `k-1` — before that slot's head projection. This is the natural serving
  counterpart of the teacher-forced training rule and makes each proposal
  condition on the actual in-block prefix.
- with a zero-init `down_proj` the path is **provably identical** to uncorrected
  parallel drafting (unit-tested), so a freshly initialized or
  `enable_hidden_correction: false` checkpoint serves exactly like DFlash.

## Ablations

- `enable_hidden_correction: false` — removes the correction (upstream default
  is true); isolates its contribution.
- `fusion_trainable: false` / `fusion_init: uniform` — fusion ablations, shared
  with DFlare.
- `context_layer_ids` — the layer-rich config above; same store as the matched
  run.

## Known differences from upstream

See `docs/upstream/dfly.md`: objective = shared CE (upstream two-stage
LK/D-PACE/e2e-TV not implemented); inference rule = documented adaptation; the
correction is applied through the repo's pre-head trainer hook instead of the
model wrapper; staged/online feature stores with explicit layer slicing replace
the live target forward; the VL MRoPE/image plumbing is this repo's.
