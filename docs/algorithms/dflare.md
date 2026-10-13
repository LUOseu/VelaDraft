# DFlare

DFlare replaces DFlash's single shared context projection with a **learnable
per-draft-layer fusion** over the captured target layers, and gives the
target-context K/V **dedicated projections** (upstream audit:
`docs/upstream/dflare.md`):

- every draft layer `d` keeps its own softmax distribution
  `layer_fusion_weights[d]` over the `T` consumed target layers and attends to
  its own weighted mixture (`einsum("t,bstd->bsd")` -> RMSNorm), instead of one
  `Linear(T*H, H)` context shared by all draft layers;
- the fusion logits are initialized to a **relative-depth prior**
  (`logits[d, t] = -2 |t - t*_d|`, `t*_d` = the target layer at draft layer
  `d`'s relative depth), so the softmax starts peaked at the matching depth
  rather than uniform;
- `k_proj_target` / `v_proj_target` project the target-context K/V while the
  draft tokens keep the shared `k_proj` / `v_proj`; the concat ->
  K-norm -> RoPE -> block-mask flow is unchanged.

Everything else is the frozen DFlash stack: sampled anchors, bidirectional
block mask, decayed block CE, the same slot convention and the same
teacher-forced capture.

## Status

Architecture + trainer ported, unit-tested on CPU, and recorded on
Qwen2.5-VL-3B: COCO 64-row smoke (token-exact 8/8, AL 1.123 vs init 1.000),
the canonical 2K matched run (`context_layer_ids: [1,17,33]` on the 5-layer
capture superset) — **AL 1.637, token-exact 128/128**, i.e. baseline parity
with the frozen DFlash 2K routing (1.630; +0.007 is inside the ~2% run-to-run
noise) — the layer-rich variant (**AL 1.670**), the fixed-prior and
shared-KV ablations (**1.637** / **1.579**) and the Qwen3-VL-8B reduced
validation (**AL 1.185**, token-exact 16/16). The ablations carry the
reading: the dedicated context K/V projections are worth **+0.058** at this
budget, while the learned fusion stays close to its distance prior
(fp32-master drift 0.011; freezing it changes AL by 0.0005). Quote the 2K
numbers as "dedicated context K/V + distance prior", not "learned fusion".
Full rows in `BENCHMARKS.md`; stage matrix in `ALGORITHM_MATRIX.md` /
`SUPPORT_MATRIX.md`.

## Config

Algorithm knobs are namespaced under `algorithm.dflare`; DFlash knobs apply
unchanged. The method consumes an **explicit layer subset** of the captured
feature store:

```yaml
drafter:
  capture_layer_ids: [1, 9, 17, 25, 33]   # ordered superset actually captured
algorithm:
  name: dflare
  dflare:
    context_layer_ids: [1, 17, 33]   # subset the fusion consumes (order = layout)
    separate_context_kv: true        # upstream: dedicated k_proj_target/v_proj_target
    fusion_trainable: true           # false = frozen distance prior (ablation)
    fusion_init: distance            # "distance" (upstream) | "uniform" (ablation)
```

Rules enforced at build time:

- `context_layer_ids` must be a subset of `drafter.capture_layer_ids` (which
  itself must contain the draft routing layers); a store/manifest that does not
  capture a consumed layer fails loud with a "recapture" hint before training.
- Training slices the concatenated store by explicit layer id; nothing is
  inferred from tensor shapes, and the consumed width must equal
  `len(context_layer_ids) * hidden_size`.
- `target_hidden_size == hidden_size` (the fusion residual assumes the draft
  width; enforced by the model).
- `layer_routing` / visual-fusion configs are rejected.
- The `algorithm.dflare` block uses a bounded schema: unknown keys fail loud
  with the supported-key list instead of being silently ignored.

Canonical configs:

- `configs/dflare/smoke_coco.yaml` — 64-row COCO pipeline smoke (captures the
  5-layer superset, consumes `[1, 17, 33]`, i.e. the smoke exercises the
  superset-slicing path on GPU);
- `configs/dflare/qwen2_5_vl_3b_2k.yaml` — canonical 2K run, matched to the
  frozen DFlash 2K baseline (same data/regen, consumes exactly the DFlash
  routing `[1, 17, 33]` from the shared 5-layer store);
- `configs/dflare/qwen2_5_vl_3b_2k_layers.yaml` — layer-rich variant on the
  same store (`context_layer_ids: [1, 9, 17, 25, 33]`; upstream fuses 9 layers
  of a 36-layer target, this repo's affordable capture is 5 — reported as a
  reduced layer set);
- `configs/dflare/qwen3_vl_8b_reduced.yaml` — Qwen3-VL-8B reduced validation
  on the shared reduced feature store.

Upstream recipe reference (AngelSpec `vllm_qwen3_8b_dflare.yaml`): lr 6e-4,
block 16, 512 anchors, decay 7.0, 9 target layers, 7 draft layers of a 4096-wide
target. This repo's canonical runs keep lr 6e-4 and the same decay but stay at
the zoo protocol (block 8, 256 anchors, 3 draft layers) so the comparison
against the other algorithms' runs is apples-to-apples; the layer count/block
size are config-level and the deviations are stated in any reported number.

## Train / eval / export

```bash
# first run of the pair owns the capture; later runs symlink
#   runs/<other>/data    -> ../qwen25vl3b_2k/data
#   runs/<other>/regen   -> ../qwen25vl3b_2k/regen
#   runs/<other>/features-> ../qwen25vl3b_dflare_2k/features
vlm-dflash train  --config configs/dflare/qwen2_5_vl_3b_2k.yaml
vlm-dflash eval   runs/qwen25vl3b_dflare_2k --config configs/dflare/qwen2_5_vl_3b_2k.yaml
vlm-dflash export runs/qwen25vl3b_dflare_2k
```

Training logs the fusion diagnostics per logging step:
`fusion_entropy_mean` (softmax entropy of the fusion distributions; a peaked
prior starts near 0, uniform pooling at `log T`), `fusion_entropy_min`,
`fusion_trainable`, `fusion_weight_abs_mean` and `fusion_prob_l1_drift`
(mean L1 distance of the fusion distributions from the prior). The metrics
read the optimizer's **fp32 master** copies (`set_master_parameters`, attached
by the training loop): the live parameter is bf16, where one AdamW step is
below the quantisation grid (ulp ~0.008 at the prior's logit scale), so
metrics read from the live parameter look frozen while the master keeps
moving. The first canonical 2K run predates this hook; its step-594 master is
extracted to `train/fusion_master_summary.json` (prob L1 drift 0.011 from the
prior, per-layer entropy `[0.496, 0.671, 0.396]` vs prior
`[0.441, 0.668, 0.439]` — layer 0 flattens, the deepest sharpens). At this
budget the prior still dominates the fusion weights; report the drift, not
"freely learned" fusion. Ablation checkpoints, same extraction at step 594:
shared-KV drift 0.0133 (the fusion moves further when the K/V are shared, but
AL drops — see Ablations), layer-rich 0.0094, fixed-prior 0.0 (the knob
verified). The export records `method: dflare`, the capture/consumed layer
lists and the `dflare` knob block.

## Ablations

- `separate_context_kv: false` — DFlash's shared context/draft K/V projections;
  isolates the projection change from the fusion change. **Measured: AL 1.579
  vs 1.637 matched** — the dedicated context K/V is the component with a
  measurable effect at this budget.
- `fusion_trainable: false` — freezes the distance prior; isolates what the
  learned fusion buys over the fixed relative-depth mixture. **Measured:
  AL 1.6369 vs 1.6374 trained (Δ0.0005; per-slot within ±0.002)** — no
  measurable gain at this budget.
- `fusion_init: uniform` — zero logits (average pooling) initialization,
  ablation-only, not the upstream default; report as such.
- `context_layer_ids` — the layer-rich run above; the same store serves both
  variants so the comparison differs only in the consumed subset.

## Known differences from upstream

See `docs/upstream/dflare.md` for the file-level audit. Summary: staged/online
feature stores with explicit layer slicing instead of a live target forward;
registered algorithm + model class instead of the `model_arch` switch; the VL
MRoPE/image plumbing is this repo's (upstream's recipe is text-only); the
ablation switches above are not upstream knobs.
