# Memory-bounded training scoring

VelaDraft offers two opt-in scoring optimizations. They preserve the selected
objective in real arithmetic, but changing GEMM shapes or the order of BF16
scaling can change numerical rounding. Existing defaults remain unchanged.

| Setting | Supported objective | Purpose | Tradeoff |
|---|---|---|---|
| `drafter.ce_chunk_size: 64` | DFlash / DFlare / DFly controlled CE | Project only supervised nonzero-weight slots, checkpoint head + CE in bounded chunks | Recomputes the head in backward; often slower |
| `drafter.ce_backward: precompute` with positive `ce_chunk_size` | Same controlled CE, frozen unsharded Linear head | Compute each chunk's weighted hidden gradient immediately; retain hidden-sized gradients | First-order only; upstream scalar scaling moves past a GEMM, so BF16 is not bitwise identical |
| `drafter.teacher_projection_dedup: true` | DFlash / DFlare / DFly `upstream_two_stage`; DSpark `upstream_dspark` | Within each consumer chunk, project unique `(sample, predecessor position)` teacher rows and restore order | Index/gather overhead; can increase transient memory and can be slower |

## Controlled CE

Start with a small explicit chunk size when vocabulary activations limit batch
size. Use the existing config for your target and data:

```bash
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml \
  --run-dir runs/dfly_chunked_ce \
  --set drafter.ce_chunk_size=64 \
  --set drafter.ce_backward=precompute
```

This requires the controlled-CE recipe. Remove an upstream recipe override if
your chosen configuration contains one. Changing the recipe changes the training
objective; compare execution modes only within the same recipe.

`ce_chunk_size=0` is the original dense path; `ce_backward=checkpoint` is the
default whenever chunking is enabled. Both modes retain the full vocabulary,
slot weights, normalization, accuracy and acceptance metrics. They omit the clean
anchor and zero-weight positions from the output projection. DFly's hidden
correction still runs before the output head, and its diagnostics are retained.
Packed-input boundary rules and sampled anchors remain in the existing trainer.

The checkpoint mode uses ordinary autograd and recomputes each projected chunk.
The precompute mode exploits the **frozen** head: within each chunk it obtains
the weighted hidden-state gradient with autograd, discards vocabulary-sized
intermediates, and stores only the hidden-sized gradient for backward. The
scalar loss and its NLL coefficients use the existing `DFlashDecayLoss` reducer,
including its optional FP32 weight mode. Backward multiplies by the incoming
scalar without modifying the saved gradient, so retained-graph reuse works.
Second derivatives are unsupported. Trainable, quantized, sharded and custom
heads are rejected in precompute mode. Heads must be ordinary Linear projections
without semantics-changing hooks/overrides; telemetry hooks must not alter values.

For `R` supervised rows, vocabulary `V`, hidden width `H`, and chunk size `C`,
precompute's scoring intermediates scale as `O(C V + R H)` rather than keeping
`O(R V)` vocabulary activations. This excludes model parameters, optimizer states,
backbone activations and padding in the original hidden layout. Measure the
complete update peak; this formula alone is not a GPU-memory guarantee.

This is integration of an established chunked/fused-loss technique, not a new
loss algorithm. The older `DFlashDecayLoss.forward_fused` helper always upcasts
CE to FP32 and does not return prediction IDs; the new trainer route preserves
the shipped CE dtype and all trainer metrics while compacting zero-weight slots.

## Teacher projection reuse

```bash
vlm-dflash train --config configs/dfly/qwen2_5_vl_3b_2k.yaml \
  --run-dir runs/dfly_teacher_reuse \
  --set algorithm.train_recipe=upstream_two_stage \
  --set capture.store_final_hidden=true \
  --set drafter.teacher_projection_dedup=true
```

Use a feature store that actually contains the final hidden state; toggling the
capture config cannot add missing data to an existing store.

Teacher identity is `(sample, label_position - 1)`, never just token ID or slot.
Different anchor-conditioned **student** states remain distinct. No teacher cache
survives a chunk or a step, so there is no stale cross-step distribution. The
complete original teacher row order is reconstructed before the original KL/TV
consumer. This also works with compacted block maps; normal sequence packing
continues to use its existing absolute position map.

The head must be an unmodified, frozen, unsharded `torch.nn.Linear`. The setting
is rejected for default CE and for algorithms with separate distillation paths,
including DFlash2 and JetSpec. It is distinct from DFlash2's existing
`teacher_projection_reuse` (sharing labels and KL projections).

Do not assume fewer projected rows means less memory or a faster update. During
restoration both unique and expanded teacher logits may coexist. Low overlap
adds index overhead without saving projection work. This option remains
experimental; the first shared-GPU probe did not establish a stable speedup.

## Measurement and reproduction

`execution_switches()` records the settings in the existing training reports.
The existing workload counters include `student_head_rows`, `teacher_head_rows`
and `scored_rows`. New teacher counters are `teacher_requested_rows` and
`teacher_reused_rows`. Head calls/rows include backward checkpoint recomputation;
they are work counts, not unique dataset-token counts.

The paired probe uses fixed anchors, the real trainer, BF16 live weights,
FP32 AdamW masters and `lr=0`. It executes optimizer work but intentionally does
not train a model. It records every timed observation, full-update peaks, GPU
occupancy snapshots, source hashes, same-code numerical floor, every parameter's
gradient coverage and BF16 differences. Synthetic target features and a single
backbone layer make it an execution probe, not a quality or pipeline benchmark.

```bash
python scripts/benchmark_teacher_projection.py \
  --device cuda:0 --optimization ce --ce-chunk-size 64 \
  --ce-backward precompute --anchors 128 --strides 1 --stages 0 \
  --out runs/scoring/ce.json

python scripts/benchmark_teacher_projection.py \
  --device cuda:0 --optimization teacher --strides 1 8 --stages 0 1 \
  --out runs/scoring/teacher.json
```

`--memory-fraction` defaults to 0.20; change it explicitly on a dedicated GPU.
Timing on a shared GPU is provisional. Compare both with the existing compiled
recipe baseline (`--compile-tv-kl`) where applicable. Do not transfer timings
between hosts or label these probes as time-to-quality results.

Measured artifacts and reviewer-driven next experiments:
[2026-10-31 report](../research/train_cost_20261031/REPORT.md).
