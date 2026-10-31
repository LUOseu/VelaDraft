# Training scoring iteration — 2026-10-31

Base: VelaDraft `origin/main` at `9404b93`; implementation branch:
`codex/diffusion-train-cost-20261031`.

The completed platform change adds opt-in scoring execution paths, tested against
existing loss/gradient/metric semantics. The measured benefits are lower allocated GPU memory and faster updates on one
guarded-exclusive synthetic padded workload. A general training-speed or
end-to-end cost reduction is **not yet established**.

## Implemented

- `drafter.ce_chunk_size`: remove non-supervised/zero-weight rows before the head
  and evaluate full-vocabulary CE in checkpointed chunks.
- `drafter.ce_backward=precompute`: for frozen plain Linear heads, compute each
  chunk's weighted hidden VJP once and retain hidden-sized gradients. Higher
  derivatives and sharded/trainable/custom heads are unsupported.
- `drafter.teacher_projection_dedup`: reuse identical sample/position teacher
  projections within each existing loss chunk; original consumer order remains.
- Config rejection for incompatible objectives; execution switches and workload
  counters; CPU regressions, paired GPU probe and raw-result summarizer.

See [user guide](../../docs/training-scoring.md) for exact supported methods and
commands. All execution options are off by default. Inference architecture, export format and loading paths
are unchanged; learned weights can differ through numerical training effects.

## Evidence

[Generated timing/memory table](summary.md) and [machine-readable summary](summary.json)
contain every GPU cell, including negative results. The original JSON files are
canonical; the summary is regenerated from them.

On the synthetic two-sample batch with 128 versus 16 kept anchors, 151,936-token
vocabulary and hidden width 2,048, precomputed CE with chunk 256 reduced projected
student rows from 2,048 to 1,008. Peak allocated memory fell from **3.058 GiB to
1.769 GiB (about 42%)**. The update-time median ratio was 1.526, but its paired
bootstrap interval includes 1 and is extremely wide under shared-GPU interference.
**The shared-GPU run does not establish a speedup.** The padding/compaction and
precompute contributions are not isolated by this combined arm; row-only and
external fused-loss ablations remain necessary.

A subsequent run used GPU 2 after it became idle, checking exclusive compute
occupancy before and after every arm without pausing any job. On the same padded
shape, baseline/optimized medians were **54.07 / 38.73 ms**, ratio **1.396** with
a paired-bootstrap 95% interval **[1.350, 1.411]**, over 15 pairs of four updates
per arm. Allocated peaks remained **3.058 / 1.769 GiB**. This supports a scoped
update-time improvement for this synthetic combined arm; it does not resolve
real-workload prevalence, component attribution, quality or pipeline cost.

The same exclusive protocol with two unpadded samples measured **54.82 / 51.03 ms**,
ratio **1.074 [1.062, 1.097]**, with the same allocated peak reduction. The smaller
speed gain is retained as the control for padding-dependent benefits.

The unpadded chunk-64 checkpoint probe reduced allocated peak by about 27% but
increased median update time. The precompute probe used the same low-memory
shape and avoided recomputing the head, yet did not establish a speed gain on
that shape. Teacher dedup likewise did not establish stable full-update gains.
These cases remain in the table rather than being filtered out.

Numerical checks compare identical inputs, fixed anchors and unchanged model
weights (`lr=0`), including a repeated baseline and complete trainable-gradient
coverage. The padded BF16 precompute run had equal scalar loss and metric counts,
with relative gradient L2 drift about 0.00514. This is a first-order numerical
check, not bitwise equality or a convergence/quality certificate.

## Protocol and limitations

- NVIDIA RTX A6000, PyTorch 2.10.0+cu128, BF16 live parameters, FP32 AdamW masters.
- Full update probe includes forward, backward, gradient copy/clipping and AdamW;
  synthetic features, one DFlash backbone layer, no target-model forward.
- `lr=0` deliberately preserves identical weights. No multi-step learning or
  held-out quality result is reported.
- Each pair alternates order; CUDA synchronization brackets timings. Allocated
  and reserved peaks are recorded separately. Six timing pairs for initial
  probes, ten for the shared padded probe and fifteen for exclusive probes. Bootstrap resamples pairs, not seeds.
- Initial probes shared GPU 3. Later probes used idle GPU 2 with process guards.
  The guard aborts rather than stopping other processes; it is sampled occupancy
  evidence, not a scheduler reservation. The host remains shared.
- In the initial shared-GPU probes, very wide intervals show that these runs cannot support
  a paper speedup claim. Emptying this process's allocator between arms does not
  remove interference from other contexts.
- The first teacher pilot predates the stronger benchmark gradient-key-set gate;
  later CE probes use it. Separate CPU trainer tests check every parameter.
- An initial harness attempt failed because its trainer buffer was left on CPU.
  `.to(device)` fixed the probe; no result from that failed attempt is used.
- Independent code review found a missing-gradient benchmark gate and a
  noncontiguous VJP-buffer issue. Both received regressions and were fixed.
- No existing GPU task was stopped, killed, restarted or otherwise altered.
  Experiments used a separate directory and a per-process 20% memory limit.

## Recompute

```bash
python scripts/summarize_training_scoring.py \
  research/train_cost_20261031/a6000_bf16_pilot.json \
  research/train_cost_20261031/a6000_bf16_ce.json \
  research/train_cost_20261031/a6000_bf16_ce_precompute.json \
  research/train_cost_20261031/a6000_bf16_ce_padded256.json \
  research/train_cost_20261031/a6000_exclusive_ce_padded256.json \
  research/train_cost_20261031/a6000_exclusive_ce_unpadded256.json \
  --out-prefix research/train_cost_20261031/summary
```

The benchmark command/shape, raw samples, source hashes, parity and device
snapshots are stored in each JSON. CPU results are in `cpu_tests.xml` (211 focused tests) and
`full_cpu_tests.xml` (1,008 passed, 6 integration skips); independent
review and additional numerical checks are in [CODE_REVIEW.md](CODE_REVIEW.md).

## Paper direction

The [ML Systems proposal](FINAL_PROPOSAL.md) separates delivered engineering from
unverified novelty. The local patch is insufficient as a primary research
contribution: the next decision gate is real-workload profiling, a strong
objective-matched baseline, row-only/fused-loss ablations, and a validated
resource-aware execution policy. Equal-quality GPU-hours and full pipeline
amortization remain unmeasured.

Local documentation checks passed with `PYTHONPATH=.` (the reused environment otherwise imports an older editable checkout). The NPU-readiness aggregate passed 137 CPU tests, and the existing 17-profile / 233-artifact evidence validation passed. No NPU hardware claim is added.

A wheel was built with `python -m build --wheel --no-isolation`; importing the wheel and building its CLI parser outside the checkout passed. This was a wheel import smoke using the existing dependency environment, not a fresh-dependency installation.
