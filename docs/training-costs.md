# Training costs (measured)

What one drafter-training pipeline costs on the reference hardware, assembled
from the recorded runs behind `BENCHMARKS.md` (the 2K algorithm-zoo program
plus the DFly two-stage recipe run). Scope: Qwen2.5-VL-3B-Instruct target, one
NVIDIA RTX A6000 48 GB per job, bf16, offline feature store, `verification:
sequential`, greedy decoding.

Protocol (the matched 2K program):

- prepare: LLaVA-Instruct pool -> 2000 train + 128 val (seed 42);
- regen: 2000 prompts, greedy, <= 160 new tokens, keep answers >= 128 tokens
  (batch 8) -> 1050 kept;
- capture: seq 2048, shard 32, bf16 -> one feature store per layer set
  (1050 rows, 35 shards);
- train: 792 optimizer steps (6 epochs x 132 steps/epoch; micro-batch 8 x
  grad-accumulation 1 = **8 samples per optimizer step** — the 16-sample batch
  belongs to the overfit-sanity gate, not the main loop), plus a 200-step
  overfit-sanity gate;
- eval: 128 held-out prompts x <= 128 new tokens, sequential verification,
  token-exact gate, `ckpt_final` + `ckpt_init` + target-only reference.

## Pipeline stages (one full 2K run)

| Stage | Wall | What dominates / evidence |
| --- | ---: | --- |
| prepare | ~20 s | pool 64969 -> 2000/128, digest `5dfc5d7e9aa5` |
| regen (target generate) | ~23.5 min | 2000 prompts at batch 8 - the single largest target-compute block |
| capture | ~2.6 min | 1050 samples, seq 2048 -> 10.04 GB (9.4 GiB) store (3 layers + final hidden); 5-layer store ~3.5 min / 12.4 GB (11.5 GiB) |
| train | 9.8-24 min | 792-step main loop (`train_summary.wall_seconds`); the main train loop alone is the 10-24 min range, sanity adds ~3-5 min |
| eval | ~28 min | 128 prompts x 128 tokens x (final + init + target-only) |
| **end-to-end** | **~65-80 min = 1.1-1.3 GPU-h** | one GPU wall-to-wall, including data generation and capture |

Stage walls come from `provenance.json`/file timestamps of the 2026-10-11 base
run and the `train_summary.json` / `eval_result.json` records; exact per-job
rows are in `docs/plan/RUN_LOG.md`.

## Drafter training by algorithm (matched 792-step protocol)

| Algorithm | Wall (s) | s/step | Peak GPU | Final loss | AL |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFlash | 586 | 0.74 | 29.5 GB | 3.781 | 1.630 |
| DFlash2 (experimental objective) | 1359 | 1.72 | 31.1 GB | 1.415 | 1.954 |
| DSpark | 710 | 0.90 | 38.1 GB | 3.531 | 1.735 |
| Domino | 775 | 0.98 | 38.4 GB | 2.203 | 1.764 |
| JetSpec | 1234 | 1.56 | 29.7 GB | 3.100 | 1.599* |
| DFlare | 1444 | 1.82 | 30.2 GB | 3.641 | 1.637 |
| DFly | 966 | 1.22 | 31.6 GB | 1.680 | 1.682 |

\* JetSpec AL = causal-mask serving variant (`eval_result_jetspec_2k_causalmask.json`);
the in-block default scored 1.338. AL = 128 prompts / 128 tokens, sequential
verification, token-exact 128/128 for every row. **Peak GPU =
`torch.cuda.max_memory_allocated` in 10^9 bytes (allocated, not reserved)** -
compare it with other allocated readings, not with reserved-memory numbers.

The walls are wall-clock on a *shared* node: identical protocols vary by up to
~1.7x between runs (DFlare 1444 s vs the same-config `fixed_fusion` ablation at
864 s). Dividing an interfered wall by its step count does **not** remove the
interference, so `s/step` inherits the same noise: it is a within-host pacing
figure, not a fair ranking device. Treat the fastest wall per algorithm as an
estimate of uncontended pace on this node. All 2K runs train the same 8
samples per optimizer step (micro-batch 8 x accumulation 1) over one shared
store, so there is no per-algorithm data cost.

## DFly upstream recipe (canonical run, 792 steps)

The P0 recipe deliverable, measured against the controlled_ce DFly record on
the same store, seed and 792-step protocol; the only knob is
`train_recipe: upstream_two_stage` (cold-start fraction 0.5, stage boundary
at step 396).

| Run | Wall (s) | s/step | Peak GPU | Final loss | AL |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFly `controlled_ce` (zoo record) | 966 | 1.22 | 31.6 GB | 1.680 | 1.682 |
| DFly `upstream_two_stage` (recipe 0.5) | 2646* | 3.34* | 46.5 GB | 1.977 | **1.839** |

\* Writeback-storm inflated: the uncontended pace was ~1.9 s/step (steps
1-200; the recovered tail matches at ~1.8), with steps ~200-620 running at
4.1-5.7 s/step during a host-wide stall. On a quiet host expect ~25 min for
the main loop; the residual ~1.5x over controlled_ce is the recipe's own
cost (the LK cold-start stage plus the e2e-TV term's extra forward work).

Stage detail (first-10 -> last-10 means): stage 0 (steps 1-396) LK/D-PACE
loss 8.31 -> 1.25; stage 1 (397-792) CE 2.00 -> 0.98, e2e TV 0.79, slot-0
accuracy 0.47 -> 0.67. Overfit sanity passed (CE drop 0.926, slot0 0.9986),
gradient coverage 42/42, `nan_seen=false`; the 46.5 GB peak is the sanity
pass (one 16-sample overfit batch) - the main loop peaks at 28.6 GB.

The recipe buys **+0.157 AL (+9.3%)** over controlled_ce: every slot accepts
more (`[.440,.207,.103,.051,.024,.014,.007]` vs
`[.384,.160,.073,.037,.019,.008,.004]`) and the draft step is slightly
cheaper in the same harness (0.86 s vs 0.94 s per prompt). Eval cost at 128
prompts: 4.50 s wall/prompt (prefill 0.108 / draft 0.862 / verify 3.471),
0.80x target-only; init control 1.003; both checkpoints token-exact 128/128.

## Acceptance eval cost (128 prompts / 128 tokens)

| Algorithm | AL | Wall/prompt | prefill | draft | verify | Speedup vs target-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash | 1.630 | 4.14 s | 0.11 s | 0.44 s | 3.54 s | 0.87x |
| DFlash2 | 1.954 | 4.43 s | 0.11 s | 0.63 s | 3.60 s | 0.82x |
| DSpark | 1.735 | 4.10 s | 0.11 s | 0.47 s | 3.46 s | 0.86x |
| Domino | 1.764 | 4.30 s | 0.10 s | 0.49 s | 3.62 s | 0.84x |
| JetSpec | 1.599 | 4.13 s | 0.11 s | 0.51 s | 3.45 s | 0.85x |
| DFlare | 1.637 | 4.32 s | 0.11 s | 0.47 s | 3.68 s | 0.85x |
| DFly | 1.682 | 4.70 s | 0.11 s | 0.94 s | 3.59 s | 0.77x |

This is the batch-1 offline harness, not a serving engine: the sequential
verification path forwards the target for every speculative block, so verify
dominates and the speedup stays below 1x at short output budgets. The drafter
earns its cost through acceptance (DFly's higher draft time is visible as the
lowest speedup in this harness). A real serving integration is the roadmap's
serving bridge; see `docs/serving-bridge.md`.

## Disk

| Artifact | Size | Note |
| --- | ---: | --- |
| Target model (3B / 8B) | 7.1 GB / 17 GB | per target family |
| Feature store, 3 layers + final hidden | 10.04 GB (9.4 GiB) | 1050 x seq 2048 bf16, 35 shards |
| Feature store, 5 layers | 12.4 GB (11.5 GiB) | serves DFlare/DFly routing |
| Run checkpoints | 1.06 GB init + 0.53 GB final (+ ~3.7 GB transient `ckpt_last`) | `ckpt_init` is kept for the parity eval |
| Run directory (train + eval + export, store excluded) | ~2-6 GB | - |

Run directories symlink their `features/` to the owning store; the store is the
one large shared artifact, and one store serves every algorithm whose consumed
layers are a subset of its captured layers (feature reuse is verified in
`RELEASE_VALIDATION.md`).

## Where the target compute goes

- **Offline (the reference protocol):** one greedy generate per prompt during
  regen (2000 x <= 160 tokens, batch 8) plus one teacher-forced forward per
  kept sample at capture; training never runs the target; eval forwards it per
  prompt for the three reference configurations.
- **Online capture** (`docs/online-training.md`) trades the store for
  per-epoch target forwarding at training time; batched, overlap-friendly
  production is the WP4 item in `docs/plan/`.
- Iterating on a new algorithm against an existing store costs only the train
  stage (~10-24 min); the full ~1.1-1.3 GPU-h applies when the data slice,
  capture layers or target family change.

Validation overhead for a training-objective port (the WP1/E1 DFly-recipe
set: select-parity, kill/resume, export/reload, twice across two sessions) was
~0.6 GPU-h in total - small next to the pipeline it guards.

## Caveats

- Shared-node effects are real: the recorded walls range up to ~1.7x for
  identical protocols, and a host under heavy writeback pressure can slow GPU
  jobs far beyond that (observed during this program). Re-measure on a quiet
  host before comparing across days.
- bf16 capture/training reproduce the protocol, not the bits (sub-2% AL
  noise); use `s/step` and peak memory, not single wall times, for capacity
  planning.
- Eval numbers are the conservative token-exact path; see
  `docs/verification-modes.md` for the experimental fast paths.
