# A100 runbook — VelaDraft convergence matrix

Executable checklist for the full benchmark round on the user-provided A100 host.
Everything here reuses the in-repo framework: `scripts/train_dflash.py`,
`scripts/eval_dflash.py`, `scripts/benchmark.py`, `vlm_dflash/presets`, and the
convergence launcher `research/convergence/a100/run_matrix.py`.

## 0. What this produces (deliverable mapping)

| Output | Feeds |
| --- | --- |
| per-leg `train_summary.json` (wall, peak mem, regroup counters) | effect + cost table (GPU-hours, memory) |
| per-leg `eval/eval_result.json` + `eval_samples.jsonl` (held-out val split) | quality table (AL, acceptance, per-slot) |
| `benchmarks/reports/benchmark_<run>_<workload>.json` | applicability boundary (4 public workloads) |
| `summary.{json,csv,md}` | aggregate tables for the paper |
| `matrix_state.jsonl` | runbook reproducibility (exact commands + SHAs) |

## 1. Host prerequisites

- 8× A100 80GB, ≥ 400GB free local disk for run dirs, ≥ 200GB for data.
- Repo at the convergence branch tip; record `git rev-parse HEAD`.
- Python env with `torch`, `transformers`, `flash-attn`/flex path per
  `docs/` setup notes; set `VELADRAFT_PYTHON` to its interpreter.
- Target model assets: Qwen2.5-VL-3B-Instruct (see `assets/` docs) and the
  Qwen3-VL-8B variant if the 8B row is in scope.
- Env exported before running:

```bash
export VELADRAFT_REPO=/path/to/VelaDraft
export VELADRAFT_PYTHON=/path/to/venv/bin/python
export VELADRAFT_RUNS_ROOT=$VELADRAFT_REPO/runs/a100
export VELADRAFT_DATA_ROOT=/data/veladraft
export VELADRAFT_TARGET_MODEL=/models/Qwen2.5-VL-3B-Instruct
```

## 2. Data build (train / val / test isolation)

1. Training slice (2K, Qwen2.5-VL-3B recipe): follow `configs/qwen2_5_vl_3b_2k.yaml`
   (LLaVA-corpus slice; capture once, reuse via the feature store).
2. Held-out val split: `$VELADRAFT_DATA_ROOT/qwen25vl3b_2k/val_heldout.jsonl`
   — prompts disjoint from the training slice, never trained on. The runner
   refuses a matrix whose `train_store` path equals `val_split`.
3. Test workloads (never trained): export the public sets once.

```bash
$VELADRAFT_PYTHON scripts/prepare_benchmark.py --workload textvqa --source <dl> --out benchmarks/data/textvqa
$VELADRAFT_PYTHON scripts/prepare_benchmark.py --workload docvqa  --source <dl> --out benchmarks/data/docvqa
$VELADRAFT_PYTHON scripts/prepare_benchmark.py --workload chartqa --source <dl> --out benchmarks/data/chartqa
```

`caption_general` reads the run's own val split (`run_val` manifest), which
keeps the fourth workload inside the held-out boundary.

## 3. Tracks

- **controlled** — every drafter trains with the identical config
  (`controlled_config`, 2000 steps, seeds 42/43, same store/schedule). This is
  the paper's like-for-like comparison.
- **native** — each drafter uses its own `configs/<drafter>/qwen2_5_vl_3b_2k.yaml`
  recipe (its documented best setup). Reported alongside, never mixed into the
  controlled table.

Both tracks run the same `stages: [train, eval, benchmark]` on the same
val/test isolation. Seeds are paired across drafters (same seed = same init
RNG stream and data order when the recipe allows).

## 4. D1 regroup factor (systems optimization)

- Factor levels: `off` (shipped canonical layout) and `grid2`
  (`train.regroup.mode=grid, sub_batch=2, fp32_weights=true,
  identity_bypass=true`).
- Default-in-off everywhere; `grid2` legs are enabled only where
  `regroup_support=true` (dflash today). `dflare` carries
  `regroup_adaptation=true`: it shares the dflash trainer core, but the
  per-algorithm parity gates must pass before its grid2 legs are trusted:

```bash
# 3-step adaptation smoke on the real store (bitwise identity + machinery path)
bash research/convergence/a100/regroup_adaptation_smoke.sh <drafter> <gpu> <config> <store>
# gate 1: canonical/8 + bypass must be bitwise identical to regroup-off
# gate 2: grid/2 must run (real machinery path) and eval token_exact=true
# only then flip regroup_support=true for that drafter in matrix.json
```

- The runner never edits supervision: regroup legs share seed, data order and
  anchor tables; `fp32_weights` + `identity_bypass` are the validated wave-2
  semantics. Any gate failure = keep the drafter at `off`.

## 5. Run sequence

```bash
# a) plan + validation (no GPU): prints every command, est. GPU-hours
run_matrix.py dry-run --verbose

# b) 2-entry pilot on one GPU (validates the full chain end-to-end)
run_matrix.py run --gpus 0 --limit 2

# c) full matrix (one serial worker per GPU)
run_matrix.py run --gpus 0,1,2,3,4,5,6,7

# d) after interrupts: resumes ckpt_last legs exactly, re-runs parked stages
run_matrix.py resume --gpus 0,1,2,3,4,5,6,7

# e) aggregation
run_matrix.py summarize --out-dir research/convergence/evidence/a100
```

- Entry ids: `<drafter>__<track>__s<seed>__rg<mode>` (32 entries ≈ 37 GPU-h by
  the A6000-calibrated cost model; expect less on A100).
- Guard: per-GPU `util ≤ 5%` and `free ≥ 40GiB` and `disk ≥ 15GB`, 30s polls,
  1h timeout → the entry parks with `guard_timeout` in `matrix_state.jsonl`.
- Failures park the entry (later stages skipped); rerun `resume` after fixing.
- `--only` (regex) and `--limit` shrink any run; state makes all of this safe
  to repeat.

## 6. Algorithm adaptation notes

- `dflash`: primary; regroup verified on this round's W3 store (bitwise).
- `dflare`, `dfly`: shared trainer core. dflare smoke (2026-10-15, real 2k store):
  canonical/8+bypass bitwise PASS, grid/2 refused by the auxiliary-metrics
  gate (`train.py:321`). Adapting dflare = make its aux-metric accounting
  group-aware, then re-run the smoke gates. dfly stays `off` until gated.
- `jetspec`: serve/eval with the causal-mask path it was trained with (the
  canonical zoo used `eval_result_jetspec_2k_causalmask.json`); benchmark
  reports keep the default protocol.
- `dspark`/`domino`/`dflash2`: standard dflash-core path; no regroup.

## 7. Post-run merge into the convergence evidence

```bash
scp -r <a100>:.../runs/a100/*/eval/eval_result.json   research/convergence/evidence/a100/legs/
scp -r <a100>:.../runs/a100/*/eval/eval_samples.jsonl research/convergence/evidence/a100/legs/
scp    <a100>:.../summary.{json,csv,md}               research/convergence/evidence/a100/
```

The convergence quality gates (`analyze_quality.py`) consume the per-leg
artifacts exactly as the W3 legs did; systems numbers come from
`train_summary.json` + `benchmark_*.json`.
