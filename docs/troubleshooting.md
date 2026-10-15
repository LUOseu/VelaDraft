# Troubleshooting

Short answers for the failures and confusions that actually come up. Each one
has a symptom, a place to look, and a next step - no destructive shortcuts.

## "There is no model at the path I passed" / the run refuses to start

- **Check:** `vlm-dflash doctor --target <dir>` (exit 0 with `ok=True` means the
  adapter, contract and file layout are readable).
- **Why:** configs run with `local_files_only: true`; the pipeline never
  downloads a target for you. An interrupted download leaves a directory that
  *looks* complete until the missing shard is read.
- **Next step:** compare the directory against the Hub file list (a complete
  Qwen2.5-VL-3B checkout has `config.json`, tokenizer/processor files and all
  `*.safetensors` shards), re-run the download client, then `doctor` again.
  Setup details: [getting-started.md](getting-started.md).

## Training spends minutes before any step is logged

- **Check:** the log lines - `regen: ...` runs a greedy target generation per
  prompt before training; recorded cost at the 2K protocol is ~23.5 min for
  2000 prompts (batch 8), plus a few minutes of capture
  ([training-costs.md](training-costs.md)).
- **Why and what is supervised:** the dataset's `answer` text is prompt
  material; the training labels are the target's regenerated answers. With
  multi-turn rows only the last assistant turn is regenerated and supervised by
  default (`last_turn_only`); see [data.md](data.md).
- **Next step:** watch `regen/regen_stats.json` for kept/skipped counts; the
  `--limit` flag shrinks a first explore run.

## "My 40 rows became 30" or the validation split is empty

- **Check:** `data/manifest.json` (`num_train` / `num_val`) and
  `regen/regen_stats.json` (`num_kept` + skip reasons).
- **Why:** `--limit` counts **training** rows (`val_size` is drawn on top), and
  regen drops rows whose regenerated answer is shorter than
  `regen.min_response_tokens` (128 in the long-answer recipes - nearly all COCO
  captions). If the pool is not larger than `limit + val_size`, val can be
  empty (prepare logs a warning).
- **Next step:** add rows, lower the slice, or use a recipe with a small
  `min_response_tokens`. Rules: [data.md](data.md).

## Online mode refuses `train.num_workers != 0` or `--features`

- **Check:** the error text names the constraint; both are also listed under
  *Boundaries and failure behavior* in
  [online-training.md](online-training.md).
- **Why:** the frozen target lives in the trainer process and cannot be copied
  into DataLoader workers; and a pre-captured store belongs to the offline
  path (online extracts live).
- **Next step:** set `train.num_workers: 0` (default) and drop `--features`;
  or switch to `--mode offline` to reuse a store.

## GPU out of memory

- **Check:** `<run_dir>/train/train_summary.json` -> `peak_gpu_mem_gb`
  (allocated, not reserved; the env notes quote *reserved* numbers separately).
- **Why - what is on the GPU:** offline runs keep only the drafter (+ a batch)
  resident; online runs keep the whole target (recorded resident ~7.5 GB for
  3B, ~17.5 GB for 8B). The overfit-sanity phase can peak well above the main
  loop (recorded: DFly recipe 46.5 GB sanity vs 28.6 GB main loop).
- **Next step:** measure first, then shrink `train.micro_batch_size` /
  `capture.seq_length`, or run online with the target on a second card
  (`--set target.device=cuda:1 --set train.device=cuda:0`, recorded smoke).
  The 24 GB preset is a suggestion, not a validated card profile.

## `--resume` is refused ("plan mismatch", "feature digest", seed/step changes)

- **Check:** the error names the specific mismatch; the gates are listed under
  *Exact resume* in [online-training.md](online-training.md) and
  [training.md](training.md).
- **Why:** exact resume re-issues the original trajectory; changing seed,
  steps, recipe, batch geometry or the feature source changes it, so the run
  refuses instead of silently branching. A refused resume leaves the run dir
  byte-identical.
- **Next step:** repeat the original command (same config + flags) and add
  `--resume`; if the plan really changed, start a new `--run-dir`.

## Online mode wrote no feature store but the disk still filled up

- **Check:** `du -sh <run_dir>/*` - `data/` (copied images), `regen/`,
  `train/` (three checkpoints: init + last + final), `eval/`, logs.
- **Why:** "no feature store" only removes the multi-GiB `features/` shards;
  everything else in a run persists.
- **Next step:** keep `config.json`, `contract.json`, `ckpt_last.pt` and the
  stage manifests if you may resume; anything you delete after that is a
  deliberate trade of resume-ability for space - not a cleanup ritual.

## What do loss / AL / token-exact / speedup actually mean?

- **Loss** (block CE for most methods): training diagnostic, not a quality
  score across algorithms.
- **Acceptance length (AL)** = new tokens per target verification step under
  **sequential (token-exact) verification**; it is *not* a wall-clock speedup.
- **Token-exact** = speculative output matches the target's own greedy decoding
  for the evaluated prompts; every recorded table gates on this.
- **Wall-clock speedup vs target-only** depends on the harness: the in-repo
  sequential verifier forwards the target per speculative block, so recorded
  speedups stay below 1x at short output budgets
  ([training-costs.md](training-costs.md)); `parallel`/`strict_parallel` are
  throughput diagnostics, never correctness
  ([verification-modes.md](verification-modes.md)).
- **Next step:** quote numbers with their protocol - see
  [../BENCHMARKS.md](../BENCHMARKS.md).
