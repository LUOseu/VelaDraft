# Frozen DFlash reference (algorithm-zoo refactor)

`manifest.json` records the pre-zoo DFlash baselines that every algorithm-zoo
refactor must preserve (GOAL section 1: "freeze the current DFlash baseline").
The run directories live on the training server - `runs/` is gitignored - so
this manifest is the in-repo record of the numbers, semantics and gates.

Contents:

- `baselines`: frozen AL / per-slot / token-exact numbers with their run dirs,
  plus the training stats where recorded.
- `checkpoint_semantics` / `feature_semantics`: what `ckpt_final.pt`,
  `ckpt_init.pt` and a `vlm-dflash-features-v1` store contain, so a refactor
  cannot silently change their meaning.
- `regression_gates`: what a refactor must re-check (token-exact first).
- `remeasurements`: every post-refactor re-check of the frozen checkpoint with
  the file it was written to. A refactor is only "no regression" when the
  token-exact gate stays green and AL stays inside the ~2% bf16 noise band.

Re-running one gate (server, e.g. GPU 2), 8-prompt smoke or the full 128:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_dflash.py \
    --config configs/qwen2_5_vl_3b_2k.yaml \
    --run-dir runs/qwen25vl3b_2k \
    --num-prompts 128 --tag my_refactor_check
```

`scripts/eval_dflash.py` slices `rows[:num_prompts]` from the frozen val split,
so a 128-prompt run evaluates exactly the canonical prompt set. Do not compare
numbers across different `max_new_tokens` settings (see `not_comparable` in the
manifest).
