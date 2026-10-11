# Adding a drafter algorithm

A new algorithm is a plugin: one module under `vlm_dflash/algorithms/` that
implements `DrafterAlgorithm` and registers itself. Everything else (dataset
preparation, target-native regeneration, frozen hidden-state capture, the
training loop, checkpoint/resume plumbing, evaluation, export) is shared and
must not be forked.

## Steps

1. Create `vlm_dflash/algorithms/<name>.py`:

   ```python
   from .base import AlgorithmMetadata, DrafterAlgorithm
   from .registry import register

   class MyAlgorithm(DrafterAlgorithm):
       name = "myalgo"
       metadata = AlgorithmMetadata(
           name="myalgo",
           description="...",
           maturity={"architecture": "experimental", "training": "experimental",
                     "multimodal": "unvalidated", "runtime": "unsupported"},
           references={"paper": "...", "upstream": "...", "commit": "..."},
           capabilities={"image": True, "multi_image": False, "video": False},
           targets={"qwen25_vl": "unvalidated", "qwen3_vl": "unvalidated"},
       )

       def validate_config(self, cfg): ...
       def build_draft_config(self, contract, cfg): ...
       def build_trainer_module(self, draft_config, contract, cfg, embed_weight, lm_head_weight=None): ...
       def load_decoder(self, checkpoint_path, map_location="cpu"): ...
       def checkpoint_metadata(self, contract, cfg, draft_config): ...

   register(MyAlgorithm())
   ```

2. Import it from `vlm_dflash/algorithms/__init__.py`? No — add it to
   `registry._load_defaults()` (lazy import keeps the registry cheap).
3. Document it in `docs/algorithms/<name>.md`: canonical paper, canonical
   repo, commit inspected, ported modules, known gaps, license, validation
   status. Do not claim upstream parity you have not verified.
4. Add tests. Minimum bar per algorithm (see roadmap "Testing Philosophy"):
   - registry/metadata test (CPU);
   - synthetic tiny overfit: loss decreases, metrics finite, checkpoint
     saves/loads;
   - one real-target smoke (train a few steps, eval a few prompts).

## Contracts the shared stack relies on

- `build_trainer_module(...)` returns a module whose `` __call__(**batch) ``
  consumes the captured-feature batch (`input_ids`, `hidden`, `loss_mask`,
  `image_mask`, optional `multimodal_position_ids`) and returns metrics with a
  `.loss` tensor. Batch-level exceptions the loop may skip must be listed in
  the module's `skippable_errors` tuple.
- `load_decoder(...)` returns `(decoder, checkpoint_meta)`; the decoder must
  expose `spec_generate(target=..., input_ids=..., max_new_tokens=...,
  stop_token_ids=..., temperature=..., target_kwargs=..., return_stats=True,
  sequential_target_verification=..., strict_parallel_margin=...)` returning
  `(output_ids, stats)` with `stats["verify_steps"]` and
  `stats["accepted_tokens"]` (the evaluator derives AL and per-slot acceptance
  from those). If your decoder cannot honor sequential verification yet,
  declare it in metadata and make the eval path fail loud, do not silently
  approximate.
- Checkpoint metadata must include an `algorithm` tag (the base stack adds it
  automatically) so eval/export can refuse mismatched checkpoints.

## What not to do

- Do not change `vlm_dflash/eval/accept.py` AL or per-slot definitions per
  algorithm; algorithms are comparable only if the metric definition is shared.
- Do not add algorithm-specific branches to `data/`, `target/`, `eval/`,
  `export/`; if the shared interface cannot express something, extend the
  interface (with a default) instead of forking the pipeline.
