# WP6 report - first non-Qwen VLM family (SmolVLM2-2.2B-Instruct)

Status: **ENGINEERING DONE (single-image smoke validated), gaps registered**
Run: `R-20261014-17` (0.53 GPU-h incl. probes/diagnostics; 1 GPU, A6000 48 GB)
Code: family adapter + hooks (`ef07b41`, `0f45fa1`, plus fixes `c2aacbb`,
`8d4cc93`, `64ba517`, `be0be8c`, `13cb205`); 14 new CPU tests in
`tests/test_smolvlm_adapter.py` (10 in the adapter commit + 4 with the fixes),
full suite 367 collected / green.

## Family facts (probed on the real checkpoint, `482adb53...`)

- `model_type: smolvlm`, `SmolVLMForConditionalGeneration`; text backbone Llama
  24 layers / hidden 2048 / 32 heads / head_dim 64; **1-D sequence positions**
  (no `compute_3d_position_ids`, no `mrope_section`) - the family trains with
  `drafter.spatial_rope_enabled=false` and `rope.sections=None`.
- `tie_word_embeddings: false`; vocab 49280, embedding rows 49280; tokenizer
  base vocab 49152. `<image>`=49190, `<fake_token_around_image>`=49189,
  `<global-img>`/`<row_i_col_j>`=49152-49188, reserved rows
  `<|reserved_special_token_0..87|>`=49191-49278, `<end_of_utterance>`=49279.
- Image token run for one image: **17 tiles x 81 = 1377** contiguous
  `<image>` ids, separated by structure markers (`<row_i_col_j>`,
  `<global-img>`, `<fake_token_around_image>`) and newlines; processor emits
  `pixel_values` + `pixel_attention_mask` (no `image_grid_thw` /
  `mm_token_type_ids`); `processor_config.json image_seq_len: 81`;
  preprocessor `size.longest_edge 1536`, `max_image_size.longest_edge 384`.
- `pad_token_id` in config is **128002 (outside the 49280-row embedding
  table)**; the tokenizer's own pad is `<|im_end|>` (in range).

## Delivered

- `vlm_dflash/adapters/smolvlm.py`: family adapter - token resolution + config
  cross-check, reserved-row mask policy, 1-D rope resolution (`sections=None`,
  rejects YAML MRoPE defaults), tile token counts (`tiles x 81`), no processor
  pixel injection, batched-capture inputs (`pixel_values` /
  `pixel_attention_mask`), one-image span rule that bridges the tile markers
  and whitespace, nested per-text batch image argument for the regen processor.
- Contract/factory/capture/eval/regen/visual-check family hooks; `factory`
  accepts `rope.sections=None` (spatial rope off), capture clamps an
  out-of-range config pad id to the embedding table.
- `configs/smoke_smolvlm.yaml` + `scripts/make_wp6_smoke_data.py` (16 train + 4
  val synthetic single-image rows, v1 row format).

## R-20261014-17 result (PASS, engineering gate)

Command chain: `vlm_dflash train --config configs/smoke_smolvlm.yaml` then
`vlm_dflash eval runs/e6/smoke_smolvlm --config configs/smoke_smolvlm.yaml`
(data `~/recipe-smoke-config/e6_assets`, target = the HF cache snapshot).

- prepare 20 rows -> 16/4; regen **16/16 kept** (greedy, `min_response_tokens 8`;
  answers 8-19 tokens - 2-3-token answers cannot anchor `block_size 4`).
- capture 16 samples, seq 2048 (rows 1441-1452): image-token check
  **1377/1377**, `visual_path_check` **passed** (replay bitwise equal, span and
  downstream changed) - first non-Qwen family through that gate.
- train 32 steps (`block_size 4`): final loss 3.66, acc 0.12, train AL 1.11;
  drafter visual check `passed=True`.
- eval (sequential parity, 4 val prompts, 32 new tokens, `ckpt_final` +
  `ckpt_init`): **token-exact 4/4 for both**; AL 1.182 / 1.083; parallel
  diagnostic 1/2 exact (the flip sits at a 0.0 top-2 margin - the same bf16
  near-tie phenomenon recorded in R-20261014-16).
- No speed claim: at this scale the trained draft's wall includes first-call
  kernel compilation (1.41 s/sample vs 0.05 s for the init checkpoint's same
  architecture); speedup numbers in `eval_result.json` are diagnostic only.

## Registered gaps / limits

1. **Schema-v2 (multi-turn) rows**: the multi-turn regen's full-render check
   fires on SmolVLM - the template renders `Assistant:` for the generation
   prompt but `Assistant: ` inside the assistant message body, so incremental
   ids drift from one full render. The WP6 smoke therefore uses v1 single-image
   rows; multi-turn SmolVLM support needs a template-aware render/verify pass.
   Multi-image is also unimplemented for this family (the span rule assumes one
   image region per text-separated block; per-image grouping by tile counts is
   the follow-up).
2. **Online mode** was not exercised for this family; the tokenizer threading
   + pad-id clamp landed preventively (CPU-tested), no GPU run.
3. Preset configs inject Qwen pixel budgets (`target.image_min_pixels`); inert
   for tile families (no processor injection) and cleared in the smoke config.
4. Smoke-scale only (16+4 synthetic rows, 32 steps): no quality claim; the
   evidence supports "the stack runs a non-Qwen family end to end", nothing
   stronger.

## Reproduce

```
python scripts/make_wp6_smoke_data.py --out data/e6_smoke --train 16 --val 4
VLM_DFLASH_SMOLVLM_TARGET=<snapshot> VLM_DFLASH_E6_SOURCE=.../source.jsonl \
  VLM_DFLASH_E6_IMAGES=.../images python -m vlm_dflash.cli train \
  --config configs/smoke_smolvlm.yaml --run-dir runs/e6/smoke_smolvlm
python -m vlm_dflash.cli eval runs/e6/smoke_smolvlm --config configs/smoke_smolvlm.yaml
```

Evidence archive: `a6000xuyang:~/recipe-smoke-config/e6_evidence.tar.gz`
(sha256 `750db94ddeb96e24...`): train/eval logs, probe logs, `provenance.json`,
`features/manifest.json`, `capture/visual_path_check.json`,
`train/train_summary.json`, `eval/eval_result.json`, `eval/samples_ckpt_final.jsonl`.
