# WP2 design - multi-image / multi-turn (schema v2)

Owner: C-01a13621 (see COORDINATION.md). Status: implemented; E2 smoke PASSED
(`R-20261014-08` - smoke-scale evidence only, see RUN_LOG.md). Everything
here is incremental on top of the v1 single-image / single-turn pipeline; all
existing feature stores and runs keep loading unchanged (v1 byte-compatible).

## Goal

Multi-image (N >= 1 per sample, attachable to any user turn) and multi-turn
(K >= 1 assistant turns supervised) data, features, supervision and position
protocol, with reproducible digests and an end-to-end smoke (E2).

## What v1 assumes today (to lift)

- `data/schema.py`: one image, first user/assistant pair only; rejects image
  lists longer than 1 (`v0.1 supports exactly one image per sample`).
- `data/prepare.py` llava source: keeps `conversations[0:2]`.
- `data/sharegpt.py`: `row_question` / `row_answer` = turns 0 / 1;
  `row_image_path` refuses != 1 image.
- `target/prompts.py`: renders one user message with one image placeholder.
- `target/regen.py`: one image per row, one greedy answer, stores
  `regen.prompt_ids` + `regen.answer_ids` (the bit-exactness anchor for capture).
- `target/sample_capture.py`: `locate_image_span` requires ONE contiguous image
  token span; one image-token-count check against one grid row; `loss_mask` =
  one contiguous answer block.
- `target/visual_check.py`: single-image swap probe.

## Row format (the shape all stages consume)

Superset of the current format; v1 rows stay valid.

```json
{
  "id": "...",
  "conversations": [{"from": "human"|"gpt", "value": "text with <image> placeholders"}, ...],
  "images": ["rel/or/abs/path.jpg", ...],
  "image_turns": [0, 0, 2],          // conversation index each image belongs to, same order as images
  "schema": "vlm-dflash-data-v2"     // optional marker; absent = v1 semantics
}
```

Rules:

- `conversations` must alternate human/gpt starting with human; K pairs.
- `len(images) == len(image_turns)`; each image's `<image>` placeholder lives
  in the text of the conversation it is attached to. The renderer emits each
  turn's images (in `images` order) at the START of that turn's content, then
  the turn text with placeholders stripped - identical to the v1 layout for
  single-image rows.
- v1 rows (no `image_turns`) are normalized to `image_turns = [0] * len(images)`
  and MUST keep `len(images) == 1` until a follow-up flips the default.

## Regen v2

For each sample: render the conversation progressively, generating each
assistant turn greedily conditioned on all previous turns and images
(teacher-forced history from the generated tokens, not the dataset text -
same rule as v1). Store per turn:

```json
"regen": {
  "turns": [{"prompt_ids": [...], "answer_ids": [...]}, ...],
  "full_ids": [...],            // incremental concat, see below
  "answer_spans": [[s, e], ...] // half-open spans into full_ids
}
```

`full_ids` is built incrementally: `prompt_ids[0] + answer_ids[0]` then for
turn t the suffix of `prompt_ids[t]` after the previous turn's committed
prefix, then `answer_ids[t]`. The builder asserts the incremental
concatenation equals a single `apply_chat_template` tokenization of the whole
conversation (fail loud on any chat-template drift, same philosophy as the
existing prompt-drift check).

Keep/skip policy: keep a sample if every generated answer is non-empty and
the TOTAL assistant tokens >= `min_response_tokens`; record per-turn counts in
`regen_stats.json`. `min_answer_chars` filtering stays at the prepare stage on
the source text of the LAST assistant turn (single-turn rows behave exactly
as v1).

## Capture v2

- `record["image_spans"]`: list of contiguous `[start, end]` spans (in
  appearance order); `image_span` keeps the FIRST span for v1 consumers.
- Per-image token counts validated against `image_grid_thw` rows
  (sum(prod(grid_i))/merge^2 == number of image tokens across spans).
- `loss_mask` = union of the assistant spans (multi-segment OK: anchor
  sampling only needs masked positions, and block scoring masks by
  `pred_mask & loss_mask`).
- `record["turn_spans"]`: `[[s, e], ...]` for provenance / evals; optional.
- `index.jsonl` rows gain `image_spans` + `n_turns`; `manifest.json` gains
  `schema_version: 2` and keeps `format: vlm-dflash-features-v1` so v1 stores
  (missing the new fields) keep loading with `image_span`-only semantics.

## Positions

`position3` stays what the target's own `compute_3d_position_ids` returns for
the full multi-image sequence (processor-emitted `image_grid_thw` +
`mm_token_type_ids` replay). `spatial_rope_mode=replace` therefore works
unchanged; the drag test in E2 checks drafter-side consumption
(`tests/test_mrope_layouts.py` gains a multi-image case).

## Training & eval

- Collate path unchanged (loss_mask already a per-position float tensor).
- Eval/regen for multi-image prompts reuses the same renderer.
- Reports: per-stage wall time + peak memory for the E2 smoke (training-cost
  report), plus the multi-image inference smoke (inference report).

## Non-goals

- Video, interleaved image-token-in-answer supervision, per-turn loss
  reweighting, and source-specific templates beyond the renderer above.
- Changing v1 behavior for any existing config: v2 activates only for rows /
  sources that opt in (`schema_version: 2` rows, `dataset.turns: all`,
  `dataset.max_images > 1`).

## Test plan (CPU)

1. schema: v2 messages parse to ordered images/turns; v1 rows byte-compatible.
2. renderer: multi-image multi-turn prompt rendering (fake processor fixture).
3. regen bookkeeping: span/mask construction from a stub tokenizer.
4. capture record: multi-span validation on fake tensors + per-image count
   check; v1 records unchanged.
5. collate: v1 store (no image_spans) and v2 store load through the same path.

E2 (GPU): synthetic 2-image / 2-turn JSONL -> prepare -> regen -> capture ->
train (tiny) -> eval token-exact; record cost report.
