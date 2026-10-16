# ViSpec runtime audit vs VelaDraft's block-verify path

Status: **code audit complete; same-machine micro-reproduction run (see §7)**. Numeric comparison
of our paths is in `docs/parallel_verification_results.md`.

Pinned revision: ViSpec `49a33ff0ca31e281b9dc8803fd6f3041c6c0e135`
(`git clone https://github.com/KangJialiang/ViSpec.git`, cloned 2026-10-16 11:09 CST). All file:line
citations below refer to that revision. VelaDraft citations refer to this repo at `5750c23` +
uncommitted working tree.

## 1. ViSpec generation loop (what actually happens per round)

`vispec/model/spec_model.py:425-520`:

```
initialize_tree(input_ids, self, past_key_values, ...)      # 1 target forward over the prompt
for idx in range(max_length):
    logits, hidden_state_new, outputs = tree_decoding(...)  # 1 target forward over the whole candidate tree
    best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates, ...)
    (input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
     new_token, hidden_state, sample_token) = update_inference_inputs(...)
```

- **one** target forward per round for the entire candidate structure (a tree, not a linear block);
- acceptance is a contiguous-prefix match against the target argmax
  (`utils.py:440-444`, `cumprod(posterior_mask, dim=1).sum(dim=1)`; `0` is possible);
- the correction token comes from the *same* verify logits (`utils.py:451`:
  `logits[best_candidate, accept_length]`), and the progress accounting is
  `new_token += accept_length + 1` (`utils.py:582`).

Our parallel path is the same shape for a linear block: one target forward over
`[anchor, draft_1..draft_k]`, `acceptance_length = (block[:,1:] == posterior[:,:-1]).cumprod(1).sum()`
and the correction token `posterior[:, acceptance_length]`
(`vlm_dflash/dflash/draft_model.py:_parallel_round`). The difference is tree-vs-linear, not
one-forward-vs-many.

## 2. ViSpec's target-side KV runtime (the part they wrote themselves)

`vispec/model/kv_cache.py`:

1. **Static preallocation, done once** (`initialize_past_key_values`): one tensor per device of shape
   `[2 * num_layers, batch, num_kv_heads, max_position_embeddings, head_dim]`, `dtype = model.dtype`,
   `torch.zeros`. Every layer's `KVCache` is a view into that buffer.
2. **Logical length on the CPU**: `current_length_data = torch.zeros(num_layers * 2, dtype=torch.long,
   device="cpu")` with the comment "kept on CPU for quick access and updates"; each `KVCache` holds a
   scalar view of it, so the committed length is a single in-place `fill_`/`add_`.
3. **Commit by in-place gather + copy** (`KVCache.copy(indices, prev_length)`): `index_select` the
   winning path's KV out of the speculative region, `narrow` the destination at `prev_length`,
   `copy_(..., non_blocking=True)`, then update the length pointer. There is **no cache
   reallocation and no crop**; the speculative tail is simply overwritten by the next round.
4. **KV budget reservation**: `spec_model.py:266` sets `max_length = max_length -
   spec_layer.total_tokens - 10`, i.e. the tree block always has room in the preallocated buffer.

## 3. ViSpec's position / mask handling for Qwen2.5-VL

- `utils.py:397-402` (`tree_decoding`): the verify forward's positions are rebuilt explicitly as
  `position_ids = tree_position_ids + input_ids.shape[1] + model.base_model.rope_deltas`, then
  `unsqueeze(0).expand(3, -1, -1)` for Qwen2.5-VL's 3-axis mRoPE. The prompt's `rope_deltas` from the
  prefill is cached on the model and reused every round instead of being recomputed.
- `modeling_qwen2_5_vl_kv.py:1535-1539`: tree attention is injected into the 4D causal mask -
  where `tree_mask == 0` the mask entries over the tree window are set to `causal_mask.min()`. So the
  single verify forward is a normal causal forward with a modified mask, not a separate attention
  kernel.
- They ship a full fork of the Qwen2.5-VL model (`modeling_qwen2_5_vl_kv.py`, 2508 lines) so the
  runtime owns mask/position/cache semantics rather than adapting to stock `generate()`.

Our equivalent: `prepare_inputs_for_generation` + `adapter.call_position_builder(...)` per verify
round, with the parity path driving the SDPA backend (`target_verify_backend()`), and
`past_key_values_target.crop(start)` after each commit
(`vlm_dflash/dflash/draft_model.py`).

## 4. What the measured numbers say about which of these ideas matter *for us*

From the instrumented profile (`scripts/scale/profile_parallel.py`, CUDA-event brackets around
`round_prep` / `draft` / `verify_prep` / `target_verify` / `target_prefill` / `commit` plus a `misc`
residual; see `docs/parallel_verification_results.md` for the full table):

| component | ms per round | share of speculative wall |
|---|---|---|
| target block verify (1 forward) | 29.77 | 62.5% |
| drafter (1 block forward) | 13.00 | 27.3% |
| target prefill (amortised) | 4.19 | 8.8% |
| cache commit (crop + writes) | 0.22 | 0.5% |
| verify prep (masks/positions/inputs) | 0.05 | 0.1% |
| round prep | 0.08 | 0.2% |
| misc (python glue, sampling, stop scans) | 0.35 | 0.7% |

(`A = 2.874`, `r = 1.000`, `d = 0.447`, `S_model = 1.99`, `S_wall = 1.855x` in the instrumented run
and 1.902x in the official harness run.)

**Reading:** the two pieces of ViSpec's runtime that are most distinctive - preallocated KV and
explicit mask/position construction - are the pieces our profile shows to be *cheap already*
(commit and prep together < 0.5% of the wall, ~0.25 ms/round). For a linear 8-wide block on a 3B
target with stock HF caching, the KV plumbing is not where the speedup is. The remaining levers are
(a) `A` (acceptance/progress per round), (b) `d` (drafter cost, 0.447 single-token decodes per round, 27% of the wall),
and (c) amortisation of prefill over a longer generation (8.8% here).

This is a *measurement* conclusion for our shape, not a claim about ViSpec's: their tree verifies up
to 30 candidates per round with `total_token=30` (`exp.sh` defaults `depth=3, top_k=8,
total_token=30`), and their KV path may well be the binding constraint at that width. Porting their
runtime wholesale is therefore **not** justified by our numbers; the parts worth keeping are the
design discipline (verifier owns positions/mask/length; commit is an explicit pointer update) and,
if `verify_prep` ever grows, their `rope_deltas` reuse trick.

## 5. Not to be ported

- tree retrieval/gather data structures (`retrieve_indices`, `tree_mask`, `tree_position_ids`);
- EAGLE/ViSpec tree topology and `topK_genrate` logic;
- the ViSpec vision adaptor and its drafter training recipe;
- the whole-model fork (2508-line `modeling_qwen2_5_vl_kv.py`) - we keep the stock HF target and the
  adapter layer, per this round's frozen recipe.

## 6. Licence

ViSpec is Apache-2.0. Nothing in this audit is copied code: the citations are for the purpose of
describing the design. If any non-trivial code is ported later, keep the licence/notice and attribute
it in the file header.

## 7. Same-machine micro-reproduction (2026-10-16 12:51-13:05 CST)

The task doc's §12 control was judged practical and run: the official Qwen2.5-VL-3B checkpoint is
small (1.7 GB, reachable via hf-mirror), and the official entry points run against the repo-pinned
dependency versions in a fresh venv.

Setup:

- ViSpec `49a33ff`, weights `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct` (rev
  `7dd32bc8a7010ed1efc16d17118fa7bdb3f29fe9`, Apache-2.0), stored under
  `/home/lyh/ViSpec-reference/ckpt_qwen3b`.
- official entry points `vispec.evaluation.gen_baseline_answer_sqa` / `gen_spec_answer_sqa` with the
  exact flags from the repo's own `baseline.sh` / `exp.sh`: `--test_split test --test_number -1
  --shot_number 0 --prompt_format QCM-ALE --temperature 0.0`; spec side adds `--use-ours True
  --num-q 2 --depth 3 --top-k 8 --total-token 30`.
- data: ScienceQA official `problems.json` / `pid_splits.json` / `captions.json` plus the loader's
  own HF source `derek-thomas/ScienceQA`; same 100-question sample (seed 42) for both arms.
- environment: fresh venv over `transformers==4.51.3`, `tokenizers==0.21.1`, `qwen-vl-utils==0.0.11`,
  `datasets==3.5.0`, `shortuuid==1.0.13`, torch 2.6.0+cu124; **no flash-attn installed** - the
  fork's SDPA fallback is what runs. Single RTX A6000, bf16, T=0, card otherwise idle (a neighbour
  holds 262 MiB on the same GPU).

| | baseline (their loop) | ViSpec spec | ratio |
|---|---|---|---|
| questions | 100 | 100 | - |
| tokens (their `speed.py` accounting: baseline retokenised text, spec `new_tokens`) | 9 514 | 8 624 | - |
| wall, total | 313 s | 141 s | - |
| wall, per question | 3.13 s | 1.41 s | - |
| mean per-question tok/s | 28.0 | 57.2 | - |
| **official-style speedup (mean of per-question tok/s)** | | | **2.038x** |
| pooled tok/s ratio | | | 2.005x |
| paired wall ratio (mean / median) | | | 1.956x / 1.859x |
| `tau` (pooled mean accepted draft tokens per round; 2 893 rounds) | | **1.9810** | |
| prompts with identical output text | | 32/100 | |

Reading:

1. **`tau = 1.98` reproduces their README's SQA number for this checkpoint (1.99)** to within 0.5%,
   and the wall speedup lands at **2.04x** on our card (their reported SQA speedup is 1.87x). The
   audit plus this run together say: on a same-class A6000 at T=0, an official ViSpec deployment is
   worth about **2x** on SQA-style prompts.
2. **That is the same league as our parallel path** (1.86-1.90x on our dev mix). With their
   `A_progress = tau + 1 = 2.98` per round against our `A = 2.87`, the per-accepted-token
   efficiency of the two systems is comparable (`S/A` 0.68 vs our 0.65-0.66) - there is no evidence
   in this control that their *runtime* is doing something ours is not; the differences
   (30-candidate tree vs 8-wide linear block, flash-attn vs SDPA) do not move the headline ratio
   much at this scale.
3. **Even ViSpec vs its own baseline is not token-exact here**: 68/100 prompts diverge
   mid-generation (median common prefix 175 chars, minimum 31; no early-prompt divergence). Their
   published claim is a speedup plus task accuracy, not bitwise equivalence - which is exactly the
   distinction `docs/parallel_verification_results.md` §3.4 insists on for our numbers.
4. Caveats: 100 SQA questions, one seed/sample, T=0, SDPA fallback (no flash-attn); cross-budget
   head-to-heads (their 30-token tree vs our block-8) are not licensed by this control.
