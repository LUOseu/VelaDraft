# Phase-1 competitive audit (pinned 2026-10-14)

Executed before any large-compute request, per the route document's execution
constraints. Method: GitHub API for repository state, shallow clones of the
three code repositories for file-level inspection, the NVIDIA tracking issue
and the arXiv abstract for the rest. Every claim below carries the pinned
commit or the exact file that was read. Caveat kept in force: **a search that
found nothing is not evidence of absence** - only the listed paths/lines were
checked, at the listed commits.

| Project | Revision audited | How checked |
| --- | --- | --- |
| SpecForge (`sgl-project/SpecForge`) | `fdcc2eddf1b56b5b24a5a15d3abcb366adfeb5aa` (main, 2026-10-07) | shallow clone + `specforge/` grep |
| Speculators (`vllm-project/speculators`) | `e83f9d0e562ed765da6988f0506d298b98c41061` (main, 2026-10-13) | shallow clone + `src/` grep |
| AngelSpec (`Tencent/AngelSpec`) | `d3412bed231025ea85d35dbf5e0af44f1ac62a5b` (main, 2026-07-31, v0.1.0) | shallow clone + `angelspec/` grep; README/technical report pointer |
| NeMo Automodel (`NVIDIA-NeMo/Automodel`) | `8d6bc82211f60d5c1adfcbacf040950732577779` (main, 2026-10-14) | GitHub tree + `issues/2958` tracking issue + recipe/config files |
| TIDE (arXiv 2602.05145) | paper only | arXiv abstract; GitHub search for the title -> 0 repositories |

## SpecForge `fdcc2edd` - text-only, but the loss ingredients are there

- Algorithm packages: `specforge/algorithms/{dflash,domino,dspark,eagle3,peagle,mtp}` - no vision/VLM module anywhere in `specforge/` (grep for `vision|multimodal|image` finds only attention "processor" and ring-attention plumbing).
- `specforge/core/lk_loss.py` implements `expected_acceptance_rate` (`sum_v min(p_target, p_draft)`) and `compute_lk_loss`, consumed by `algorithms/eagle3/model.py` and the DFlash family providers.
- `specforge/algorithms/common/dflash_family_model.py` has `_dpace_weight`, TV-loss terms (`tv_loss_num`), and LK usage - the AngelSpec/DFly loss ingredients now exist here too.
- `specforge/core/compact_teacher.py` reconstructs teacher quantities from the target's last hidden states + `lm_head` **without materializing full-vocab fp32 logits** (`_tiled_logsumexp_argmax`, `DEFAULT_VOCAB_CHUNK_SIZE = 32768`), with draft-vocab compaction.
- No `cold_start` / `two_stage` string in the audited tree: the *staged* LK/D-PACE -> end-to-end TV schedule our DFly `upstream_two_stage` recipe ports was not found at this revision (the composable loss terms are present).
- Training topology: online (server capture, disaggregated external target) + offline configs under `examples/configs/`.

Implication for VelaDraft: nothing here is multimodal. The DFly/LK/TV objective work is **not novel** and is actively maintained by at least two other projects; our recipe port's value is faithful, testable VLM execution + evidence, not the objective itself. Chunked/memory-bounded teacher reconstruction likewise has precedent here (`compact_teacher.py`).

## Speculators `e83f9d0e` - VLM support is target plumbing, not image supervision

- README claims "Training is supported for MoE, non-MoE, and Vision Language models"; the code shows what that means:
  - `train/cli.py:152`, `train/vocab_mapping.py:131`, `models/eagle3/core.py:182`: "For multimodal models (Qwen3VL, etc.), extract text_config" - i.e. **targets** can be VLMs, handled by reading their text sub-config for vocab/head work.
  - `train/data.py` has `_has_multimodal_content(messages)` and per-row multimodal handling comments, i.e. some data-path awareness.
  - `cli/regenerate_responses.py:31` states "On-policy regeneration has no multimodal support yet"; `docs/cli/response_regeneration.md` rejects the `sharegpt4v_coco` multimodal preset because the pipeline "cannot send its image content or retain it in a speculator-format row" and tells users to generate responses elsewhere and convert text.
- Algorithms: `eagle3, peagle, dflash, dflash2, dspark, mtp, xpress` - more method names than ours; no image/vision/processor module in the tree.

Implication: a text-only supervision stream over a VLM target is *not* the same capability as image-aware supervision. This is the narrow gap VelaDraft's schema-v2 + `image_mask` + visual conditioning occupies - but it is a **scope** gap, not a large technical moat; treat it as such.

## AngelSpec `d3412bed` - upstream of our DFly recipe, LLM-only

- README v0.1.0: 6 draft architectures (DFly, DFlash, DFlare, Eagle3, DSpark, MTP) behind one pipeline; acceptance-aligned objectives (CE, top-k KL, LK, D-PACE, end-to-end TV); document-aware sequence packing; **online evaluation against the latest checkpoint during training** ("genuine speculative decoding ... as measured by the serving engine"); disaggregated inference/training GPU groups over a Mooncake tensor store.
- `angelspec/training/dfly_trainer.py` exposes exactly the knobs we ported: `dpace_alpha`, `lk_loss_weight`, `lk_loss_type`, `lk_eta`, `e2e_tv_loss_weight` (plus `kl_topk`, `gate_entropy_weight` which we do not port).
- Multimodal surface: `multimodal_inputs` is an opaque per-sample field passed to the vLLM/SGLang **rollout engines** (`angelspec/controller/{inference_manager,training_controller}.py`, `inference/engine/{vllm_engine,sgl_engine}.py`); `inference.vllm.enable_multimodal` is an inference-config flag. The draft training/hidden-state path itself has no image handling; MRoPE appears only via an importable `apply_multimodal_rotary_pos_emb` helper in a draft model file.

Implication: AngelSpec is the reference for the recipe and a source of the "training-time real evaluation" and "document-aware packing" features our roadmap borrows. VLM prompt rollout through an engine is not VLM supervision; but our claims must give AngelSpec credit for the recipe and packing, and must not claim the staged objective as a contribution.

## NeMo Automodel `8d6bc822` - the closest competitor on the multimodal draft path

Main already ships VLM draft work, not just plans (the [`Tracking] Speculative decoding` issue #2958 is the maintainers' own map; it lists both in-flight and landed PRs):

- **ViSpec two-stage VL recipe for Qwen2.5-VL is on main**: `examples/speculative/vispec/qwen2_5_vl_vispec.yaml` (stage 2) and `qwen2_5_vl_vispec_stage1.yaml` (text-only EAGLE draft), with unit tests `tests/unit_tests/speculative/test_vispec{,_recipe}.py` and `components/speculative/regenerate_vlm.py` for target-regenerated VLM data (LLaVA-Pretrain example; "long answers" requirement called out in the config).
- The stage-2 config documents exactly the multimodal-supervision hazards this repo also handles: image tokens inflating sequence length ("micro_batch_size MUST be 1" to allow per-sample compressed lengths), `image_max_pixels` capping so the assistant turn survives truncation, and the failure mode "loss_mask is all zeros, and the step contributes an exactly-zero loss".
- Draft-side VLM data path: `components/datasets/vlm/{dspark_collate,fake_image,neat_packing_vlm,pp_media}.py`; a text DSpark draft with multimodal data for MiniMax-M3 VL (#2877).
- DFlash-family (DFlash/Domino/JetSpec) register only `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`; issue #2958's "not implemented" list says VLM targets "are being handled by #3166 and #3173, so far scoped to the Qwen-VL family".
- The same issue lists implemented items our roadmap also wants: offline precomputed hidden-state cache with opt-in **top-k `target_probs` compression**, distributed multi-node precompute, online regeneration loop with a detached worker (#3042), **periodic real acceptance length in a live vLLM server from the trainer** (#3037), sequence packing for DFlash/DSpark/Domino/JetSpec, fp8 drafts, vLLM serving for DFlash+JetSpec. Not implemented there: train-with-decode closed loop, engine-side offline capture, DSpark/Domino inference.

Implication (important for go/no-go): "train a multimodal block/parallel drafter for a VLM" is **already being built in the open** by NVIDIA for Qwen-VL with a ViSpec/EAGLE lineage. A VelaDraft paper claim of the form "first/only multimodal drafter training" is not supportable. What remains defensible: the DFlash-family (block-parallel L2) multimodal execution with a documented supervision representation, controlled cross-method/cross-recipe comparisons under fixed protocols, the resource-budget analysis, and reproducible open weights - none of which this audit found packaged together elsewhere.

## TIDE (arXiv 2602.05145, paper only)

- "Temporal Incremental Draft Engine": serving-engine-native online draft adaptation; reuses target hidden states produced **during inference** as training signals (no extra target compute), adaptive runtime control to activate speculation/draft training only when beneficial, heterogeneous clusters (inference and training on different GPU classes).
- No public repository found from the title as of 2026-10-14 (GitHub search: 0 results); cannot verify code-level claims ("search found nothing != absence").
- Relation to VelaDraft: TIDE targets *serving-time self-improvement* for LLMs; our online mode is same-process teacher-forced feature extraction during an offline training loop. Different lifecycle, and TIDE's "reuse inference hidden states" idea is orthogonal to (not covered by) our current feature-store design - if the roadmap ever adds draft rollout/self-improvement, TIDE is the closest prior work and must be compared.

## Net effect on the claim map (phase-1 view)

1. **Not novel, do not claim**: LK / D-PACE / end-to-end-TV objectives (AngelSpec + SpecForge present, NeMo Eagle-3 LK); chunked teacher reconstruction (SpecForge `compact_teacher`); training-time real acceptance evaluation (AngelSpec, NeMo #3037); sequence packing (all three); serving bridges for DFlash (NeMo/vLLM, Speculators); offline feature caches and top-k teacher compression (NeMo); image-span query compression as a mechanism (ViSpec/NeMo `img_adaptor`).
2. **Defensible, keep as the paper's spine**: one supervision representation for multimodal (image spans, per-turn masks, positions) executed across seven block-parallel drafters with unified train/eval/export; controlled comparisons (recipes, budgets) at fixed protocols; measured resource accounting and a documented split of measured vs projected costs; reproducible releases (zoo) for a non-Qwen family included (SmolVLM2), where the audited competitor scope is Qwen-VL only.
3. **Research direction with live competition**: "visual context budget" must be positioned against ViSpec's query compression and NeMo's `image_max_pixels`/`num_query_tokens` knobs - a budget study is still viable, but it must measure drafter time/VRAM/acceptance *and* compare against those published knobs, not against a naive full-context baseline alone.
4. **Re-audit before submission**: these repositories move weekly (SpecForge and NeMo both pushed within the last 7 days of this audit). The comparison table in any paper must re-pin commits and re-check ViSpec/EAGLE-VLM status at writing time.
