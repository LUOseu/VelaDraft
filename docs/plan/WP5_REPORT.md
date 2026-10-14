# WP5 report - vLLM 0.22.1 native DFlash bridge: load path verified, token-exact gate not passed

Runs: `R-20261014-14` (vLLM bridge check), `R-20261014-15` (in-repo
serving-convention arms), `R-20261014-16` (divergence margin probe; diagnostic
added during execution, not pre-registered). Plan owner C-01a13621. GPU0
(R-14), GPU2 (R-15), 1 GPU (R-16). Code: `vlm_dflash/export/vllm_bridge.py`,
`scripts/{export_vllm_draft,vllm_bridge_check,probe_divergence_margin}.py`,
CPU coverage in `tests/test_export_vllm_bridge.py` (5 tests).

## What vLLM 0.22.1 provides (verified against the installed wheel)

- Native `dflash` speculative-decoding method:
  `vllm/v1/spec_decode/dflash.py` (`DFlashProposer`) +
  `vllm/model_executor/models/qwen3_dflash.py` (`DFlashDraftModel`). The draft
  directory follows the `speculators` layout: `config.json` with
  `speculators_model_type: "dflash"`, `draft_arch qwen3-dflash-block`, the
  target verifier path, the mask token and the auxiliary hidden-state layer
  ids, next to the draft weights.
- Query tokens are the bonus token + mask tokens, embedded through the
  target's own input embeddings (dfash.py: "For DFlash we use the input
  embeddings to embed the mask token"); context K/V is precomputed from
  captured target hidden states. The checkpoint may omit
  embed_tokens/lm_head; vLLM then shares the target's (load log: "Detected
  EAGLE model without its own embed_tokens ... Sharing").
- Layer-id convention: vLLM's `aux_hidden_state_layer_ids` is the HF
  `output_hidden_states` index; it equals this repo's `target_layer_ids + 1`
  (the repo captures `hidden_states[layer_id + 1]`). vLLM derives its internal
  `target_layer_ids = aux - 1`, reproducing the repo ids verbatim. Dry-run
  dump: aux `(2, 18, 34)` -> internal `(1, 17, 33)`, `use_dflash: true`,
  n_spec 7, mask 151669 - all matching the export.
- Divergence from this repo's serving found in code: vLLM's draft path uses
  plain 1-D RoPE positions, while the repo's serving default is MRoPE-replace
  (`spatial_rope_sections [16, 24, 24]`). Measured impact in R-15 below.
- The mask-slot input convention is the same in both runtimes: the repo feeds
  `target_embeddings(block_output_ids[:, :block_size])` (`draft_model.py`,
  `spec_generate`), i.e. the target token embedding of `mask_token_id` for
  masked block slots - exactly what vLLM does. (No `mask_hidden` parameter
  exists in the repo; the earlier suspicion of a mismatch there is void.)

## Converter (implemented + tested)

`vlm_dflash/export/vllm_bridge.py` converts a vlm-dflash export directory into
the speculators layout: weights are copied unchanged (36 tensors for 3
layers: `fc.weight [2048, 6144]`, `hidden_norm`, `norm`, `layers.*`), tensor
names/shapes validated, plus `bridge.json` (provenance, layer-id mapping,
MRoPE section record) and `VLLM_DRAFT.md` (usage + caveats). CLI:
`scripts/export_vllm_draft.py --export <dir> --out <dir>`. Converted draft for
the frozen 3B DFlash checkpoint: weights sha256 `2383b226...` (36 tensors).

## Protocol (pinned in RUN_LOG before launch)

- 32 prompts = first 32 rows of the 2K val split (`runs/qwen25vl3b_2k/data/val.jsonl`),
  1 image each, greedy, 128 new tokens, bf16, min/max pixels 200704/1003520.
- R-14 (vLLM): one process, phase 1 target-only, phase 2
  `spec_method=dflash, spec_tokens=7`; success = both phases complete +
  token-exact 32/32 + spec-decode metrics recorded.
- R-15 (repo eval, sequential parity): arm M = canonical `ckpt_final.pt`
  (MRoPE-replace); arm 1D = derived checkpoint with
  `dflash_config.spatial_rope_enabled=false` (the 1-D convention vLLM's draft
  path implements). Run dir `/tmp/wp5/run_1d` (symlinked data; no writes to
  the source run). Same gate.

## Results

### R-14: vLLM 0.22.1, converted draft (GPU0)

- Load smoke (2 prompts, 16 tokens): PASS - target-only 2/2 exact, dflash 2/2
  exact.
- Full run: both phases complete. dflash: drafts 2796 / draft tokens 19572 /
  accepted 1135 -> **AL 1.4059**; per-position acceptance `[0.280, 0.087,
  0.029, 0.009, 0.001, 0, 0]`. **Token-exact vs target-only: 5/32** - the
  32/32 gate is NOT passed. Wall: target-only 12.5 s vs dflash 17.2 s on this
  tiny latency-bound workload (JIT warmup included; not a speed claim).

### R-15: in-repo arms on the same 32 prompts (GPU2)

| Arm | AL | token_exact | per-slot acceptance |
| --- | ---: | --- | --- |
| M (MRoPE-replace) | 1.6924 | 32/32 | 0.435 / 0.171 / 0.061 / 0.019 / 0.005 / 0.002 / 0.001 |
| 1D (`spatial_rope_enabled=false`) | 1.6866 | 32/32 | 0.439 / 0.172 / 0.060 / 0.018 / 0.003 / 0.0004 / 0.0 |

The RoPE convention moves AL by 0.006 (~0.3%): **MRoPE-vs-1D is not the
driver** of the vLLM gap. The same eval's parallel diagnostic (16 prompts)
reproduces the repo's documented phenomenon: 6/16 exact, every listed flip at
a top1-top2 margin <= 0.125 (bf16 near-ties).

### R-16: divergence margin probe (1 GPU)

The first 4 vLLM-vs-target-only divergences, re-derived with the repo target
(sequential). Margins are the target's top1-top2 logit gap at the traced
step; `rank` is the vLLM-dflash token's rank in the repo target distribution.

| row | step | prefix re-derived | margin | vLLM token rank | logit gap |
| --- | ---: | --- | ---: | ---: | ---: |
| 000000208145 | 83 | no | 0.125 | 2 | 0.125 |
| 000000026368 | 41 | yes | 0.0 | 1 | 0.0 |
| 000000420750 | 93 | no | 0.125 | 1 | 0.0 |
| 000000376912 | 45 | yes | 0.125 | 1 | 0.0 |

In 3/4 cases the vLLM-dflash token IS the repo target's top-1 (gap 0.0): the
vLLM *verify* path resolved a bf16 near-tie differently, it is not a wrong
draft token. In the 2 prefix-not-reproduced cases the sequences differ
earlier than the traced step (expected across runtimes).

## Attribution and registered gaps

Ruled out as the driver: the RoPE convention (R-15, ~0.3%) and the mask-slot
input convention (code-verified identical). The gate itself: vLLM's dflash
verification is a block/parallel-shaped path whose greedy output is not
bit-identical to vLLM's own target-only greedy on 27/32 prompts (all flips at
ulp-scale margins); the repo's token-exact mode works precisely because
sequential verification reuses the `generate()`-backend logits, and vLLM
0.22.1 exposes no equivalent sequential-parity verify mode. Therefore a
strict token-exact claim under vLLM's dflash path cannot be made today.

Open gap (the real remaining work): **AL 1.4059 (vLLM) vs 1.6866/1.6924
(in-repo) on identical weights and prompts (-17%)**. Candidate causes for a
follow-up root-cause pass, none evaluated yet: draft query position semantics
inside vLLM, context K/V layout through the triton input-prep kernel
(`copy_and_expand_dflash_inputs_kernel`), attention-kernel numerics in the
draft and verify paths, and scheduling/batching effects. Until this is
resolved no vLLM AL or speed number is quotable and the zoo export stays
repo-runtime-only.

## Verdict against the serving-bridge rules

- Builder: implemented, CPU-tested, load-verified end-to-end; conversion is
  layout-exact.
- Gate: NOT passed -> no "vLLM/SGLang support" claim, no speed claim; the
  README/docs say "bridge exists, token-exact gate not passed".
- Status per plan rule (blocked by runtime incompatibility: deliver the
  independently implementable part, register the gap, do not mark done, do
  not re-run the same failing job): WP5 = BLOCKED, and execution moves on to
  WP6. See `TASK_LEDGER.md`.

## Cost

0.55 GPU-h: R-14 0.11 (GPU0: smoke ~4.1 min + full ~2.6 min) + R-15 0.36
(GPU2: two eval processes, ~21 min combined device wall) + R-16 0.08
(~4.6 min). First-round budget after WP5: **3.57 / 8.00 GPU-h used**.

## Artifacts

- `a6000xuyang:~/recipe-smoke-config/wp5_evidence.tar.gz` (sha256
  `2e2687d9...`): `bridge_check.json`, `bridge_smoke.json`,
  `divergence_margins.json`, both eval result JSONs + sample files +
  config/contract, derived-1D-checkpoint sha256, and all four job logs.
- Live (tmp, renewable): `/tmp/wp5/{vllm_draft,run_1d}`; source run
  `runs/qwen25vl3b_2k` untouched.
