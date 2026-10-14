# WP5 report - vLLM 0.22.1 native DFlash bridge: load path verified; the 5/32 token-exact reading diagnosed as bf16 ties + engine nondeterminism (no bridge defect observed); AL gap open

Runs: `R-20261014-14` (vLLM bridge check), `R-20261014-15` (in-repo
serving-convention arms), `R-20261014-16` (divergence margin probe; diagnostic
added during execution, not pre-registered; v1 probe design, v2 replay probe
implemented under the R5 review remediation - see the R-16 section),
`R-20261014-24` (baseline rerun) and `R-20261014-25` (in-engine diagnosis,
GPU_PENDING #4). Plan owner C-01a13621. GPU0
(R-14), GPU2 (R-15, R-25), 1 GPU (R-16, R-24). Code: `vlm_dflash/export/vllm_bridge.py`,
`scripts/{export_vllm_draft,vllm_bridge_check,probe_divergence_margin}.py`,
CPU coverage in `tests/test_export_vllm_bridge.py` (5 tests) +
`tests/test_probe_divergence_margin.py` (7 tests).

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
driver** of the vLLM gap. The same eval's parallel diagnostic (16 prompts,
repo-internal, no vLLM involved) reproduces the repo's documented
small-margin flip class: 6/16 exact, every listed flip at a top1-top2 margin
<= 0.125.

### R-16: divergence margin probe (1 GPU) - HF-assisted diagnostic (v1, superseded design)

The first 4 vLLM-vs-target-only divergences, traced with the repo target
(sequential). The v1 probe **regenerated its own prefix** with the repo
target before reading logits (`prefix re-derived` column); that design is
replaced by the v2 replay probe below (R5 remediation, review `950b394`).
Margins are the target's top1-top2 logit gap at the traced step; `rank` is
the vLLM-dflash token's rank in the repo target distribution.

| row | step | prefix re-derived | margin | vLLM token rank | logit gap |
| --- | ---: | --- | ---: | ---: | ---: |
| 000000208145 | 83 | no | 0.125 | 2 | 0.125 |
| 000000026368 | 41 | yes | 0.0 | 1 | 0.0 |
| 000000420750 | 93 | no | 0.125 | 1 | 0.0 |
| 000000376912 | 45 | yes | 0.125 | 1 | 0.0 |

What these numbers do and do not say:

- Margins are small (0.0-0.125) at every traced step and in 3/4 cases the
  vLLM-dflash token is the repo target's argmax at the traced step. That is
  an **observation** consistent with near-tie resolution - it does not
  establish the verify path as the root cause.
- 2/4 cases have `prefix re-derived = no`: the v1 probe's own regeneration
  did not reproduce the sequence before the traced step, so those cases were
  not on the same condition distribution. With 4 of 27 divergences traced,
  nothing here generalises to the remaining flips.
- The token-exact gate therefore stays **BLOCKED at 5/32**; nothing in this
  probe relaxes it.

Replacement diagnostic (implemented + CPU-tested):
`scripts/probe_divergence_margin.py`, output format
`vlm-dflash-divergence-margin-v2`:

- replays the *frozen* target-only prefix from the bridge JSON through one
  teacher-forced forward (no regeneration); the fed ids are asserted to end
  with the saved prefix;
- causal gate per case: probe prompt token length equals vLLM's recorded
  `prompt_len`; preprocessing is digest-deterministic across two renders; and
  every frozen prefix token is the argmax of the position predicting it
  (prefix lies on the model's own greedy trajectory). Failing cases carry an
  `exclusion_reason` and are excluded from `summary.causal_cases`;
- records top-k candidates, the vLLM token's rank/logit gap and the absolute
  probe position; evidence only a same-engine run can provide (vLLM-side
  processor digests, verify logits, per-request acceptance positions, KV
  state) is marked `missing`;
- CPU coverage: `tests/test_probe_divergence_margin.py` (7 tests: saved
  prefix enters the forward; off-trajectory / prompt-length-mismatch /
  non-deterministic cases excluded; missing-evidence marks; case selection).

### R-24/R-25: in-engine diagnosis (GPU_PENDING #4) - no bridge defect observed in the sampled range

Constraint: the vLLM 0.22.1 dflash path exposes no sequential-parity verify
mode, so token-exactness cannot be guaranteed on that path by construction.
The remaining question was whether *our* bridge/acceptance plumbing adds
divergences on top of engine nondeterminism. Env `c1vllm` (vLLM 0.22.1 /
torch 2.11), same 32 prompts and converted draft as R-14.

- **R-24 (baseline rerun)**: the R-14 reading reproduces exactly -
  token-exact 5/32, AL 1.40534, acceptance 0.0579, per-position
  `[0.283, 0.084, 0.028, 0.009, 0.002, 0, 0]` (wall dflash 18.0 s vs target
  13.1 s; warmup-dominated, not a speed measurement).
- **R-25 Phase A (determinism matrix)**: rerunning the *same engine twice* on
  the same prompts is not token-exact either - `T1a-T1b` **12/32**,
  `T1a-T2a` 12/32, `T1a-D1a` 9/32, `T1a-D2a` 9/32, `T2a-D1a` 2/32,
  `D1a-D2a` 8/32. In-engine run-to-run noise is the same magnitude as every
  cross-config pair: token-exactness is not a property this engine has.
- **R-25 Phase B (instrumented dump)**: an env-gated rejection-sampler patch
  (inert by default; pristine + patched + install/revert shipped) dumped 469
  spec steps / 638 emitted tokens over 5 prompts (batch 1, prompts
  0/1/2/4/10). Emissions reconstruct the transcripts exactly (5/5
  prefix-exact); the v2 analyzer classifies 467 steps as **453 consistent +
  14 exact bf16 tie-branch, 0 genuine inconsistencies**; first cross-engine
  divergence per prompt sits at margins 0.0 (p0/p2/p4), 0.125 (p1), and none
  within 128 tokens (p10).

Conclusion (registered): within the sampled range (5 prompts / 467
instrumented spec steps), no acceptance-logic or bridge logic error was
observed; the 5/32 reading is consistent with bf16 tie flips plus
engine-level nondeterminism. **The token-exact gate is not a usable
correctness criterion for this stack as-is**, and a complete serving
correctness verification remains outstanding; quality signals remain AL and
acceptance metrics (vLLM AL is still not quotable - see the AL gap below).

## Attribution and registered gaps

Ruled out as the driver: the RoPE convention (R-15, ~0.3%) and the mask-slot
input convention (code-verified identical). R-24/R-25 diagnosis (sampled
range): within 5 prompts / 467 instrumented spec steps no bridge/acceptance
logic error was observed - 0 genuine inconsistencies, every observed
cross-engine divergence first appears at a bf16 tie or near-tie (margins
0.0/0.125), and same-engine reruns themselves agree only 12/32. The 27/32
token flips are consistent with bf16 tie flips + engine-level run-to-run
nondeterminism; a complete serving correctness verification remains
outstanding. The repo's token-exact mode works by reusing
`generate()`-backend logits in sequential verification, and vLLM 0.22.1
exposes no equivalent sequential-parity verify mode; a strict token-exact
claim under vLLM's dflash path cannot be made today (R5 + R-25 attribution).

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
- Gate: closed as **not a usable criterion for this stack** (R-25): the
  in-engine diagnosis observed no bridge/acceptance logic error within the
  sampled range, and two runs of the same engine differ at the same rate as
  any cross-engine pair (12/32). No
  "vLLM/SGLang support" or speed claim is made; README/docs say "bridge
  exists; the vLLM path is not token-exact by construction and its AL gap is
  unresolved".
- Status per plan rule (deliver the independently implementable part,
  register the gap, do not mark done): WP5 = BLOCKED on the unexplained AL
  gap (-17%); the token-exact reading is retired as a criterion (R5 + R-25)
  and a complete serving correctness verification remains outstanding. See
  `TASK_LEDGER.md`.

## Cost

0.85 GPU-h: R-14 0.11 (GPU0: smoke ~4.1 min + full ~2.6 min) + R-15 0.36
(GPU2: two eval processes, ~21 min combined device wall) + R-16 0.08
(~4.6 min) + R-24 0.08 (baseline rerun, ~4.5 min) + R-25 0.22 (determinism
matrix ~10.7 min + single probe ~2.3 min + CPU patch/analysis). Running
total: `RESOURCE_LEDGER.md`.

## Artifacts

- `a6000xuyang:~/recipe-smoke-config/wp5_evidence.tar.gz` (sha256
  `2e2687d9...`): `bridge_check.json`, `bridge_smoke.json`,
  `divergence_margins.json`, both eval result JSONs + sample files +
  config/contract, derived-1D-checkpoint sha256, and all four job logs.
- `a6000xuyang:~/recipe-smoke-config/wp5_gpu4_evidence.tar.gz` (sha256
  `25fba026...`, 17 files, `EVIDENCE_README.txt` inside):
  `determinism_matrix.json`/`.log`, `bridge_check_rerun.json`/`bridge_rerun.log`,
  `probe_b1/{single_probe.json,steps.jsonl,verify_path_analysis_v2.json}`,
  `tools/analyze_dump_v2.py`, `wp5_{determinism_check,single_probe}.py`,
  `rejection_sampler.{patched,pristine}.py`, `install/revert_wp5_patch.sh`.
- Live (tmp, renewable): `/tmp/wp5/{vllm_draft,run_1d}`; source run
  `runs/qwen25vl3b_2k` untouched.
