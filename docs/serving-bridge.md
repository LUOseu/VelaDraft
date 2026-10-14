# Serving bridge — first build (vLLM 0.22.1), strict gate not passed

Status (2026-10-14): **builder implemented and load-verified; the token-exact
gate is NOT passed; no vLLM/SGLang support claim.** The repo trains,
evaluates and exports drafters; the only token-exact inference path is the
in-repo sequential verifier. A converter +
check harness for vLLM's native `dflash` method now exists (see below), and
WP5 measured exactly how far it gets and where it stops — full numbers,
divergence analysis and registered gaps in
[`plan/WP5_REPORT.md`](plan/WP5_REPORT.md) (runs R-20261014-14/15/16).

## Build status (2026-10-14)

- vLLM **0.22.1** ships a native `dflash` speculative-decoding method
  (`DFlashProposer` + `DFlashDraftModel`) that consumes the same draft
  architecture this repo ports: block mask tokens embedded via the target's
  input embeddings, auxiliary target hidden states, shared target
  embed_tokens/lm_head.
- Implemented here: `vlm_dflash/export/vllm_bridge.py` (export ->
  speculators-format draft, weights copied unchanged, tensor
  names/shapes validated, `bridge.json` + `VLLM_DRAFT.md` written),
  `scripts/export_vllm_draft.py` (CLI), `scripts/vllm_bridge_check.py`
  (two-phase target-only vs dflash greedy check + `llm.get_metrics()`
  spec-decode accounting), `scripts/probe_divergence_margin.py` (frozen-prefix margin replay with a
  causal gate; HF-assisted diagnostic only), 5 bridge + 7 probe CPU tests.
- Verified end-to-end on the frozen 3B DFlash checkpoint: the draft loads
  (`DFlashDraftModel`, aux layers `(2, 18, 34)` -> internal `(1, 17, 33)`,
  mask 151669, n_spec 7), generation runs, metrics are readable.
- **Not achieved:** token-exact greedy output. On 32 pinned prompts,
  vLLM-dflash matched its own target-only output on 5/32; AL 1.4059 vs
  in-repo sequential 1.6866/1.6924 on identical weights (the margin probe
  observed small top1-top2 margins at the 4 traced flips - an HF-assisted
  observation; the root cause stays unresolved). The remaining AL gap (-17%)
  is registered as the open blocker; until it is closed, no vLLM number is
  quotable.

## Why "exported checkpoint" ≠ "serving-compatible"

A zoo export is HuggingFace-style *weights plus JSON metadata*: an LLaVA/Qwen
config (`config.json`), the drafter's `draft_config.json`, the target contract
(`target_contract.json`), the algorithm record (`algorithm.json`) and
`model.safetensors`. A serving runtime (SGLang, vLLM, TensorRT-LLM, …) consumes
something quite different:

| Concern | Export | Serving runtime expects |
| --- | --- | --- |
| Weight layout | single safetensors, one state dict | runtime-specific module mapping (fused QKV, sharded weights) |
| Block forward | draft config + attention backend name | a registered model class implementing the runtime's forward API |
| Verify path | in-repo sequential verifier (token-exact) | runtime's speculative-decoding scheduler and KV bookkeeping (bonus tokens, cache rollback) |
| Multimodal conditioning | captured hidden states + image spans (training-side) | runtime's own vision tower + MRoPE position ids re-derived per step |
| Mask token | `mask_token_id` recorded in the contract | extra embedding row registered as a special token |
| Precision/quantization | bf16/fp16 weights | runtime's quantization format (fp8/int4/int8) |

Two of these are silent-failure traps: (1) verifying with a *different*
attention backend changes near-tie logits in bf16 (the DSpark token-exact gate
failure fixed in v0.3.0 came from exactly this); (2) serving against a
different head than the one recorded in `target_contract.json`
(tied embedding vs the target's own `lm_head`) collapses acceptance to ~1.0
without any error. Any bridge must therefore consume `target_contract.json` and
fail loudly on mismatch, exactly like `train`/`eval`/`export` already do.

## Smallest honest first step (status 2026-10-14: steps 1-3 implemented; step 4 gate NOT passed; step 5 not reached)

Pick **one** target, **one** drafter, **one** runtime and build a thin adapter:

1. Qwen2.5-VL-3B-Instruct target + the frozen DFlash export
   (`qwen25vl-3b-dflash`, AL 1.6298) + one runtime that already supports
   Qwen2.5-VL speculative decoding (SGLang or vLLM draft-model API).
2. Register the draft block model as a module from `draft_config.json`
   (three draft layers, block size 8, anchors 256).
3. Port the verification bookkeeping from `vlm_dflash/eval/run.py`
   (sequential verify: accepted tokens, bonus token, cache rewind) onto the
   runtime's spec-decode hooks.
4. Prove, before any speed number is quoted: **greedy token-exactness against
   target-only decoding on ≥32 prompts**, with the runtime's own target call
   (same dtype/backends), reusing the equivalence harness
   (`tests/` + `eval_result.json` format).
5. Only then measure latency (time-to-first-token, inter-token, tokens/forward)
   and state hardware + verification mode next to every number.

What the WP5 build actually used: vLLM 0.22.1's native `dflash` method on
Qwen2.5-VL-3B-Instruct (steps 1-3 done via the converter + harness); the
step-4 gate failed (5/32 token-exact) and the residual AL gap is the
registered blocker (`plan/WP5_REPORT.md`).

## What must not be claimed until it is real

- No "vLLM/SGLang support" line in the README before a token-exact run exists
  on a pinned runtime version.
- No speedup claim from a non-exact path (`parallel` / `strict_parallel` are
  diagnostics; see `docs/verification-modes.md`).
- No re-labeled AL: runtime acceptance must be measured with the same
  `AL = new_tokens / target_verification_steps` definition and the same prompt
  set as the in-repo numbers, otherwise the tables are not comparable.
- No vLLM AL or latency table until the registered AL gap (1.4059 vs
  1.6866/1.6924 on identical weights, `plan/WP5_REPORT.md`) is resolved and
  the run is token-exact on the pinned prompt set.

## What the repo already provides for the bridge

- self-describing exports (`checksums.json`, `model_card.md`, contract +
  algorithm JSON) so a bridge can validate what it loaded;
- `target_contract.json` with mask token, MRoPE sections, draft layer ids,
  pixel bounds and the head record;
- the equivalence harness and the frozen DFlash baseline
  (`tests/reference_dflash/manifest.json`) to regression-test a bridge against
  the reference acceptance;
- provenance gates that make teacher/target drift fail loudly.
