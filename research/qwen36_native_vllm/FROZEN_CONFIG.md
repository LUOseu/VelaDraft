# Qwen3.6-35B-A3B seven-method native E2E evaluation: frozen configuration

Frozen before the formal MMSpec runs. Every value below is fixed for all seven
methods; nothing is chosen per method or from test-set speed.

## Code and runtime

| Item | Value |
| --- | --- |
| Evaluator base | `LUOseu/VelaDraft` main `7d7eab4ff39eaadfa3033c6d5093fc3e7db03926` |
| Evaluator changes | branch `khazic/feat/qwen36-native-eval` (this directory + `vlm_dflash/eval/native*.py`); commit recorded in every plan |
| Runtime | vLLM 0.22.1, torch 2.11.0+cu130, transformers 5.12.1 (training version) |
| vLLM sources | `qwen3_dflash.py`, `spec_decode/dflash.py`, `llm_base_proposer.py`, `mrope.py` byte-identical to the Qwen3-VL-8B campaign pins |

## Target and prompt

| Item | Value |
| --- | --- |
| Target | `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, BF16 |
| Tensor parallel | TP=2 per lane (target and draft), concurrent lanes on disjoint GPU pairs of one node, batch 1 |
| Chat template | target template with `enable_thinking=false` (training identity) |
| System prompt | MMSpec reference (`evaluation/utils.py::build_prompt`): "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions." |
| Multi-turn | image on the first user turn only; earlier assistant turns are the untimed native AR reference outputs (MMSpec construction) |
| Image pixels | min 65,536 / max 1,048,576 (training identity; at most 1,024 visual tokens) |
| Stop ids | target `generation_config.eos_token_id` = [248046, 248044] |
| Decoding | greedy (temperature 0), `max_new_tokens` 1024 (MMSpec reference), `max_model_len` 8192 |

## Arms

| Arm | Engine |
| --- | --- |
| AR | vLLM native defaults (automatic attention backend, default compile and `FULL_AND_PIECEWISE` CUDA graphs, custom all-reduce) |
| SD | `FLASH_ATTN` for target and draft, linear DFlash block verification, S7/B8 (`num_speculative_tokens=7`), VLLM_COMPILE, target `FULL_AND_PIECEWISE` graphs at capture size 8, draft graphs PIECEWISE |

Shared by both arms: prefix caching off, multimodal processor cache off, a fresh
image UUID per physical request, `disable_log_stats=True` in timed runs.

### Deviation from the Qwen3-VL-8B protocol: SD graph mode

The 8B protocol pins SD to PIECEWISE graphs. On this hybrid target (30 Gated
DeltaNet + 10 attention layers, 256-expert MoE) the split ops dominate a
piecewise step: in a 12-session diagnostic AR fell from 153 to 48 tok/s under
PIECEWISE, and SD (DFlash) ran at 77 tok/s under PIECEWISE versus 162 tok/s with
the engine-native `FULL_AND_PIECEWISE` target graphs (acceptance length 2.8 to 2.9
in both). The SD target therefore uses the same native graph mode as AR. The draft
keeps PIECEWISE graphs: a FULL draft graph together with custom all-reduce faults
(illegal memory access) in vLLM 0.22.1 at TP=2, and disabling custom all-reduce
would penalise the SD arm.

## Timing protocol (unchanged native-eval)

Process order AR0, SD0, SD1, AR1, AR2, SD2 per method, each a fresh process and
engine; every source turn is warmed before each timed stage; startup, compile and
warmup excluded. Request wall time includes image read and RGB decode, template,
processing, vision, prefill, decode and output collection. Headline speedup is
sum(AR wall) / sum(SD wall) over all timed pairs; 95% interval resamples whole
sessions (2000 draws, seed 42). Inputs must reproduce the reference token ids in
every request.

Acceptance length comes from a separate stats-enabled supplement (one repeat,
`disable_log_stats=False`, per-request official counters), never from the timed
runs: vLLM convention 1 + accepted draft tokens / drafts.

## GPU ownership in a PID-namespaced pod

`nvidia-smi` reports host PIDs that the pod cannot resolve, so the per-PID
ownership check is replaced by count mode. Every leased GPU must be empty at start.
After each stage the lease waits up to 180 s for the stage's processes to release the
cards. While a stage runs, the first GPU may carry two compute apps (the client
context and TP rank 0) and every other GPU one (its TP worker). An extra app inside
a timed window invalidates the run; extra apps outside timed windows (engine load,
compile, warmup, stage boundaries) are counted and reported. Per-GPU memory is
sampled every two seconds. No continuous exclusivity is claimed.

The formal runs were made at evaluator commit `69efbba`, which applied the two-app
bound to every GPU of a lane; `audit_ownership.py` re-checks their resource evidence
against the per-GPU bounds above.

## Methods and scope

| Method | Implementation scope |
| --- | --- |
| DFlash | parallel block drafter, target features [1,19,37], spatial partial MRoPE |
| DSpark | fixed-block S7 integrated scope: anchor-slot drafting, vanilla Markov bias on real previous tokens; confidence head loaded, unused by fixed-length drafting |
| Domino | anchor-slot drafting, prefix GRU over real previous-token embeddings, correction head from slot 1 |
| JetSpec | local shared-block integrated single-path reference (block-causal draft queries); not the upstream parallel-tree runtime |
| DFlash2 | block-local grouped dynamic convolutions, top-16 unary candidates reranked by the bilinear selector, sequential |
| DFlare | per-layer softmax fusion of target layers with dedicated context K/V, no FC |
| DFly | FC base + per-layer fusion residual context, sequential greedy hidden correction |

All seven load every trained tensor (export is bitwise equal to the training
checkpoint, verified again at run time), share the target's embedding and untied
LM head, and were checked on CPU against the training implementation with the
real weights (exact token equality on synthetic hidden states).

## Data

MMSpec `killthefullmoon/MMSpec@290486861eabd746075ca776adb66175636c1fc0`, full
test: 600 sessions, 723 turns, six topics x 100 sessions. Smoke: 12 testmini
sessions (first two per topic; multi-turn topic restricted to sessions with at
least two turns), functional checks only.

## Fused-MoE kernels

vLLM 0.22.1 ships no tuned fused-MoE config for this expert shape on the
evaluation hardware, so both arms use the default kernel configuration.
