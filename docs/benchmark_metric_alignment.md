# Benchmark metric alignment: VelaDraft AL vs ViSpec tau

Status: **first pass, code-traced** (2026-10-16). Everything below cites the actual code that
produces the number; anything not yet traced is listed as an open item at the end.

Revisions pinned for this trace:

| side | revision |
|---|---|
| ViSpec (official, `https://github.com/KangJialiang/ViSpec`) | `49a33ff0ca31e281b9dc8803fd6f3041c6c0e135` (clone HEAD at 2026-10-16 11:09 CST) |
| VelaDraft (this repo) | `5750c23` + local working tree (no commit made) |

## TL;DR

**No, they are not the same quantity, and the difference is exactly one token per round.**

```
ViSpec  tau          = accepted draft tokens / verification rounds        (correction token NOT counted)
VelaDraft mean_al    = emitted output tokens  / verification rounds       (correction token counted)
                       = al_progress_tokens_per_round
VelaDraft al_draft   = accepted draft tokens / verification rounds        (== ViSpec tau by construction)
```

So the aligned comparison between our R3 drafter and the ViSpec README number is **1.80-1.85 vs
2.19**, not 2.80 vs 2.19. And even that comparison is still not apples-to-apples (candidate budget —
see the caveats section); it is only the *metric* that is now aligned.

## 1. What ViSpec actually reports as `tau`

Trace through the official code:

1. `vispec/model/spec_model.py:469-470` — per round, the raw value returned by
   `evaluate_posterior` is appended unchanged:
   ```python
   if return_acceptance_len:
       acceptance_len.append(int(accept_length))
   ```
2. `vispec/model/utils.py:437-451` — that `accept_length` is the greedy contiguous-prefix match
   length, i.e. the number of *draft* tokens that equal the target argmax, starting from the first
   speculative position. It can be `0` (first draft token rejected).
3. `vispec/model/utils.py:582` — the guaranteed-progress token is accounted for **separately**, in the
   token counter only:
   ```python
   new_token += accept_length + 1
   ```
4. `vispec/evaluation/speed.py:65` — the aggregation that prints the table number pools the per-round
   list across every datapoint and averages it:
   ```python
   acc_len += datapoint["choices"][0]["acceptance_length"]
   print(f"acc len {sum(acc_len) / len(acc_len)}")
   ```
   `vispec/evaluation/gen_spec_answer_*.py` (e.g. `gen_spec_answer_textvqa.py:229-268`) is what writes
   that list into the result jsonl.

So `tau` is the mean **raw accepted draft length per speculative round**, pooled over rounds. It is
*not* `generated_tokens / rounds`, and it does not include the correction/bonus token.

Independent cross-check from the README itself (`README.md:181`, "Speedup ratios and average
acceptance lengths tau"): the Medusa rows report `tau = 0.57-0.84` (e.g. Qwen2.5-VL 3B T=0 Medusa
`0.61`). A number below 1.0 is impossible for a progress-inclusive metric, which confirms the
draft-only reading from the code. The same convention is used by their Medusa/EAGLE-2 wrappers
(`vispec/model/spec_model_medusa.py:464`, `spec_model_ours.py:510`).

## 2. What VelaDraft reports as `mean_al`

`vlm_dflash/eval/accept.py:summarize_samples` computes, over all prompts of an eval point:

```
rounds            = sum(verify_steps)              # speculative rounds
new_tokens        = sum(new_tokens)               # emitted output tokens (correction included)
accepted_tokens   = sum(accepted_tokens)          # accepted draft tokens
mean_al           = new_tokens / rounds           # == AL_progress
```

and now also reports the draft-only counterpart explicitly:

```
al_progress_tokens_per_round = new_tokens      / rounds
al_draft_tokens_per_round    = accepted_tokens / rounds
```

`mean_al` is kept as a legacy alias of `al_progress_tokens_per_round`; both are written into
`eval_result_*.json`, and `scripts/scale/speed_paired.py` / `profile_parallel.py` report the same
pair. `accepted_tokens` counts a round's accepted draft tokens only (`_bounded_acceptance_counts`,
`vlm_dflash/dflash/draft_model.py`), so the draft-only definition matches ViSpec's `accept_length`
semantics (contiguous prefix, 0 allowed) rather than the "accepted minus clipped suffix" variant.

Measured on the R3/R5 20K round (`ckpt_final.pt`, full 256-row splits, real greedy decoding,
sequential verification):

| arm | split | rounds | AL_draft (= ViSpec tau) | AL_progress (= old `mean_al`) | difference |
|---|---|---|---|---|---|
| R3 | dev | 6793 | **1.8510** | **2.8557** | 1.0047 |
| R3 | test (report-only) | 6726 | **1.7979** | **2.8023** | 1.0044 |
| R5 | dev | 6847 | **1.8296** | **2.8332** | 1.0036 |
| R5 | test (report-only) | 6746 | **1.7901** | **2.7940** | 1.0039 |

The difference is `~1.00` as expected, and the small excess over exactly 1.0 is the EOS / output-cap
boundary effect (a round that stops on a stop token does not contribute a correction token).

## 3. Speedup metric alignment

| side | headline speedup | what it is |
|---|---|---|
| ViSpec | `vispec/evaluation/speed.py:63-73` | mean over samples of `tokens / wall_time`, divided by the same mean for the baseline (`np.array(speeds).mean() / np.array(speeds0).mean()`), i.e. a **ratio of per-sample mean throughputs**. `wall_time` is the whole `specgenerate` call (prefill + image processing + decode), model loaded outside the timer. Baseline token counts are recomputed by re-tokenizing the reference answer text. |
| VelaDraft | `scripts/scale/speed_paired.py` | `speedup_ratio_of_means = sum(target_only_wall) / sum(spec_wall)` over the same prompts, i.e. a **ratio of total wall times**; also prints the paired mean/median of per-sample ratios. Target-only wall comes from the same HF `generate` reference call. |

Both are end-to-end wall-clock (domain: prefill + decode, excluding model load and dataset I/O), so
they are in the same family - but one is a mean-of-ratios over samples and the other is a ratio of
means. On identical workloads these differ only by weighting; when we quote a headline number we
state which of the two it is (this repo reports both fields).

## 4. Rules for any cross-paper statement (enforced from now on)

1. Never write "VelaDraft AL 2.80 > ViSpec tau 2.19". The aligned pair is `AL_draft` vs `tau`.
2. Any comparison must also state the candidate budget, the benchmark set, the output-length
   distribution, the target runtime, and the dtype/kernel stack.
3. When quoting an external table, first locate the code that produces it (this document does that
   for ViSpec at the pinned revision) and record the file:line in the report.
4. Our own artifacts must keep reporting `al_draft`, `al_progress`, rounds, **and** actual target
   forward calls separately (a round is not a target call under sequential verification or a
   strict-parallel fallback).

## 5. Not-comparable list (ViSpec vs this round), for the record

- **Candidate budget**: ViSpec `exp.sh` defaults `depth=3, top_k=8, total_token=30` - up to 30
  candidate tokens verified per round as a tree. Our DFly drafter verifies a **linear block of 8**
  (`block_size=8`, at most 7 accepted draft tokens + 1 correction) per round.
- **Benchmarks**: ViSpec reports 8 VLM benchmarks (SQA, MM-Vet, TextVQA, MME, COCO Caps, VizWiz, GQA,
  SEED-Bench) with per-benchmark output lengths; our numbers are a 256-row held-out split of one
  LLaVA-style corpus.
- **Drafter**: ViSpec trains its own tree drafter per target; ours is a 3-layer DFly diffusion drafter
  trained from scratch in this round.
- **Target runtime**: ViSpec ships a fork of Qwen2.5-VL with its own KV/mask/mRoPE runtime
  (`modeling_qwen2_5_vl_kv.py`); we drive stock HF `generate` semantics plus a block-verify path.
- **Token budget during decode**: ViSpec caps generation per benchmark; our harness uses
  `max_new_tokens=128` on 256 prompts.
- **dtype/kernels/versions**: ViSpec pins `transformers==4.51.3`; our runtime pins its own stack
  (recorded in the round's environment capture).

## 6. Open items (to finish before any published cross-comparison)

- [ ] EOS round handling on the ViSpec side: what happens to `acceptance_length` when the loop breaks
      on EOS - is the final truncated round included in the pooled mean? (Affects the third decimal.)
- [ ] T=1 sampling path: `evaluate_posterior`'s logits-processor branch has its own accept logic;
      confirm the sampling `tau` has the same draft-only semantics as the greedy one.
- [ ] Confirm that the README table was produced by `speed.py` (not by a private script) - the
      aggregation matches the code, but the exact invocation is not committed.
- [ ] For a same-machine comparison, record ViSpec's own `r`/`d` from its runtime (see
      `docs/vispec_runtime_comparison.md`), not just its tau/speedup.
