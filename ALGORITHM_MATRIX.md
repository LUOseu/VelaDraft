# ALGORITHM_MATRIX

Machine-checkable progress per algorithm. Statuses: `DONE` / `RUNNING` /
`PLANNED` / `N/A`. An algorithm may only be called **Supported** when every
row up to `Documentation` is `DONE` (GOAL §30); otherwise it stays
`Experimental` or `Architecture only`.

| Stage | dflash | dspark | domino | jetspec | dflash2 |
| --- | --- | --- | --- | --- | --- |
| Upstream audit doc | DONE | DONE | DONE | DONE | DONE (objective undisclosed upstream) |
| Architecture implemented | DONE | DONE | DONE | DONE (unmodified DFlash backbone) | DONE (conv + selector) |
| Loss implemented | DONE | DONE | DONE (dual-logit + curriculum) | DONE (forward KL + causal mask) | DONE (experimental split: unary KL + selector CE) |
| Unit tests | DONE | DONE | DONE (11 tests) | DONE (19 tests) | DONE (24 tests) |
| Synthetic overfit (CPU) | DONE | DONE | DONE (overfit test) | DONE (overfit test) | DONE (overfit test) |
| Gradient coverage audit | DONE | DONE | DONE (10% audit point) | DONE (36/36) | DONE (51/51 incl. conv + selector) |
| Real VLM smoke (Qwen2.5-VL-3B) | DONE | DONE (8/8 exact) | DONE (8/8 exact) | DONE (8/8 exact) | DONE (8/8 exact) |
| Visual-path check | DONE | DONE | DONE | DONE | DONE (2K logits_diff 7.92) |
| Checkpoint save/load | DONE | DONE (roundtrip test) | DONE (eval loads ckpts) | DONE (eval loads ckpts) | DONE (eval + export round trip) |
| Eval (canonical metrics) | DONE | DONE (AL 1.735) | DONE (AL 1.764) | DONE (AL 1.599, causal-mask serving) | DONE (AL 1.954, 128/128 exact) |
| Export | DONE | DONE (verified) | DONE (verified) | DONE (verified, serving flag recorded) | DONE (verified + reload) |
| Documentation | DONE | DONE | DONE | DONE | DONE (marked Experimental) |
| 2K apples-to-apples benchmark | DONE (baseline) | DONE (+ ablation) | DONE (+ ablation) | DONE (runtime-mask decomposition) | DONE (AL 1.954 + conv/selector decomposition) |
| Runtime integration | N/A (future) | N/A | N/A | N/A | N/A |

Notes

- DSpark smoke evidence: COCO 64-row pipeline (overfit sanity passed,
  drafter visual check passed), eval re-run token-exact 8/8 after the
  verify-backend parity fix; the 2K canonical run trains + evals now.
- Domino "Export / Eval / benchmark" rows flip after the smoke + 2K runs;
  its audit is completed in `docs/upstream/domino.md`.
- JetSpec (stage 4): capture v2 stores the target's final hidden state; the
  teacher is the frozen lm_head applied to it (verified bit-exact vs the
  target's own logits). Eval serves the causal in-block mask the model was
  trained with (`inference_in_block_causal`); serving the default unmasked
  block forward costs 0.26 AL (7.6% greedy-argmax flips). The parallel-tree
  runtime is not ported; eval results state that.
- DFlash2 (stage 5, the last zoo stage) is **Experimental** by contract:
  upstream publishes the inference architecture but no training objective, so
  the port implements speculators' experimental split objective. Records: the
  conv + selector architecture with canonical checkpoint key names, the
  serving-time selector walk, a 2K canonical run (AL 1.954 token-exact) and a
  conv/selector decomposition. Audit + `what is official / inferred /
  experimental` split: `docs/upstream/dflash2.md`.
