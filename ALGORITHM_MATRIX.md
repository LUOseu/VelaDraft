# ALGORITHM_MATRIX

Machine-checkable progress per algorithm. Statuses: `DONE` / `RUNNING` /
`PLANNED` / `N/A`. An algorithm may only be called **Supported** when every
row up to `Documentation` is `DONE` (GOAL §30); otherwise it stays
`Experimental` or `Architecture only`.

| Stage | dflash | dspark | domino | jetspec | dflash2 | dflare | dfly |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Upstream audit doc | DONE | DONE | DONE | DONE | DONE (objective undisclosed upstream) | DONE | DONE |
| Architecture implemented | DONE | DONE | DONE | DONE (unmodified DFlash backbone) | DONE (conv + selector) | DONE (fusion + separate context K/V) | DONE (fusion + hidden correction) |
| Loss implemented | DONE | DONE | DONE (dual-logit + curriculum) | DONE (forward KL + causal mask) | DONE (experimental split: unary KL + selector CE) | DONE (shared DFlash block-CE) | DONE (shared DFlash block-CE) |
| Unit tests | DONE | DONE | DONE (11 tests) | DONE (19 tests) | DONE (24 tests) | DONE (12 tests) | DONE (12 tests) |
| Synthetic overfit (CPU) | DONE | DONE | DONE (overfit test) | DONE (overfit test) | DONE (overfit test) | DONE (overfit test) | DONE (overfit test) |
| Gradient coverage audit | DONE | DONE | DONE (10% audit point) | DONE (36/36) | DONE (51/51 incl. conv + selector) | DONE (2K 42/42 incl. fusion) | DONE (2K 38/42 at step 0; zero-init path 42/42 on the trained ckpt — probe + test) |
| Real VLM smoke (Qwen2.5-VL-3B) | DONE | DONE (8/8 exact) | DONE (8/8 exact) | DONE (8/8 exact) | DONE (8/8 exact) | DONE (8/8 exact, AL 1.123) | DONE (8/8 exact, AL 1.438) |
| Visual-path check | DONE | DONE | DONE | DONE | DONE (2K logits_diff 7.92) | DONE (smoke) | DONE (smoke) |
| Checkpoint save/load | DONE | DONE (roundtrip test) | DONE (eval loads ckpts) | DONE (eval loads ckpts) | DONE (eval + export round trip) | DONE (roundtrip test) | DONE (roundtrip test) |
| Eval (canonical metrics) | DONE | DONE (AL 1.735) | DONE (AL 1.764) | DONE (AL 1.599, causal-mask serving) | DONE (AL 1.954, 128/128 exact) | DONE (AL 1.637, 128/128 exact) | DONE (AL 1.682, 128/128 exact) |
| Export | DONE | DONE (verified) | DONE (verified) | DONE (verified, serving flag recorded) | DONE (verified + reload) | DONE (smoke verified) | DONE (smoke verified) |
| Documentation | DONE | DONE | DONE | DONE | DONE (marked Experimental) | DONE (marked Experimental) | DONE (marked Experimental) |
| 2K apples-to-apples benchmark | DONE (baseline) | DONE (+ ablation) | DONE (+ ablation) | DONE (runtime-mask decomposition) | DONE (AL 1.954 + conv/selector decomposition) | DONE (+ fixed-fusion & shared-KV ablations) | DONE (+ no-correction ablation) |
| Runtime integration | N/A (future) | N/A | N/A | N/A | N/A | N/A | N/A |

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
- DFlare / DFly (AngelSpec ports, stage 6): the training objective is the
  unmodified DFlash block-CE unless `algorithm.train_recipe` selects the ported
  upstream two-stage recipe (LK / D-PACE cold start -> e2e TV; DFly only, exact
  math in `vlm_dflash/objectives.py`); what changes is the context pathway (DFlare:
  learnable per-draft-layer fusion + dedicated context K/V; DFly: DFlash
  shared-KV layers + FC context + fusion residual + zero-initialised
  hidden-states correction). Canonical 2K runs, ablations and Qwen3-VL
  reduced validations are recorded (all token-exact): DFlare AL 1.637 —
  parity with the frozen baseline, but its ablations show the dedicated
  context K/V is worth +0.058 while the learned fusion is prior-dominated
  (Δ0.0005 frozen; drift 0.011) — and DFly AL 1.682, where the hidden-states
  correction carries +0.113 (no-correction ablation 1.569). Upstream
  semantics, the ported DFly two-stage objective and the remaining
  non-port (the tree/parallel inference correction form) are documented in
  `docs/upstream/dflare.md` / `docs/upstream/dfly.md`; protocols in
  `docs/algorithms/{dflare,dfly}.md`.
