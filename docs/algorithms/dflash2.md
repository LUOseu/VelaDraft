# DFlash2 (experimental)

DFlash2 keeps the **DFlash block-parallel backbone** (same anchors, same mask
token denoising, same multimodal plumbing) and adds two modules on top of it
(upstream audit: `docs/upstream/dflash2.md`):

- **local dynamic convolutions** (`GroupedDynamicCausalConv`): every decoder
  layer wraps both its attention sublayer and its MLP sublayer in a grouped,
  token-conditioned convolution whose taps are **causal within the anchor
  block** (a tap never reads across a block boundary — training flattens many
  anchored blocks into one token dimension). A projection of the current hidden
  state emits the two dynamic kernels; a learned base kernel is shared across
  positions and channels in a group share the dynamic term. At initialization
  the convolution is the identity, so a fresh DFlash2 stack is numerically equal
  to DFlash (unit-tested).
- **candidate selector** (`CandidateSelector`): the unary logits propose the
  top-K candidates per position, and a low-rank bilinear edge score reranks
  them,
  `score(c) = unary(c) + dot(predecessor_codebook(prev) * hidden_projection(h), successor_codebook(c))`.
  Serving keeps the single parallel draft forward and lets only the small
  selector walk the candidate path: it starts at the anchor token and conditions
  each position on the token selected at the preceding position.

## Status: **Experimental**

This is the one algorithm in the zoo whose **training objective is not
published upstream** ("the public DFlash2 implementation specifies inference but
does not publish its training objective"). What we ported is the maintainer
implementation's *experimental split objective*:

```text
unary term   : forward KL KL(target || draft) on the unary logits
               (upstream default; gamma = 4.0 fixed exponential decay)
selector term: hard K-way CE over the unary top-K candidates with the previous
               sequence token teacher-forced; when the target id is absent from
               the top-K the weakest candidate (rank K-1) is replaced by the
               target so the CE stays defined
total        : unary + selector_loss_alpha * selector, alpha = 1.0
               both terms divide by the masked position count (upstream
               loss_function semantics)
```

Consequences that must be reported honestly:

- the *architecture* (conv + selector shapes, checkpoint key names, candidate
  semantics) follows the maintainer implementation and is unit-tested against a
  loop reference;
- the *serving walk* is inferred from the upstream user guide (the pinned
  speculators revision contains no runtime); this repo implements it on the
  shared block decoder;
- the *objective* is a reconstruction, not a reproduction. It must not be cited
  as "DFlash2's training method", and the paper/blog parameter-count claims do
  not apply to these weights.
- full vocabulary is required: the selector reranks candidates *before* any
  draft→target vocab mapping, so a pruned draft vocabulary is rejected
  (`doctor --algorithm dflash2` reports the full-vocab check).

Validation status: architecture `validated`, training `experimental`, runtime
`experimental`; Qwen2.5-VL-3B validated (canonical 2K run, 128/128
token-exact; see `ALGORITHM_MATRIX.md`), Qwen3-VL reduced-scale validation
(256-row slice, 16-prompt sequential eval, export + reload, token-exact --
`RELEASE_VALIDATION.md`). The training-objective caveat above still stands
for every number on this page.

## Config

Algorithm knobs are namespaced under `algorithm.dflash2`:

```yaml
capture:
  store_final_hidden: true        # REQUIRED: capture v2 (teacher signal)
drafter:
  loss_decay_gamma: 4.0           # upstream DFlash2 default decay
algorithm:
  name: dflash2
  dflash2:
    conv_kernel_size: 2           # causal taps per convolution
    conv_group_size: 16           # hidden channels sharing one dynamic kernel
    selector_rank: 256            # selector factorization rank
    selector_top_k: 16            # unary candidates reranked per position
    selector_loss_alpha: 1.0      # weight of the selector CE term
    kd_chunk_size: 1024           # positions per teacher-projection chunk
```

`validate_config` fails loud when `store_final_hidden` is missing,
`sample_from_anchor` is true (the selector alignment assumes the DFlash slot
convention), the Domino correction head is configured, or
`conv_kernel_size > block_size`.

Teacher signal: the same staged, exact teacher as JetSpec — the target's final
hidden state is captured once (capture v2) and projected through the frozen
target `lm_head` chunk-wise at training time
(`max |lm_head(h_last) - target logits| = 0.0` on a real sample).

Memory note: like JetSpec, the forward-KL backward retains fp32
log-probabilities of every valid position (`[positions, vocab]`), so the
canonical protocol (micro-batch 8, 256 anchors, block 8, `kd_chunk_size` 1024)
peaks at ~27 GB on a 48G card; `overfit_sanity.num_samples: 8` keeps the sanity
run inside the same envelope.

## Train / eval / export

```bash
vlm-dflash train  --config configs/dflash2/qwen2_5_vl_3b_2k.yaml
vlm-dflash eval   runs/qwen25vl3b_dflash2_2k --config configs/dflash2/qwen2_5_vl_3b_2k.yaml
vlm-dflash export runs/qwen25vl3b_dflash2_2k
```

The train log carries `unary_kl`, `selector_ce`, `candidate_recall` (fraction of
positions whose target is inside the unary top-K), the selector's realized
acceptance length and `unary_oracle_al` (the top-K oracle). Eval summaries record
`draft_candidate_selector: true`, so a result says which runtime path produced
it. The export records `method: dflash2`, the conv/selector hyper-parameters and
a `runtime_note` spelling out the experimental objective.

## Slot conventions / runtime

`sample_from_anchor: false` (DFlash convention; enforced). Block offset `k`
predicts the token at `anchor + k`; the selector's predecessor for that slot is
the token at `anchor + k - 1` — during training teacher-forced from the block
(`label_indices[..., :-1]`), during serving the token selected at the previous
slot (`_sample_draft_tokens` walks the block left to right). Training metrics
that describe serving (`accuracy`, `accept_len`) are gated on positions whose
target was inside the *un-injected* top-K, matching upstream.

## Upstream reference

```text
z-lab/dflash `dflash/model.py` @ 07ebd93db9f472af339b644bb70221ad8428328a (MIT)
vllm-project/speculators @ 55ebbd5b9b49e9bd223f47949c986aa4abfffbc3 (Apache-2.0)
blog: https://inco.ai/blog/dflash2/
```
