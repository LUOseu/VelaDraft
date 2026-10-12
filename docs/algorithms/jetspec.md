# JetSpec

JetSpec keeps the **unmodified DFlash draft backbone** (no new parameters) and
changes exactly two things (upstream audit: `docs/upstream/jetspec.md`):

- **block-causal in-block attention**: the shared mask builder's `causal=True`
  path masks in-block attention so that a query at within-block offset `i`
  attends only to offsets `j <= i`, instead of DFlash's bidirectional block
  mask. The draft factorization then mirrors the target's autoregressive order;
- **forward-KL distillation**: the decay-weighted hard-label CE is replaced by
  a temperature-scaled forward KL `KL(P_target || Q_draft)`
  (`loss = T^2 * KL(logits/T)`, uniform weighting over the supervised block
  offsets `1..block_size-1`, no depth decay). The trainer never degenerates to
  CE: even a one-hot teacher would stay a KL.

The teacher signal is exact: capture v2 stores the target's **final hidden
state** per position (`capture.store_final_hidden: true`), and the trainer
projects the gathered teacher positions through the frozen target `lm_head`.
On a real sample this reproduces the target's own logits bit-for-bit
(max|diff| = 0.0), so the KD teacher is the true target distribution rather than
a top-K or hidden-state-space approximation. Nothing extra is stored on disk:
`[S, hidden]` fp16/bf16 per sample (~+25% over the fused context).

Training metrics follow upstream semantics: `accuracy` is **draft-vs-teacher**
greedy agreement (the target's argmax, not the ground-truth token) and
`accept_len` is the expected accepted-prefix length per block under that
agreement.

## Status

`validated` on Qwen2.5-VL-3B (see `ALGORITHM_MATRIX.md`, `SUPPORT_MATRIX.md`):
architecture, training, unit tests, COCO smoke and the canonical 2K run all
pass; the checkpoint exports and reloads. Qwen3-VL is untested.

**Runtime caveat (read before citing eval numbers).** JetSpec's production
runtime is *parallel tree drafting with branch conditioning*, which is **not
ported**. What this repo does serve is the closest factorization it can:

- checkpoints trained by this algorithm are served with the **same causal
  in-block mask** the trainer used (`inference_in_block_causal: true` in the
  draft config; the loader forces it for checkpoints written before the field
  existed). `spec_generate` builds an additive mask that hides exactly the
  in-call future block columns, which reproduces training's
  ``[cached prefix | anchor | causal block]`` visibility. Each eval run records
  `draft_inference_in_block_causal` in its per-checkpoint summary.
- what is still missing is **tree conditioning**: proposals are per-slot greedy
  along one path, not a top-k tree expanded over the causal factorization.

Serving-fidelity measurement (mask-mismatch diagnostic, greedy-argmax flips
between the unmasked shared forward and the training mask, one sample, cached
features): smoke checkpoint max |dlogit| 0.07 / 0 flips; **2K checkpoint max
|dlogit| 9.0 / mean 0.53 / 7.6% of slot argmaxes flip**. The trained checkpoint
therefore *does* rely on the causal factorization, which is why the default
runtime now serves the mask instead of the plain block forward; eval numbers
produced with the mask are the comparable ones (see `BENCHMARKS.md`).

## Config

Algorithm knobs are namespaced under `algorithm.jetspec` (DFlash knobs apply
unchanged; two JetSpec-specific requirements):

```yaml
capture:
  store_final_hidden: true        # REQUIRED: capture v2 (teacher hidden states)
drafter:
  loss_decay_gamma: null          # REQUIRED: uniform position weighting (no decay)
algorithm:
  name: jetspec
  jetspec:
    kd_temperature: 1.0           # softmax temperature; loss is T^2-scaled
    kd_chunk_size: 1024           # positions per teacher-projection chunk
```

`validate_config` fails loud when `store_final_hidden` is missing,
`loss_decay_gamma` is set, or `sample_from_anchor` is true (the teacher gather
alignment assumes the DFlash slot convention: offset `k` predicts
`anchor + k`, teacher at `anchor + k - 1`). Training on an old feature store
raises a capture-v2 error instead of training against a wrong teacher.

Memory note: the KD backward retains fp32 log-probabilities of every valid
position (`[positions, vocab]`), so `kd_chunk_size` bounds the transient
buffers but not that retention; at the canonical protocol (micro-batch 8,
256 anchors, block 8) the run peaks at ~29 GB on a 48G card. Lower the
micro-batch or the anchor count for smaller cards.

## Train / eval / export

```bash
vlm-dflash train  --config configs/jetspec/qwen2_5_vl_3b_2k.yaml
vlm-dflash eval   runs/qwen25vl3b_jetspec_2k --config configs/jetspec/qwen2_5_vl_3b_2k.yaml
vlm-dflash export runs/qwen25vl3b_jetspec_2k
```

Training writes the standard artifacts; the log carries the KD loss
(`kd_loss`, `kd_temperature`) plus the draft-vs-teacher agreement and
acceptance-length metrics. The export records `method: jetspec`,
`method_details` (`kd_temperature`, `kd_chunk_size`, `causal_in_block`) and a
`runtime_note` about the unported tree runtime.

## Slot conventions

`sample_from_anchor: false` (DFlash convention; enforced). Block offset `k`
(`k = 1..block_size-1`) predicts the token at `anchor + k`; the teacher
distribution for that position is the target's output at `anchor + k - 1`
(`label_indices[..., :-1]`), i.e. the same shift rule as the upstream
`_gather_teacher_logits`.
