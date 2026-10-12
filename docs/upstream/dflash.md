# Upstream audit — DFlash

Audit before any DFlash-adjacent change: this is the frozen reference the whole
repo is calibrated against.

```text
Paper:                     "DFlash: Block Diffusion for Flash Speculative Decoding"
                           (Z Lab), arXiv:2602.06036, project page z-lab.ai/projects/dflash
Official repo:             z-lab (project page / arXiv listing); the maintained
                           engine-side implementation lives in vllm-project/speculators
                           and NVIDIA-NeMo/Automodel
Reference implementations: vllm-project/speculators `src/speculators/models/dflash/*`
                             + `docs/user_guide/algorithms/dflash.md`
                           NVIDIA-NeMo/Automodel `nemo_automodel/components/speculative/dflash/core.py`
                             + `nemo_automodel/components/speculative/dflash/draft_qwen3.py`
                           jacky566/Automodel_lao @ d21fb5aa (read-only personal fork runbook;
                             Apache-2.0) — the reference this repo's port was taken from
Commit inspected:          speculators @ 55ebbd5b9b49e9bd223f47949c986aa4abfffbc3 (2026-10-08)
                           NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace (main)
License:                   Apache-2.0 (both upstreams)

Backbone:                  Qwen3-style draft stack (3 full-attention layers by default),
                           init from the target's text config, forward over [context | noise]
Extra modules:             none beyond the draft stack; the block attention mask
                           (`dflash_mask.py`) and the target-context fusion `fc` layer
Training objective:        decayed block cross-entropy (Eq. 4, w_k = exp(-(k-1)/gamma))
                           over sampled anchors; slot 0 (the clean anchor) is not supervised
Inference procedure:       block-parallel drafting; per round the target verifies
                           block_size-1 proposals and emits a bonus token on partial accept
Checkpoint format:         engine-side (speculators) HF-style speculator checkpoints;
                           this repo uses its own `vlm-dflash-ckpt-v1` container (see §14 of the GOAL)
```

## What we port

The training and verification semantics that are already frozen in this repo
(see `docs/REFERENCE.md` for the file-level table): anchor sampling, mask-token
denoising, block attention mask, decayed block CE, the speculative decode loop,
and the acceptance-length accounting `AL = new_tokens / verification_steps`
(bonus-inclusive).

The fork's own port originated from NeMo's DFlash core; the local modules are
`vlm_dflash/dflash/{draft_model,trainer_core,mask,decay_loss}.py`.

## What we change (and why)

- Multimodal plumbing (image-span features, MRoPE/spatial-rope draft positions,
  visual-path checks) — not present upstream, added here.
- Offline/staged feature training (target never resident during training) —
  upstream trains with the target online.
- `sample_from_anchor` support (see below) and the `input_shift` /
  `first_supervised_slot` trainer seam, added for DSpark; DFlash keeps the
  upstream default (`False`) and its semantics are unchanged.

## `sample_from_anchor` (semantics verified against upstream)

Speculators' DFlash doc states the two modes explicitly:

- `False` (DFlash default): anchor is the bonus token, only mask tokens predict,
  slot 0 is not trained -> `block_size - 1` speculative tokens per round.
- `True` (DSpark default, from the DSpark paper arXiv:2607.05147): sample from
  anchor and all mask positions -> `block_size` speculative tokens per round.

Our trainer mirrors this with `sample_from_anchor`: label window
`anchor + input_shift + k` with `input_shift = 1` and every slot supervised
(`first_supervised_slot = 0`) for `True`; round window `block_size + 1` at
inference. DFlash stays `False` (= the frozen 3B/8B baselines).

## Parity level

**Maintainer implementation** (via the fork's NeMo-derived port), with the
documented changes above. The frozen acceptance baselines in `BENCHMARKS.md`
(3B-512: AL 1.198; 3B-2K: 1.630; 3B-5K: 1.817; 8B-512: 1.322,
random-init ~1.003) are the regression contract for this repo: any refactor
that moves them beyond bf16 run-to-run noise is a bug.
