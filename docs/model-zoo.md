# Model Zoo index

Downloadable drafter checkpoints, one row per published archive. All archives
are **pipeline-validation reference artifacts** (trained under documented
development protocols, exported self-describingly), not production-tuned
drafters. Sizes are the release assets in 10^6 bytes as reported by the GitHub
release API. Every archive reloads through `load_export`; the checks recorded
here are the *recorded* ones, not new measurements.

| Archive | Target | Method / recipe | Recorded check | Card |
| --- | --- | --- | --- | --- |
| `qwen25vl-3b-dflash.tar.gz` (387 MB, `zoo-v0.3.0`) | Qwen2.5-VL-3B | DFlash, shared block CE | canonical 2K protocol, AL 1.630, 128/128 token-exact | [card](zoo/qwen25vl-3b-dflash.md) |
| `qwen25vl-3b-domino.tar.gz` (464 MB, `zoo-v0.3.0`) | Qwen2.5-VL-3B | Domino, curriculum | canonical 2K protocol, AL 1.764, 128/128 token-exact | [card](zoo/qwen25vl-3b-domino.md) |
| `qwen25vl-3b-dspark.tar.gz` (511 MB, `zoo-v0.3.1`) | Qwen2.5-VL-3B | DSpark, CE + auxiliary heads | canonical 2K protocol, AL 1.735, 128/128 token-exact | [card](zoo/qwen25vl-3b-dspark.md) |
| `qwen25vl-3b-jetspec.tar.gz` (388 MB, `zoo-v0.3.1`) | Qwen2.5-VL-3B | JetSpec, forward-KL | canonical 2K protocol with the causal in-block mask (its serving flag), AL 1.599, 128/128 token-exact | [card](zoo/qwen25vl-3b-jetspec.md) |
| `qwen25vl-3b-dflash2-exp.tar.gz` (522 MB, `zoo-v0.3.1`) | Qwen2.5-VL-3B | DFlash2, **experimental reconstructed objective** | canonical 2K protocol, AL 1.954, 128/128 token-exact; not an upstream-objective claim | [card](zoo/qwen25vl-3b-dflash2-exp.md) |
| `qwen25vl-3b-dflare.tar.gz` (373 MB, `zoo-v0.3.1`) | Qwen2.5-VL-3B | DFlare, shared block CE | canonical 2K protocol, AL 1.637, 128/128 token-exact | [card](zoo/qwen25vl-3b-dflare.md) |
| `qwen25vl-3b-dfly.tar.gz` (420 MB, `zoo-v0.3.1`) | Qwen2.5-VL-3B | DFly, shared block CE (`controlled_ce`) | canonical 2K protocol, AL 1.682, 128/128 token-exact | [card](zoo/qwen25vl-3b-dfly.md) |
| `qwen25vl-3b-dfly-recipe.tar.gz` (420 MB, `zoo-v0.3.2`) | Qwen2.5-VL-3B | DFly, ported upstream two-stage recipe (`upstream_two_stage`: LK/D-PACE cold start, then e2e TV) | canonical 2K protocol, AL 1.839 vs 1.682 for the CE run on the same store/seed, 128/128 token-exact | [card](zoo/qwen25vl-3b-dfly-recipe.md) |
| `qwen3vl-8b-dflash.tar.gz` (1.00 GB, `zoo-v0.3.1`) | Qwen3-VL-8B | DFlash, shared block CE | full 2K pipeline, 32/32 token-exact (AL 1.665) | [card](zoo/qwen3vl-8b-dflash.md) |

The two DFly archives share the same architecture; the difference is the
training recipe recorded in `algorithm.json` / the model card
(`objective_profile`: `controlled_ce` vs `upstream_two_stage`). If you want the
trained-upstream-recipe variant, take the `zoo-v0.3.2` archive.

## Download and load

```bash
curl -fL -O https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.1/qwen25vl-3b-dfly.tar.gz
tar -xzf qwen25vl-3b-dfly.tar.gz

export VLM_DFLASH_TARGET=/path/to/Qwen2.5-VL-3B-Instruct   # local checkout
vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dfly
```

`demo --drafter` reports token-exactness, acceptance length and per-slot
acceptance against the target's own greedy output. `python
examples/score_pretrained.py` runs the same code path without the CLI.

Verify the download before use: each release carries `SHA256SUMS`, and every
archive contains a `checksums.json` for its files (see
`RELEASE_VALIDATION.md` for the recorded clean re-download checks).

## Notes

- **Release versions are distinct objects**: `v0.3.1` (code release) ships the
  packaging/first-run fixes; `zoo-v0.3.*` are dataset-style asset releases.
  The zoo version does not track the package version.
- Cards under `docs/zoo/` are copies of the archive-generated model cards:
  maturity/support fields inside them are the values recorded **at export
  time**; current machine-readable states live in
  [`SUPPORT_MATRIX.md`](../SUPPORT_MATRIX.md).
- Acceptance numbers depend on the exact target revision, processor and eval
  protocol recorded in each card (`BENCHMARKS.md` has the protocol tables).
- Serving integration is separate roadmap work; see
  [`serving-bridge.md`](serving-bridge.md). These archives are for
  evaluation/training interop with this repo's tooling.
