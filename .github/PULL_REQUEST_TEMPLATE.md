## What changed

<!-- One or two sentences: the user problem this PR fixes. -->

## Checklist

- [ ] Facts match the code: I updated any affected default value, supported
      scope or command example in the same PR.
- [ ] `README.md` and `README.zh-CN.md` still share the same section skeleton
      and the same shared facts (public capability statements live in both).
- [ ] `python scripts/check_docs.py` passes locally (relative links, referenced
      repo paths, CLI flags in examples, generated-doc drift).
- [ ] Generated docs were changed through their generators
      (`scripts/render_support_matrix.py`, `scripts/render_training_signals.py`),
      never hand-edited.
- [ ] Numbers and commands state how they were verified: what was actually
      run, and what is a *recorded* result from the reference host (GPU steps
      that were not re-run are not marked "verified here").
- [ ] Single sources stayed single: algorithm/recipe states come from
      metadata, public checkpoints from the model-zoo index, numbers from
      records that carry a protocol / run id.
