# WP4 design - batched online capture + one profiled I/O optimization

Owner: C-01a13621 (see COORDINATION.md). Status: draft, implementation next.
Builds on the existing online source (`vlm_dflash/target/online.py`,
`feature_mode=online`); offline stores and the per-sample online path must stay
byte-compatible.

## Task (plan wording)

- batched online capture (the online target forward currently runs one sample
  per forward in `OnlineFeatureSource.extract`);
- at most one further I/O optimization chosen from profiling (not several);
- a measurement report with the profiling evidence.

## Design sketch

- Split `build_sample_record` into render (processor / prompt building) and
  forward (`forward_and_capture`) halves, so a batch can share one forward:
  rows in a batch are padded to the batch max, each keeps its own attention
  mask + MRoPE positions + image spans; records are split back per row and
  must be tensor-identical to the per-sample path on the same rows.
- `OnlineFeatureSource.extract_batch(indices) -> list[record]`: group size =
  1 (off by default; the per-sample path stays the reference) or N via
  `train.online_batch_size`. The dataset prefetches one batch window and hands
  out records in order; fast-forward placeholders stay per-item.
- Padding must not change records: verify by asserting equality against
  `extract()` on a smoke store (CPU test with a stub model + a GPU smoke).
- Measurement report: same rows, `online_batch_size` 1 vs N, same steps:
  `target_forwards_total`, `extraction_wall_seconds_total`, target peak, and
  per-phase wall time from `train_summary.json`; run on a small registered
  smoke (row count pinned before launch; budget in RESOURCE_LEDGER).
- I/O optimization: profile the extraction path first (image decode / prompt
  renders / caption tokens); pick at most one measured win (candidate: decoded
  image cache keyed by content hash, or processor-input prefetch). Re-run the
  same measurement with the optimization on; keep it only if it measures.

## Non-goals

- On-policy / rollout generation (this is teacher-forced extraction).
- Multi-process target serving, multi-node, DataLoader workers for the target.
- Changing offline stores or default-on behavior: batching is opt-in per run.
