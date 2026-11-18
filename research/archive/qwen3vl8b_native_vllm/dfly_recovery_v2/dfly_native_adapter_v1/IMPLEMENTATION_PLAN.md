# DFly independent native mount

The authorized design is an owned DFlash-compatible model and proposer carrying
raw ordered `[C,6144]` target features without changing query hidden width 2048.
Each layer retains the original FC plus live BF16 softmax fusion before official
RMSNorm and shared original QKV K/V projection. Query execution remains stock,
noncausal B8. Slots 1..7 feed the original two-norm, three-linear hidden correction
with the actual preceding selected token; the first token is the target bonus.

Only this directory is writable. No GPU, installed vLLM edits, target/verifier
changes or shared harness edits are authorized at this stage. No Git repo exists.

- [x] Write and execute failing canonical context, correction, sequential proposal,
  strict schema, loader partition and export/launch guard tests.
- [x] Implement owned context/correction, backbone/model and raw-context proposer
  with official operators and independently owned compile metadata.
- [x] Strictly export original step37500 42 BF16 tensors into a new versioned
  directory; reopen and verify every shape, dtype and raw byte.
- [x] Execute actual ModelConfig, SpeculativeConfig and EngineArgs constructors
  for target/draft FLASH_ATTN, page16, TP1, maxseq1 and PIECEWISE capture[8].
- [x] Run synthetic and actual-trained-weight CPU component proofs against exact
  canonical methods; pin canonical and installed source hashes.
- [x] Bind production installation/model/proposer to explicit CPU receipt SHA,
  actual export/model/target paths and exactly three regular export files.
  Reject symlinks, shards/indexes, dummy/quantized/custom loaders and bypass flags.
- [x] Freeze source and evidence; self-audit and hand to root for spec/quality
  review before any GPU activity.
