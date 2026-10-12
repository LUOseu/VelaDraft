# Security policy

## Scope

This project trains and evaluates speculative drafters; it does not run a
network service. The relevant security surface is the local pipeline:

- loading model checkpoints / feature stores from disk;
- `datasets`-backed data preparation;
- exporting checkpoints (self-describing directories, no code execution).

## Reporting

Please report suspected issues privately (GitHub security advisory on this
repository, or email the maintainers) rather than in a public issue.

## Design notes relevant to safety

- `local_files_only: true` is the default for target loading; the pipeline
  never uploads data or weights anywhere.
- The export format is data-only (`model.safetensors` + JSON); loading an
  export never executes code from the checkpoint directory.
- `pickle`-based `.pt` run artifacts (checkpoints, feature shards) are trusted
  inputs: run them in a sandbox you control, as with any PyTorch checkpoint.
