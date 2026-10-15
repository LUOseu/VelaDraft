# Getting started: first install and first run

VelaDraft is the project name; the Python package and CLI are still called
`vlm-dflash` (see `pyproject.toml`). This page is the shortest verified path
from a fresh checkout to a first meaningful run. Historical measurements below
are labelled; they were recorded on the reference host and are not re-run here.

## 0. What you need

- **Linux + one NVIDIA GPU.** The validated reference is a single RTX A6000
  48 GB. A 24 GB preset exists but is a *suggested* configuration (no recorded
  run on a real 24 GB card yet). CPU-only machines can run the test suite, not
  training.
- **Python >= 3.10** and a virtualenv. The validated dependency combination is
  pinned in `pyproject.toml` (`.[tested]`) and mirrored in `docs/env-preflight.md`.
- **Disk**: ~7.1 GB for the Qwen2.5-VL-3B target (17 GB for Qwen3-VL-8B), plus
  feature stores and run directories (2K protocol: ~10 GB store, ~2-6 GB per
  run; see `docs/training-costs.md`).
- **A local target model directory.** The configs run with
  `local_files_only: true`, so the target must be downloaded before training -
  the pipeline will not fetch it for you.

## 1. Install (source install)

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tested]"
```

There is no PyPI package; the source install is the supported path today.
`pip install -e ".[tested]"` also installs the Hugging Face Hub client used below.

## 2. Prepare the target model

```bash
export VLM_DFLASH_TARGET="$PWD/assets/models/Qwen2.5-VL-3B-Instruct"
mkdir -p "$(dirname "$VLM_DFLASH_TARGET")"

# Verified client (huggingface_hub, installed with the package):
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$VLM_DFLASH_TARGET"

# Restricted networks: the same command through a mirror endpoint.
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$VLM_DFLASH_TARGET"
```

Verification status: the `hf download` client, the repo id and the file layout
were checked on 2026-10-15 with a metadata-only fetch (`--include config.json`)
through `hf-mirror.com`; the full 7.1 GB download is the recorded reference-host
procedure, not re-run for this page. The reference host itself used ModelScope
(`modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct ...`) because its
direct HF connectivity was throttled - both clients work if the model directory
ends up complete.

`Qwen3-VL-8B-Instruct` uses the same pattern (`Qwen/Qwen3-VL-8B-Instruct`,
exported to `${VLM_DFLASH_TARGET_QWEN3VL}` as `configs/qwen3_vl_8b.yaml` expects).

## 3. Preflight with `doctor`

```bash
vlm-dflash doctor --target "$VLM_DFLASH_TARGET"
```

Success looks like a `vlm-dflash doctor - python ... ok=True` header, exit code
0, `[ok]` lines for the target files/adapter/contract checks, and a resolved
contract line (`mask=151669 mrope=[16,24,24] layers=[1,17,33] ...`). A missing or
incomplete model directory fails here with the specific file it could not read.
`--json` prints the same report machine-readably; `--algorithm <name>` checks
one method's requirements too.

## 4. First run: the five-minute demo

No training and no external dataset - the demo synthesizes a tiny image sample:

```bash
vlm-dflash demo --target "$VLM_DFLASH_TARGET"
```

Recorded on the reference host (RTX A6000 48 GB, historical): **45 s
end-to-end including model load**, peak 8.2 GB, AL 1.000, 4/4 token-exact
(random-init drafter baseline).

Then score a released drafter (downloads one archive, ~387 MB):

```bash
curl -fL -O https://github.com/LUOseu/VelaDraft/releases/download/zoo-v0.3.0/qwen25vl-3b-dflash.tar.gz
tar -xzf qwen25vl-3b-dflash.tar.gz
vlm-dflash demo --target "$VLM_DFLASH_TARGET" --drafter qwen25vl-3b-dflash
```

Recorded: **42 s**, AL 1.313, 4/4 token-exact. The printed report ends with
`Results: <run_dir>/eval/eval_result.json` - that file is the successful
output: acceptance length, per-slot acceptance, and the token-exactness check
against the target's own greedy decoding.

To also exercise the full training pipeline on the same tiny sample:

```bash
vlm-dflash train-demo --target "$VLM_DFLASH_TARGET" --steps 80
```

Recorded: **3 min 13 s**, loss 13.0 -> 0.013 (tiny-sample overfit), eval AL
1.138 vs random-init 1.000, 4/4 token-exact, export written under
`runs/demo_dflash/export/`. This is a plumbing smoke, not a quality result.

## 5. Where to go next

- Train on your own image-text data: [data.md](data.md), then [training.md](training.md).
- Compare methods from one capture: `examples/reuse_features_compare.py` (see
  [../examples/README.md](../examples/README.md)).
- Live feature extraction: [online-training.md](online-training.md).
- Something failed: [troubleshooting.md](troubleshooting.md).
