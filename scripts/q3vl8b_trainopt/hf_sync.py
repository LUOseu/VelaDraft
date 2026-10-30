"""Incremental Hugging Face upload of the campaign data package (contract section 13).

    python hf_sync.py <campaign_dir> [--once] [--redact-gpu]

Every cycle (30 min): ``export_raw.py``, then the allow-listed part of the campaign
(raw records, tables, configs, provenance, manifests, checks, scripts, top-level
markdown) is mirrored into a node-local staging tree with every text file sanitized
and leak-checked, and only files whose content changed are committed to
``<repo>/training_system_eval/<campaign_id>/``. Checkpoints, optimizer states, target
weights and images are never uploaded; other paths of the repository are untouched.
Stops after a cycle that finds ``<campaign>/UPLOAD_FINAL`` (written when the campaign ends).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_raw import leaks, rules, sanitize

REPO_ID = "khazic/veladraft-qwen3vl8b-200k-dflash2"
REPO_TYPE = "model"
INTERVAL = 1800
# ``dir/**/*`` (not ``dir/**``, which Path.glob resolves to directories only).
ALLOW = [
    "raw/formal/**/*", "raw/windows/**/*", "raw/pilot/**/*", "raw/gpu_samples.jsonl.gz", "raw/*_gpu_time_ledger.jsonl",
    "raw/export_status.json", "tables/**/*", "configs/**/*", "provenance/**/*", "manifests/**/*", "checks/**/*",
    "scripts/**/*", "*.md", "schema.json", "artifacts_manifest.json",
]
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".py", ".sh", ".diff", ".log", ".csv", ".yaml"}
SKIP_SUFFIXES = {".srcmtime", ".pt", ".safetensors", ".bin", ".jpg", ".png"}


def say(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def collect(camp: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in ALLOW:
        files.update(path for path in camp.glob(pattern) if path.is_file())
    return sorted(path for path in files if path.suffix not in SKIP_SUFFIXES)


def staged_bytes(path: Path, redact_gpu: bool) -> bytes | None:
    """Sanitized bytes of one file, or None when it still leaks after sanitizing."""
    if path.suffix == ".gz":
        data = path.read_bytes()
        text = gzip.decompress(data).decode("utf-8", errors="replace")
        return None if leaks(text) or (redact_gpu and rules()[2].search(text)) else data
    if path.suffix in TEXT_SUFFIXES:
        text = sanitize(path.read_text(encoding="utf-8", errors="replace"), redact_gpu)
        return None if leaks(text) else text.encode("utf-8")
    return None


def cycle(api: HfApi, camp: Path, state: dict, redact_gpu: bool, stage: Path) -> None:
    export = [sys.executable, str(HERE / "export_raw.py"), str(camp)] + (["--redact-gpu"] if redact_gpu else [])
    result = subprocess.run(export, capture_output=True, text=True, check=False)
    say(f"export rc={result.returncode} {result.stdout.strip()[-300:]}")
    prefix = f"training_system_eval/{camp.name}"
    operations, blocked = [], []
    for path in collect(camp):
        rel = path.relative_to(camp).as_posix()
        data = staged_bytes(path, redact_gpu)
        if data is None:
            blocked.append(rel)
            continue
        digest = hashlib.sha256(data).hexdigest()
        if state.get(rel) == digest:
            continue
        local = stage / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        operations.append((rel, digest, CommitOperationAdd(path_in_repo=f"{prefix}/{rel}", path_or_fileobj=str(local))))
    if blocked:
        say(f"not uploaded (leak check): {blocked[:20]}{' ...' if len(blocked) > 20 else ''}")
    for start in range(0, len(operations), 200):
        batch = operations[start : start + 200]
        info = api.create_commit(
            repo_id=REPO_ID, repo_type=REPO_TYPE, operations=[op for _, _, op in batch],
            commit_message=f"training_system_eval/{camp.name}: {len(batch)} files",
        )
        for rel, digest, _ in batch:
            state[rel] = digest
        state["_last_commit"] = info.oid
        say(f"committed {len(batch)} files -> {info.oid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--redact-gpu", action="store_true")
    args = parser.parse_args()
    camp = args.campaign
    state_path = camp / "raw" / "hf_sync_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    stage = Path(os.environ["VELA_LOCAL"]) / "hf_stage" / camp.name
    api = HfApi()
    while True:
        final = (camp / "UPLOAD_FINAL").exists()
        try:
            cycle(api, camp, state, args.redact_gpu, stage)
            state_path.write_text(json.dumps(state, indent=1))
        except Exception as exc:  # network or Hub errors: retry next cycle
            say(f"cycle failed: {exc!r}")
        if args.once or final:
            say("HF_SYNC_STOP")
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
