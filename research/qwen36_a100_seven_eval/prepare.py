"""Freeze the inputs of a run: target snapshot, the seven final checkpoints, and their vLLM exports.

Checkpoints come from the revisions pinned in ``research/qwen36_seven_20261102/checkpoint_manifest.json``
(never from a moving ``main``); a local copy is used only when its SHA-256 matches. Each checkpoint is
exported once with the repository's CPU exporter and then given its vLLM 0.32 dispatch config:
DFlash as exported, DFlash2/JetSpec via ``prepare_native``, DSpark via ``prepare_dspark``,
Domino/DFlare/DFly via ``prepare_custom``. All weight bytes stay unchanged (hard links).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from research.qwen36_a100_seven_eval.settings import (
    REPO,
    checkpoint_manifest,
    read_json,
    save_json,
    sha256,
)


def hashed(path: Path, cache: Path) -> str:
    """SHA-256 cached by (path, size, mtime_ns) so repeated preflights do not reread 70 GB."""
    stat = Path(path).stat()
    key = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    table = read_json(cache, {})
    if key not in table:
        table[key] = sha256(path)
        save_json(cache, table)
    return table[key]


def target_identity(path: Path, revision: str, cache: Path) -> dict:
    path = Path(path).resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"target snapshot not found at {path}")
    files = sorted(p for p in path.iterdir() if p.is_file())
    config = (
        json.loads((path / "generation_config.json").read_text())
        if (path / "generation_config.json").exists()
        else {}
    )
    eos = config.get("eos_token_id")
    stops = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    return {
        "path": str(path),
        "revision": revision,
        "files": {p.name: hashed(p, cache) for p in files},
        "stop_token_ids": stops,
    }


def fetch_checkpoint(
    method: str, entry: dict, local_candidates: list[Path], dest: Path, cache: Path
) -> Path:
    for candidate in local_candidates:
        if (
            candidate.is_file()
            and candidate.stat().st_size == entry["size"]
            and hashed(candidate, cache) == entry["sha256"]
        ):
            return candidate
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            entry["repo"],
            "ckpt/ckpt_final.pt",
            revision=entry["revision"],
            local_dir=dest / method,
        )
    )
    if hashed(path, cache) != entry["sha256"]:
        raise ValueError(
            f"{method}: downloaded checkpoint SHA-256 differs from the pinned manifest"
        )
    return path


def export(method: str, checkpoint: Path, root: Path) -> Path:
    """<root>/raw/<method> (CPU exporter) then <root>/vllm032/<method> (dispatch config)."""
    raw, final = root / "raw" / method, root / "vllm032" / method
    if final.exists():
        if not all(
            (final / name).is_file()
            for name in ("config.json", "model.safetensors", "ready.json")
        ):
            raise ValueError("incomplete export; use a fresh checkpoint work directory")
        receipt = read_json(final / "ready.json")
        if any(sha256(final / name) != digest for name, digest in receipt.items()):
            raise ValueError("export content differs from its receipt")
        return final
    if not (raw / "export_receipt.json").exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(REPO / "research/qwen36_native_vllm/export_checkpoint.py"),
                "--method",
                method,
                str(checkpoint),
                str(raw),
            ],
            check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO)),
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    destination = final
    staging = Path(tempfile.mkdtemp(prefix=f"{method}-", dir=final.parent))
    final = staging / "export"
    if method == "dflash":
        final.mkdir()
        for name in ("config.json", "export_receipt.json"):
            (final / name).write_bytes((raw / name).read_bytes())
        os.link(raw / "model.safetensors", final / "model.safetensors")
    elif method in ("dflash2", "jetspec"):
        from research.qwen36_seven_20261102.prepare_native import prepare

        prepare(method, raw, final)
    elif method == "dspark":
        from research.qwen36_seven_20261102.prepare_dspark import prepare

        prepare(raw, final)
    else:
        from research.qwen36_seven_20261102.prepare_custom import prepare

        prepare(method, raw, final)
    save_json(
        final / "ready.json",
        {name: sha256(final / name) for name in ("config.json", "model.safetensors")},
    )
    final.rename(destination)
    staging.rmdir()
    return destination


def freeze_checkpoints(methods, config: dict, cache: Path) -> dict:
    pinned = checkpoint_manifest()
    root = Path(config["checkpoints"]["work_dir"]).resolve()
    local_root = config["checkpoints"].get("local_root")
    out = {}
    for method in methods:
        entry = pinned[method]
        candidates = (
            [Path(local_root) / method / "train" / "ckpt_final.pt"]
            if local_root
            else []
        )
        checkpoint = fetch_checkpoint(
            method, entry, candidates, root / "checkpoints", cache
        )
        final = export(method, checkpoint, root / entry["sha256"])
        out[method] = {
            "repo": entry["repo"],
            "revision": entry["revision"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": entry["sha256"],
            "path": str(final),
            "config_sha256": sha256(final / "config.json"),
            "model_safetensors_sha256": hashed(final / "model.safetensors", cache),
        }
    return out


def verify_inputs(run: Path, manifest: dict) -> None:
    """Check frozen weights and configuration before launch or resume."""
    cache = run / "input_hash_cache.json"
    target = manifest["target"]
    for name, digest in target.get("files", {}).items():
        if hashed(Path(target["path"]) / name, cache) != digest:
            raise ValueError("target snapshot changed after freezing")
    for value in manifest["exports"].values():
        for name, key in (
            ("config.json", "config_sha256"),
            ("model.safetensors", "model_safetensors_sha256"),
        ):
            if key in value and hashed(Path(value["path"]) / name, cache) != value[key]:
                raise ValueError("draft export changed after freezing")
