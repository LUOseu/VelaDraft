"""Mechanical CPU export of the frozen 37,500-update draft to official HF layout.

The 36 trained tensors retain their names, shapes, dtypes and exact values.
Embedding/LM head are absent in the training checkpoint: official DFlash shares
the existing target's full embedding and LM head through its upstream loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from safetensors.torch import save_file, load_file

from official_spatial_adapter import MARKER, validate_draft_config

EXPECTED_CHECKPOINT_SHA256 = "86b40b550dae022d8f8669302022c50aaf36441c6f36c1f0adde6628f5c6135c"
REGISTERED_ARCHITECTURE = "DFlashDraftModel"


def validate_registered_architecture(config: dict) -> None:
    # The registry key differs from the Python implementation class name.
    # Checking the installed registry is CPU-only and prevents a late engine
    # startup failure after producing an apparently valid export.
    from vllm.model_executor.models import ModelRegistry
    architectures = config.get("architectures", [])
    if architectures != [REGISTERED_ARCHITECTURE]:
        raise ValueError(f"Official DFlash requires registry architecture {REGISTERED_ARCHITECTURE}")
    for architecture in architectures:
        if architecture not in ModelRegistry.get_supported_archs():
            raise ValueError(f"Architecture {architecture!r} is absent from the installed vLLM registry")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export(checkpoint: Path, output: Path) -> dict:
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash differs from the frozen parent37500 checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export destination must be empty: {output}")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if payload.get("algorithm") != "dflash" or payload.get("step") != 75000:
        raise ValueError("Expected the finished 75000-step DFlash checkpoint")
    config = payload["draft_config"]
    validate_draft_config(config, require_marker=False)
    weights = payload["model"]
    if len(weights) != 36:
        raise ValueError(f"Frozen checkpoint must have 36 trained tensors, found {len(weights)}")
    if any("embed_tokens" in name or "lm_head" in name for name in weights):
        raise ValueError("Frozen checkpoint unexpectedly includes embedding/head weights")
    config = json.loads(json.dumps(config))
    # These are dispatch metadata only; default RoPE frequencies and all weights
    # are unchanged. MRoPE makes trained spatial replace semantics explicit.
    config["architectures"] = [REGISTERED_ARCHITECTURE]
    config["dflash_config"][MARKER] = True
    config["rope_parameters"]["mrope_section"] = [24, 20, 20]
    config["rope_parameters"]["mrope_interleaved"] = True
    validate_draft_config(config)
    validate_registered_architecture(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model_file = output / "model.safetensors"
    save_file(weights, str(model_file), metadata={"format": "pt"})
    restored = load_file(str(model_file), device="cpu")
    tensors = {}
    if weights.keys() != restored.keys():
        raise AssertionError("Export changed tensor names")
    for name, weight in weights.items():
        other = restored[name]
        if weight.dtype != other.dtype or weight.shape != other.shape or not torch.equal(weight, other):
            raise AssertionError(f"Export changed tensor {name}")
        tensors[name] = {"shape": list(weight.shape), "dtype": str(weight.dtype),
                         "sha256": hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if torch.cuda.is_initialized():
        raise AssertionError("CPU-only exporter initialized CUDA")
    receipt = {"source_checkpoint": str(checkpoint), "source_sha256": checkpoint_hash,
               "config_sha256": sha256_file(output / "config.json"),
               "registered_architecture": REGISTERED_ARCHITECTURE,
               "architecture_checked_against_installed_registry": True,
               "model_safetensors_sha256": sha256_file(model_file),
               "tensor_count": len(weights), "all_tensors_bitwise_equal": True,
               "cuda_initialized": False, "config_changes": ["architectures", "dflash_config." + MARKER,
                   "rope_parameters.mrope_section", "rope_parameters.mrope_interleaved"],
               "head_and_embedding": "Absent in source; official upstream target sharing retained",
               "tensors": tensors}
    (output / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    receipt = export(args.checkpoint, args.output)
    print(json.dumps({k: v for k, v in receipt.items() if k != "tensors"}, indent=2))
