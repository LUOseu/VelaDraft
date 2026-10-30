"""Mechanical CPU export of a finished Qwen3.6 draft to the native layout.

Every trained tensor keeps its name, shape, dtype and exact bytes (one
model.safetensors). Embedding and LM head are absent in the training
checkpoint: the official DFlash loader shares the target's own. Only dispatch
metadata changes in config.json (listed in the receipt's config_changes).

  python export_checkpoint.py --method dflash <ckpt_final.pt> <empty_out_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256 as sha256_file
import contract
import dflare_adapter
import dflash2_adapter
import dfly_adapter
import head_adapters
import jetspec_adapter

FINAL_STEP = 75000
DFLASH_ARCHITECTURE = "DFlashDraftModel"
DFLASH_TENSOR_COUNT = 36
# method (also the checkpoint's training algorithm field) -> (trained tensor count,
# vLLM architecture, method markers); adapter methods take their own constants.
METHODS = {
    "dflash": (DFLASH_TENSOR_COUNT, DFLASH_ARCHITECTURE, ()),
    "jetspec": (DFLASH_TENSOR_COUNT, DFLASH_ARCHITECTURE, (jetspec_adapter.MARKER,)),
    **{name: (adapter.TENSOR_COUNT, adapter.ARCH, (adapter.MARKER,))
       for name, adapter in (("dfly", dfly_adapter), ("dflare", dflare_adapter), ("dflash2", dflash2_adapter))},
    **{name: (count, arch, (marker,)) for name, (arch, marker, count, _) in head_adapters.METHODS.items()},
}


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def native_config(config: dict, method: str) -> tuple[dict, list[str]]:
    """Dispatch-only config edits; trained hyperparameters stay untouched."""
    contract.validate_draft_config(config, require_marker=False)
    _, architecture, markers = METHODS[method]
    config = json.loads(json.dumps(config))
    config["architectures"] = [architecture]
    for marker in (contract.MARKER, *markers):
        config["dflash_config"][marker] = True
    config["rope_parameters"]["mrope_section"] = list(contract.MROPE_SECTION)
    config["rope_parameters"]["mrope_interleaved"] = True
    contract.validate_draft_config(config)
    return config, ["architectures", *("dflash_config." + m for m in (contract.MARKER, *markers)),
                    "rope_parameters.mrope_section", "rope_parameters.mrope_interleaved"]


def export(method: str, checkpoint: Path, output: Path) -> dict:
    algorithm = method
    tensor_count = METHODS[method][0]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export destination must be empty: {output}")
    checkpoint_hash = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if payload.get("algorithm") != algorithm or payload.get("step") != FINAL_STEP:
        raise ValueError(f"Expected the finished {FINAL_STEP}-step {algorithm} checkpoint, "
                         f"got {payload.get('algorithm')}@{payload.get('step')}")
    identity = payload.get("identity", {})
    if identity.get("mask_token_id") != contract.MASK_TOKEN_ID or payload.get("target_layer_ids") != contract.TARGET_LAYER_IDS:
        raise ValueError("Checkpoint identity differs from the frozen Qwen3.6 contract")
    weights = payload["model"]
    if len(weights) != tensor_count:
        raise ValueError(f"{method} checkpoint must have {tensor_count} trained tensors, found {len(weights)}")
    if any("embed_tokens" in name or "lm_head" in name for name in weights):
        raise ValueError("Checkpoint unexpectedly includes embedding/head weights")
    config, changes = native_config(payload["draft_config"], method)
    if config["architectures"] == [DFLASH_ARCHITECTURE]:
        from vllm.model_executor.models import ModelRegistry
        if any(arch not in ModelRegistry.get_supported_archs() for arch in config["architectures"]):
            raise ValueError(f"{config['architectures']} absent from the installed vLLM registry")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model_file = output / "model.safetensors"
    save_file({k: v.contiguous() for k, v in weights.items()}, str(model_file), metadata={"format": "pt"})
    restored = load_file(str(model_file), device="cpu")
    if weights.keys() != restored.keys():
        raise AssertionError("Export changed tensor names")
    tensors = {}
    for name, weight in weights.items():
        other = restored[name]
        if weight.dtype != other.dtype or weight.shape != other.shape or not torch.equal(
                weight.contiguous().reshape(-1).view(torch.uint8), other.contiguous().reshape(-1).view(torch.uint8)):
            raise AssertionError(f"Export changed tensor {name}")
        tensors[name] = {"shape": list(weight.shape), "dtype": str(weight.dtype), "sha256": tensor_sha256(weight)}
    if torch.cuda.is_initialized():
        raise AssertionError("CPU-only exporter initialized CUDA")
    receipt = {
        "method": method, "source_checkpoint": str(checkpoint), "source_sha256": checkpoint_hash,
        "step": payload["step"], "training_plan_digest": payload.get("training_plan_digest"),
        "target_identity": identity, "train_recipe": payload.get("train_recipe"),
        "config_sha256": sha256_file(output / "config.json"),
        "model_safetensors_sha256": sha256_file(model_file),
        "tensor_count": len(weights), "all_tensors_bitwise_equal": True, "cuda_initialized": False,
        "config_changes": changes,
        "head_and_embedding": "absent in source; official upstream target sharing retained",
        "tensors": tensors,
    }
    (output / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = export(args.method, args.checkpoint, args.output)
    print(json.dumps({k: v for k, v in receipt.items() if k != "tensors"}, indent=2))
