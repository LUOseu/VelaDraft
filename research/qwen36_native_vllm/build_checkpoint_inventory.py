"""Checkpoint inventory for the seven Qwen3.6 rows (spec section 3).

  python build_checkpoint_inventory.py --runs <training task>/runs --exports <eval task>/exports --out inventory.json

Per method: training checkpoint path + sha256 (from the export receipt, i.e. the
bytes that were exported), step, epochs, train recipe, training-plan digest,
target identity (revision, fingerprints), run-config sha256, export config /
weights sha256, tensor count, HF mirror repo id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import contract
from common import METHODS, sha256

FINAL_STEP = 75000
EPOCHS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inventory = {"target": {"model": "Qwen/Qwen3.6-35B-A3B", "revision": contract.TARGET_REVISION,
                            "chat_template_kwargs": contract.TARGET_CHAT_TEMPLATE_KWARGS,
                            "image_pixels": [contract.IMAGE_MIN_PIXELS, contract.IMAGE_MAX_PIXELS]},
                 "protocol": "400K OneVision requests, 3 epochs x 25,000 updates, global batch 16, seed 42",
                 "methods": {}}
    for method in METHODS:
        export = args.exports / method
        receipt_path = export / "export_receipt.json"
        if not receipt_path.exists():
            inventory["methods"][method] = {"status": "not exported"}
            continue
        receipt = json.loads(receipt_path.read_text())
        run = args.runs / method
        config = run / "config.json"
        checkpoint = Path(receipt["source_checkpoint"])
        entry = {
            "status": "exported", "checkpoint": str(checkpoint), "checkpoint_sha256": receipt["source_sha256"],
            "step": receipt["step"], "epochs": EPOCHS if receipt["step"] == FINAL_STEP else None,
            "final_scheduled_endpoint": receipt["step"] == FINAL_STEP,
            "train_recipe": receipt.get("train_recipe"), "training_plan_digest": receipt.get("training_plan_digest"),
            "target_identity": receipt.get("target_identity"),
            "run_config": str(config), "run_config_sha256": sha256(config) if config.exists() else None,
            "export_dir": str(export), "export_config_sha256": receipt["config_sha256"],
            "export_model_safetensors_sha256": receipt["model_safetensors_sha256"],
            "tensor_count": receipt["tensor_count"], "all_tensors_bitwise_equal": receipt["all_tensors_bitwise_equal"],
            "config_changes": receipt["config_changes"],
            "hf_mirror": f"khazic/veladraft-qwen36-400k-{method}",
        }
        if sha256(checkpoint) != receipt["source_sha256"]:
            raise ValueError(f"{method}: checkpoint bytes changed since export")
        inventory["methods"][method] = entry
    steps = {m: e.get("step") for m, e in inventory["methods"].items() if e.get("status") == "exported"}
    inventory["equal_training_budget"] = len(set(steps.values())) == 1 and len(steps) == len(METHODS)
    args.out.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"methods": steps, "equal_training_budget": inventory["equal_training_budget"]}))


if __name__ == "__main__":
    main()
