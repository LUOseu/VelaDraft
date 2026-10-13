"""Online target feature source: live teacher-forced extraction during training.

``feature_mode=online`` runs the frozen target VLM in the same process as the
trainer and extracts exactly the records the staged pipeline would have
written to a feature store (one per regenerated row; see
:mod:`vlm_dflash.target.sample_capture`). The regen answers remain the
supervision targets -- this is *not* on-policy rollout generation, not
continual learning from live traffic, and not a serving runtime.

Scope of the first version (v0.3.2):

- one Hugging Face target, one process; target and drafter may share a GPU or
  live on two devices of the same machine (``target.device`` / ``train.device``);
- per-sample extraction, merged into batches by the shared collator. The
  target forward never happens in a DataLoader worker (``num_workers`` must be
  0); remote producers, batching the capture forward and multi-node training
  are later stages.

Costs are real: the target stays resident for the whole run and every epoch
re-runs the target forward over the training rows. Extraction counters and
wall time are recorded per phase (train / overfit sanity / calibration) so
the numbers in the run summary stay explainable; the startup visual-path
check runs a fixed 3 capture forwards that are not part of those counters.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..data.sharegpt import load_sharegpt
from ..provenance import identity_from_contract
from ..utils import sha256_json
from .load import load_target
from .sample_capture import build_sample_record, extract_target_head, storage_dtype


class OnlineFeatureDataset(Dataset):
    """Per-sample live extraction with the offline record contract.

    ``__getitem__`` runs one teacher-forced target forward and returns the same
    fields (``input_ids`` / ``hidden`` / ``loss_mask`` / ``position3`` /
    ``image_span`` / ``sample_id`` / optional ``final_hidden``) that
    ``FeatureDataset`` reads from a captured store, with the same dtypes, so
    both sources feed the identical collator and training loop.
    """

    #: Guard read by the training loop: this dataset requires the main process.
    require_main_process = True

    def __init__(self, source: "OnlineFeatureSource"):
        self.source = source
        self._placeholder_remaining = 0

    def __len__(self) -> int:
        return len(self.source.rows)

    # -- resume fast-forward ------------------------------------------------
    def start_fast_forward(self, items: int) -> None:
        """Answer the next ``items`` items with cheap placeholders (no target call).

        Used by the training loop when resuming: the loader still replays the
        exact shuffle order, but already-consumed items are skipped without
        re-running the target. Placeholders consume no RNG and never reach the
        optimizer (the loop discards them).
        """
        self._placeholder_remaining += max(0, int(items))

    def end_fast_forward(self) -> None:
        self._placeholder_remaining = 0

    # -- extraction ----------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError(
                "OnlineFeatureDataset must run in the main process; feature_mode=online requires "
                "train.num_workers=0 (the target model cannot be copied into DataLoader workers)."
            )
        if self._placeholder_remaining > 0:
            self._placeholder_remaining -= 1
            return self._placeholder_record(str(index))
        record = self.source.extract(int(index))
        return self._normalize(record)

    def _placeholder_record(self, index: str) -> dict:
        length = 2
        return {
            "input_ids": torch.zeros(length, dtype=torch.long),
            "loss_mask": torch.zeros(length, dtype=torch.float32),
            "hidden": torch.zeros(length, self.source.fused_hidden_size, dtype=self.source.dtype),
            "position3": torch.zeros(3, length, dtype=torch.long),
            "image_span": (0, 0),
            "sample_id": f"fast-forward:{index}",
        }

    @staticmethod
    def _normalize(record: dict) -> dict:
        """Copy target outputs out of inference mode into normal CPU tensors.

        ``detach()`` alone is not enough: tensors produced inside
        ``torch.inference_mode()`` stay *inference tensors* and cannot be saved
        for backward. Cloning here runs outside inference mode, which produces
        normal tensors the drafter can backprop through, and keeps the online
        record byte-compatible with the offline store's dtypes.
        """
        out = {
            "input_ids": record["input_ids"].detach().to(torch.long).clone(),
            "loss_mask": record["loss_mask"].detach().to(torch.float32).clone(),
            "hidden": record["hidden"].detach().clone(),
            "position3": record["position3"].detach().to(torch.long).clone(),
            "image_span": tuple(record["image_span"]),
            "sample_id": record.get("sample_id"),
        }
        if record.get("final_hidden") is not None:
            out["final_hidden"] = record["final_hidden"].detach().clone()
        return out


class OnlineFeatureSource:
    """Owns the frozen target + processor and hands out datasets that extract from it."""

    backend = "huggingface"

    def __init__(self, cfg: dict, contract: dict, run_dir: str | Path, limit: int | None = None):
        run_dir = Path(run_dir)
        rows, media_dir = load_sharegpt(
            run_dir / "regen" / "target_data.jsonl", meta_path=run_dir / "data" / "meta.json"
        )
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(
                "feature_mode=online needs at least one regenerated row (regen/target_data.jsonl); "
                "run the prepare + regen stages first."
            )
        self.run_dir = run_dir
        self.rows = rows
        self.media_dir = media_dir
        self.contract = contract
        cap_cfg = cfg.get("capture", {}) or {}
        self.seq_length = int(cap_cfg.get("seq_length", 2048))
        self.dtype = storage_dtype(cap_cfg.get("dtype", "bfloat16"))
        self.store_final_hidden = bool(cap_cfg.get("store_final_hidden", False))
        self.layer_ids = [int(layer) for layer in contract["layers"]["draft_layer_ids"]]
        self.image_token_id = int(contract["tokens"]["image_token_id"])
        self.merge_size = int(contract["image"]["merge_size"])
        self.fused_hidden_size = len(self.layer_ids) * int(contract["text"]["hidden_size"])
        target_cfg = cfg.get("target", {}) or {}
        self.target_dtype = str(target_cfg.get("dtype", "bfloat16"))
        self.attn_implementation = str(target_cfg.get("attn_implementation", "sdpa"))

        self.model, self.processor = load_target(cfg, contract)
        self.target_device = next(self.model.parameters()).device
        offenders = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if offenders or self.model.training:
            raise RuntimeError(
                "feature_mode=online target is not frozen: "
                f"training={self.model.training}, requires_grad tensors={offenders[:5]}. "
                "The teacher must stay in eval() with no trainable parameters."
            )
        self.embed_weight, self.lm_head_weight, self.tied_head = extract_target_head(self.model, contract)
        self.target_resident_params = sum(parameter.numel() for parameter in self.model.parameters())
        # Weights only (no activations/KV): the teacher's resident footprint on
        # its own device. When target.device == train.device the trainer's
        # reported peak already includes it; otherwise the two are separate.
        self.target_resident_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )

        self.phase = "train"
        self.stats: dict[str, dict] = {}
        self._dataset_digest = sha256_json(
            {
                "rows": [
                    {
                        "id": row.get("id"),
                        "prompt_len": len(row["regen"]["prompt_ids"]),
                        "answer_len": len(row["regen"]["answer_ids"]),
                    }
                    for row in self.rows
                ],
                "seq_length": self.seq_length,
                "storage_dtype": str(self.dtype).replace("torch.", ""),
                "store_final_hidden": self.store_final_hidden,
            }
        )

    # -- extraction bookkeeping ---------------------------------------------
    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def extract(self, index: int) -> dict:
        started = time.perf_counter()
        record = build_sample_record(
            self.model,
            self.processor,
            self.rows[index],
            self.media_dir,
            layer_ids=self.layer_ids,
            image_token_id=self.image_token_id,
            merge_size=self.merge_size,
            dtype=self.dtype,
            seq_length=self.seq_length,
            store_final_hidden=self.store_final_hidden,
        )
        entry = self.stats.setdefault(self.phase, {"target_forwards": 0, "wall_seconds": 0.0})
        entry["target_forwards"] += 1
        entry["wall_seconds"] += time.perf_counter() - started
        return record

    # -- trainer-facing surface ----------------------------------------------
    def dataset(self) -> OnlineFeatureDataset:
        return OnlineFeatureDataset(self)

    def digest(self) -> str:
        """Content digest for provenance/resume (never includes absolute paths)."""
        identity = identity_from_contract(self.contract)
        identity.pop("target_model_path", None)
        return sha256_json(
            {
                "feature_mode": "online",
                "backend": self.backend,
                "identity": identity,
                "dataset": self._dataset_digest,
                "draft_layer_ids": self.layer_ids,
                "target_dtype": self.target_dtype,
                "attn_implementation": self.attn_implementation,
            }
        )

    def manifest(self) -> dict:
        return {
            "format": "vlm-dflash-online-source-v1",
            "feature_mode": "online",
            "backend": self.backend,
            "num_samples": len(self.rows),
            "seq_length": self.seq_length,
            "storage_dtype": str(self.dtype).replace("torch.", ""),
            "stores_final_hidden": self.store_final_hidden,
            "draft_layer_ids": self.layer_ids,
            "target_dtype": self.target_dtype,
            "attn_implementation": self.attn_implementation,
            "target_device": str(self.target_device),
            "digest": self.digest(),
        }

    def summary(self) -> dict:
        total_forwards = sum(entry["target_forwards"] for entry in self.stats.values())
        total_wall = sum(entry["wall_seconds"] for entry in self.stats.values())
        return {
            **self.manifest(),
            "target_resident_params": self.target_resident_params,
            "target_resident_bytes": self.target_resident_bytes,
            "extractions": {phase: dict(entry) for phase, entry in sorted(self.stats.items())},
            "target_forwards_total": total_forwards,
            "extraction_wall_seconds_total": total_wall,
            "extraction_wall_seconds_note": (
                "per-phase counters cover train / overfit sanity / calibration extraction; the "
                "startup visual-path check runs a fixed 3 capture forwards outside these "
                "counters, and a resumed run does not re-extract fast-forwarded items"
            ),
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
