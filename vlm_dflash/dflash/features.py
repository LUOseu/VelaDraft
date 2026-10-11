"""Read captured feature shards and collate training batches."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..utils import read_json, read_jsonl


class FeatureStore:
    """Random access over ``features/shard_*.pt`` described by ``index.jsonl``."""

    def __init__(self, features_dir: str | Path):
        self.dir = Path(features_dir)
        self.rows = read_jsonl(self.dir / "index.jsonl")
        self.manifest = read_json(self.dir / "manifest.json")
        self._shard_name = None
        self._shard = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, name: str):
        if name != self._shard_name:
            self._shard = torch.load(self.dir / name, map_location="cpu", weights_only=False)
            self._shard_name = name
        return self._shard

    def get(self, index: int) -> dict:
        row = self.rows[index]
        shard = self._load_shard(row["shard"])
        return shard["samples"][row["shard_index"]]


class FeatureDataset(Dataset):
    def __init__(self, features_dir: str | Path):
        self.store = FeatureStore(features_dir)
        self.image_token_id = None

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, index: int) -> dict:
        record = self.store.get(index)
        return {
            "input_ids": record["input_ids"].to(torch.long),
            "loss_mask": record["loss_mask"].to(torch.float32),
            "hidden": record["hidden"],
            "position3": record["position3"].to(torch.long),
            "image_span": tuple(record["image_span"]),
            "sample_id": record["sample_id"],
        }


def collate_features(batch: list[dict], image_token_id: int) -> dict:
    """Pad a batch of variable-length feature records for the trainer module."""
    lengths = [item["input_ids"].numel() for item in batch]
    max_len = max(lengths)
    size = len(batch)
    hidden_size = batch[0]["hidden"].shape[-1]
    input_ids = torch.zeros(size, max_len, dtype=torch.long)
    loss_mask = torch.zeros(size, max_len, dtype=torch.float32)
    hidden = torch.zeros(size, max_len, hidden_size, dtype=batch[0]["hidden"].dtype)
    position3 = torch.zeros(3, size, max_len, dtype=torch.long)
    for row, item in enumerate(batch):
        length = lengths[row]
        input_ids[row, :length] = item["input_ids"]
        loss_mask[row, :length] = item["loss_mask"]
        hidden[row, :length] = item["hidden"]
        position3[:, row, :length] = item["position3"]
    image_mask = input_ids == image_token_id
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden": hidden,
        "position3": position3,
        "image_mask": image_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
