"""Read captured feature shards and collate training batches."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..utils import read_json, read_jsonl, sha256_file, sha256_json

#: Content digest version for feature stores. Bumping this refuses exact
#: resume across the change instead of silently comparing digests from two
#: different schemes (see ``offline_feature_digest``).
OFFLINE_DIGEST_VERSION = "vlm-dflash-offline-store-digest-v2"


def offline_feature_digest(features_dir: str | Path) -> str:
    """Content digest over everything a training run consumes from a store.

    Covers the actual shard bytes, the index, the frozen head and the
    content-bearing manifest fields; absolute paths and timestamps stay out so
    the same store copied to another directory keeps its identity, while a
    changed shard, swapped head or edited index changes it.
    """
    features_dir = Path(features_dir)
    store = FeatureStore(features_dir)
    manifest = {key: value for key, value in store.manifest.items() if key != "identity"}
    identity = dict(store.manifest.get("identity") or {})
    identity.pop("target_model_path", None)
    shard_names = sorted({row["shard"] for row in store.rows})
    return sha256_json(
        {
            "version": OFFLINE_DIGEST_VERSION,
            "manifest": manifest,
            "identity": identity,
            "index_sha256": sha256_file(features_dir / "index.jsonl"),
            "shards": {name: sha256_file(features_dir / name) for name in shard_names},
            "target_head_sha256": sha256_file(features_dir / "target_head.pt"),
        }
    )


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
        item = {
            "input_ids": record["input_ids"].to(torch.long),
            "loss_mask": record["loss_mask"].to(torch.float32),
            "hidden": record["hidden"],
            "position3": record["position3"].to(torch.long),
            "image_span": tuple(record["image_span"]),
            "sample_id": record["sample_id"],
        }
        # Capture-v2 stores additionally carry the target's final hidden state
        # (the teacher signal for distillation trainers). Old stores return None.
        if record.get("final_hidden") is not None:
            item["final_hidden"] = record["final_hidden"]
        return item


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
    # Only a fully capture-v2 batch carries the distillation teacher signal;
    # mixed batches would silently train JetSpec on zeros for part of the batch.
    has_final_hidden = all(item.get("final_hidden") is not None for item in batch)
    final_hidden = None
    if has_final_hidden:
        final_size = batch[0]["final_hidden"].shape[-1]
        final_hidden = torch.zeros(size, max_len, final_size, dtype=batch[0]["final_hidden"].dtype)
    for row, item in enumerate(batch):
        length = lengths[row]
        input_ids[row, :length] = item["input_ids"]
        loss_mask[row, :length] = item["loss_mask"]
        hidden[row, :length] = item["hidden"]
        position3[:, row, :length] = item["position3"]
        if has_final_hidden:
            final_hidden[row, :length] = item["final_hidden"]
    image_mask = input_ids == image_token_id
    collated = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden": hidden,
        "position3": position3,
        "image_mask": image_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    if final_hidden is not None:
        collated["final_hidden"] = final_hidden
    return collated
