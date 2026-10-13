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

Resume identity: the source records a **content digest** (``ONLINE_DIGEST_VERSION``)
over the consumed prompt/answer token ids, the per-row image bytes and the
extraction settings -- not just row lengths -- so a same-length edit of an
answer or a same-size image swap changes the digest instead of passing the
resume gate. Dtypes are made explicit: features are stored in ``capture.dtype``
and the frozen embedding/head handed to the trainer are cast to that same
storage dtype (mirroring ``capture.save_target_head``), while the target itself
keeps its native compute dtype.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..data.sharegpt import load_sharegpt, row_image_path
from ..provenance import identity_from_contract
from ..utils import read_json, sha256_file, sha256_json
from .load import load_target
from .sample_capture import build_sample_record, extract_target_head, storage_dtype

#: Version tag for the online content digest. Checkpoints that record a
#: different (or missing) version cannot be resumed exactly; the gate in
#: ``validate_resume_state`` refuses them instead of comparing digests from
#: two different schemes.
ONLINE_DIGEST_VERSION = "vlm-dflash-online-source-digest-v2"


def _manifest_image_hashes(run_dir: Path) -> dict[str, str]:
    """``id -> image_sha256`` from the data stage, when the manifest exists."""
    path = Path(run_dir) / "data" / "manifest.json"
    if not path.exists():
        return {}
    manifest = read_json(path)
    return {
        str(row.get("id")): str(row["image_sha256"])
        for row in manifest.get("rows", [])
        if row.get("image_sha256")
    }


def load_online_rows(run_dir: str | Path, limit: int | None = None) -> tuple[list[dict], Path]:
    """Rows and media dir an online run consumes (shared by source and preflight)."""
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
    return rows, media_dir


def online_source_digest(cfg: dict, contract: dict, rows: list[dict], media_dir, run_dir) -> str:
    """Content digest of the data a live-extraction run will actually consume.

    Covers the token ids of every consumed prompt/answer, the image bytes (not
    the path or the ``data/manifest.json`` claim -- the record is verified
    against the file and a stale manifest is a loud error) and the extraction
    settings. The same content copied to another directory keeps its digest;
    equal-length token edits or same-size image swaps change it.
    """
    run_dir = Path(run_dir)
    cap_cfg = cfg.get("capture", {}) or {}
    target_cfg = cfg.get("target", {}) or {}
    recorded_hashes = _manifest_image_hashes(run_dir)
    content = []
    for row in rows:
        image_path = row_image_path(row, Path(media_dir))
        if not image_path.exists():
            raise FileNotFoundError(
                f"feature_mode=online: image for row {row.get('id')!r} is missing ({image_path}); "
                "re-run the prepare/regen stages or point the run at the original media directory."
            )
        image_sha256 = sha256_file(image_path)
        recorded = recorded_hashes.get(str(row.get("id")))
        if recorded is not None and recorded != image_sha256:
            raise RuntimeError(
                f"feature_mode=online: image bytes for row {row.get('id')!r} no longer match "
                "data/manifest.json (the staged data changed after prepare). Re-run prepare/regen "
                "or start a new run directory."
            )
        content.append(
            {
                "prompt_ids": [int(token) for token in row["regen"]["prompt_ids"]],
                "answer_ids": [int(token) for token in row["regen"]["answer_ids"]],
                "image_sha256": image_sha256,
            }
        )
    identity = identity_from_contract(contract)
    identity.pop("target_model_path", None)
    return sha256_json(
        {
            "version": ONLINE_DIGEST_VERSION,
            "backend": OnlineFeatureSource.backend,
            "identity": identity,
            "content": content,
            "settings": {
                "seq_length": int(cap_cfg.get("seq_length", 2048)),
                "storage_dtype": str(storage_dtype(cap_cfg.get("dtype", "bfloat16"))).replace("torch.", ""),
                "store_final_hidden": bool(cap_cfg.get("store_final_hidden", False)),
                "draft_layer_ids": [int(layer) for layer in contract["layers"]["draft_layer_ids"]],
                "target_dtype": str(target_cfg.get("dtype", "bfloat16")),
                "attn_implementation": str(target_cfg.get("attn_implementation", "sdpa")),
            },
        }
    )


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
        # Defaults first, then the real body: ``close()`` must be safe even
        # when construction fails partway (the except below releases a
        # partially built source instead of leaving the target referenced).
        self._closed = False
        self.model = None
        self.processor = None
        self.embed_weight = None
        self.lm_head_weight = None
        self.stats: dict[str, dict] = {}
        self._peak_open = False
        try:
            self._init(cfg, contract, run_dir, limit=limit)
        except BaseException:
            self.close()
            raise

    def _init(self, cfg: dict, contract: dict, run_dir: str | Path, limit: int | None = None) -> None:
        run_dir = Path(run_dir)
        rows, media_dir = load_online_rows(run_dir, limit=limit)
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
        # Mirror capture.save_target_head: the trainer consumes the frozen
        # embed/head in the *feature storage dtype*, not in the target's native
        # compute dtype (an fp32 target with bf16 features would otherwise feed
        # fp32 head weights into a bf16 drafter). Only the copies handed to the
        # trainer are cast; the target model keeps its own precision.
        self.embed_weight = self.embed_weight.to(self.dtype)
        if self.lm_head_weight is not None:
            self.lm_head_weight = self.lm_head_weight.to(self.dtype)
        self.target_resident_params = sum(parameter.numel() for parameter in self.model.parameters())
        # Weights only (no activations/KV): the teacher's resident footprint on
        # its own device. When target.device == train.device the trainer's
        # reported peak already includes it; otherwise the two are separate.
        self.target_resident_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.model.parameters()
        )

        self.train_device = str((cfg.get("train") or {}).get("device", "cuda:0"))
        self.phase = "train"
        self.digest_value = online_source_digest(cfg, contract, self.rows, self.media_dir, self.run_dir)
        self._reset_target_peak()
        self._open_peak_window()

    # -- extraction bookkeeping ---------------------------------------------
    def _reset_target_peak(self) -> None:
        if self.target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.target_device)

    def _close_peak_window(self) -> None:
        # Tolerates partially-constructed instances (no target loaded yet).
        if not getattr(self, "_peak_open", False):
            return
        self._peak_open = False
        target_device = getattr(self, "target_device", None)
        if target_device is None or target_device.type != "cuda":
            return
        entry = self.stats.setdefault(self.phase, {"target_forwards": 0, "wall_seconds": 0.0})
        peak = torch.cuda.max_memory_allocated(target_device) / 1e9
        entry["target_device_peak_gb"] = max(float(entry.get("target_device_peak_gb", 0.0)), float(peak))

    def _open_peak_window(self) -> None:
        if self.target_device.type == "cuda":
            self._reset_target_peak()
            self._peak_open = True

    def set_phase(self, phase: str) -> None:
        self._close_peak_window()
        self.phase = phase
        self._open_peak_window()

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
        """Content digest for provenance/resume (never includes absolute paths).

        Computed by :func:`online_source_digest` over the consumed token ids,
        the per-row image bytes and the extraction settings; see
        ``ONLINE_DIGEST_VERSION`` for the scheme version recorded alongside it.
        """
        return self.digest_value

    def manifest(self) -> dict:
        return {
            "format": "vlm-dflash-online-source-v1",
            "digest_version": ONLINE_DIGEST_VERSION,
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
            "target_train_device": self.train_device,
            "digest": self.digest(),
        }

    def summary(self) -> dict:
        # Close an open peak window so the final phase's peak is recorded even
        # if ``close()`` never runs; idempotent afterwards.
        self._close_peak_window()
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
                "counters, a resumed run does not re-extract fast-forwarded items and reuses "
                "the recorded visual-path check instead of rerunning it"
            ),
            "target_device_peak_note": (
                "target_device_peak_gb is the CUDA peak on the target's device during that phase "
                "(reset per phase). When target.device == train.device the phase peak includes the "
                "trainer's own allocations; on split devices it isolates the target, which "
                "target_resident_bytes (weights only) cannot show."
            ),
        }

    def close(self) -> None:
        """Release the target, processor and frozen head copies; idempotent.

        Tolerates partially-constructed instances (the constructor calls this
        when loading fails after the target became resident). Dropping these
        references is what lets the GC reclaim the teacher; ``empty_cache`` is
        deliberately not relied on, because it cannot free objects that are
        still referenced.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._close_peak_window()
        self.model = None
        self.processor = None
        self.embed_weight = None
        self.lm_head_weight = None
