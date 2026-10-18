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

import math
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..data.sharegpt import load_sharegpt, row_image_paths
from ..provenance import identity_from_contract
from ..utils import read_json, sha256_file, sha256_json
from .cache_budget import CacheBudgetController
from .load import load_target
from .sample_capture import (
    extract_target_head,
    finalize_prepared_records,
    prepare_sample_record,
    storage_dtype,
)

#: Version tag for the online content digest. Checkpoints that record a
#: different (or missing) version cannot be resumed exactly; the gate in
#: ``validate_resume_state`` refuses them instead of comparing digests from
#: two different schemes.
ONLINE_DIGEST_VERSION = "vlm-dflash-online-source-digest-v2"

#: Fraction of the configured cap the static pinned plan packs into
#: (``train.online_feature_cache_policy: pinned``). The greedy smallest-first
#: pack stops here, leaving headroom so a byte-cap eviction never has to drop
#: a pinned record. Same 0.95 fill as the validated research plan
#: (research/training_systems/scripts/s4_pinned_plan.py).
PINNED_PLAN_FILL_FRACTION = 0.95


def _record_nbytes(record: dict) -> int:
    """Bytes of the tensor payload of one extracted record (cache accounting)."""
    total = 0
    for value in record.values():
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
    return total


def _manifest_image_hashes(run_dir: Path) -> dict[str, str]:
    """``id -> image_sha256`` (or the per-image list for multi-image rows) from the data stage."""
    path = Path(run_dir) / "data" / "manifest.json"
    if not path.exists():
        return {}
    manifest = read_json(path)
    out: dict = {}
    for row in manifest.get("rows", []):
        if row.get("image_sha256_list"):
            out[str(row.get("id"))] = [str(value) for value in row["image_sha256_list"]]
        elif row.get("image_sha256"):
            out[str(row.get("id"))] = str(row["image_sha256"])
    return out


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
        image_paths = row_image_paths(row, Path(media_dir))
        image_hashes = []
        for image_path in image_paths:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"feature_mode=online: image for row {row.get('id')!r} is missing ({image_path}); "
                    "re-run the prepare/regen stages or point the run at the original media directory."
                )
            image_hashes.append(sha256_file(image_path))
        recorded = recorded_hashes.get(str(row.get("id")))
        if recorded is not None:
            expected = [recorded] if isinstance(recorded, str) else list(recorded)
            if expected != image_hashes:
                raise RuntimeError(
                    f"feature_mode=online: image bytes for row {row.get('id')!r} no longer match "
                    "data/manifest.json (the staged data changed after prepare). Re-run prepare/regen "
                    "or start a new run directory."
                )
        regen = row["regen"]
        if "turns" in regen:
            entry = {
                "turns": [
                    {
                        "prompt_ids": [int(token) for token in turn["prompt_ids"]],
                        "answer_ids": [int(token) for token in turn["answer_ids"]],
                    }
                    for turn in regen["turns"]
                ],
                "full_ids": [int(token) for token in regen["full_ids"]],
                "image_sha256": image_hashes,
            }
            # Supervision protocol of multi-turn rows. Only rows that record a
            # mode fold it in, so digests of legacy (mode-less, all-turn)
            # artifacts stay byte-identical; a regenerated row under a different
            # mode can never silently pass an exact-resume gate.
            if regen.get("mode") is not None:
                entry["multi_turn_mode"] = str(regen["mode"])
            content.append(entry)
        else:
            content.append(
                {
                    "prompt_ids": [int(token) for token in regen["prompt_ids"]],
                    "answer_ids": [int(token) for token in regen["answer_ids"]],
                    "image_sha256": image_hashes[0],
                }
            )
    identity = identity_from_contract(contract)
    identity.pop("target_model_path", None)
    settings = {
        "seq_length": int(cap_cfg.get("seq_length", 2048)),
        "storage_dtype": str(storage_dtype(cap_cfg.get("dtype", "bfloat16"))).replace("torch.", ""),
        "store_final_hidden": bool(cap_cfg.get("store_final_hidden", False)),
        "draft_layer_ids": [int(layer) for layer in contract["layers"]["draft_layer_ids"]],
        "target_dtype": str(target_cfg.get("dtype", "bfloat16")),
        "attn_implementation": str(target_cfg.get("attn_implementation", "sdpa")),
    }
    # The per-sample path keeps its original digest scheme (no extra field for
    # batch size 1). A batched run folds its batch size into the digest: batch
    # composition changes the padding/numerics, so an exact resume must not
    # silently switch to a different grouping.
    online_batch_size = int((cfg.get("train") or {}).get("online_batch_size", 1) or 1)
    if online_batch_size > 1:
        settings["online_batch_size"] = online_batch_size
    return sha256_json(
        {
            "version": ONLINE_DIGEST_VERSION,
            "backend": OnlineFeatureSource.backend,
            "identity": identity,
            "content": content,
            "settings": settings,
        }
    )


class OnlineFeatureDataset(Dataset):
    """Per-sample live extraction with the offline record contract.

    ``__getitem__`` runs one teacher-forced target forward and returns the same
    fields (``input_ids`` / ``hidden`` / ``loss_mask`` / ``position3`` /
    ``image_span`` / ``sample_id`` / optional ``final_hidden``) that
    ``FeatureDataset`` reads from a captured store, with the same dtypes, so
    both sources feed the identical collator and training loop.

    ``fetch_batch`` serves the same records for a list of indices with one
    batched target forward; the trainer's online iterator uses it when
    ``train.online_batch_size > 1`` (see ``OnlineBatchIterator``).
    """

    #: Guard read by the training loop: this dataset requires the main process.
    require_main_process = True

    @property
    def batched_capture(self) -> bool:
        """Whether the trainer should fetch via ``fetch_batch`` (train.online_batch_size > 1)."""
        return self.source.online_batch_size > 1

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

    def note_epoch_boundary(self) -> dict | None:
        """Forward the training loop's natural epoch-end signal to the source.

        The cache policy uses it to freeze the static pinned plan at the first
        train-phase epoch end; for every other configuration it is a no-op.
        """
        return self.source.note_epoch_boundary()

    # -- extraction ----------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError(
                "OnlineFeatureDataset must run in the main process; feature_mode=online requires "
                "train.num_workers=0 (the target model cannot be copied into DataLoader workers)."
            )
        if self._placeholder_remaining > 0:
            self._placeholder_remaining -= 1
            self.source.note_fast_forward()
            return self._placeholder_record(str(index))
        cached = self.source.cache_lookup(int(index))
        if cached is not None:
            return cached
        record = self.source.extract(int(index))
        out = self._normalize(record)
        self.source.cache_store(int(index), out)
        return out

    def fetch_batch(self, indices: list[int]) -> list[dict]:
        """Records for ``indices``, in order, with one batched target forward.

        Every real item is rendered first (no target call), then finalized in
        a single padded forward; fast-forward placeholders stay per-item and
        consume no target work. Output order matches ``indices`` exactly, so
        the collated batch is identical to fetching the same indices one by
        one.
        """
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError(
                "OnlineFeatureDataset must run in the main process; feature_mode=online requires "
                "train.num_workers=0 (the target model cannot be copied into DataLoader workers)."
            )
        records: list[dict | None] = [None] * len(indices)
        prepared, positions = [], []
        for position, index in enumerate(indices):
            if self._placeholder_remaining > 0:
                self._placeholder_remaining -= 1
                self.source.note_fast_forward()
                records[position] = self._placeholder_record(str(index))
                continue
            cached = self.source.cache_lookup(int(index))
            if cached is not None:
                records[position] = cached
                continue
            prepared.append(self.source.prepare(int(index)))
            positions.append(position)
        if prepared:
            extracted = self.source.extract_prepared(prepared)
            for position, record in zip(positions, extracted):
                out = self._normalize(record)
                self.source.cache_store(int(indices[position]), out)
                records[position] = out
        missing = [position for position, record in enumerate(records) if record is None]
        if missing:
            raise RuntimeError(f"fetch_batch left positions {missing} unresolved.")
        return records  # type: ignore[return-value]

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
        if record.get("image_spans") is not None:
            out["image_spans"] = [tuple(span) for span in record["image_spans"]]
            out["n_turns"] = int(record.get("n_turns", 1))
        if record.get("turn_spans") is not None:
            out["turn_spans"] = [tuple(span) for span in record["turn_spans"]]
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
        self.target_devices: list[torch.device] = []
        # Bounded record-cache defaults (replaced by _init when it runs); the
        # constructor's failure path calls close(), which must be safe even
        # when the cache was never built.
        self.cache_capacity_bytes = 0
        self.cache_policy = "lru"
        self._record_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._record_cache_bytes = 0
        self._record_sizes: dict[int, int] = {}
        self._pinned_plan: dict | None = None
        self._pinned_set: set[int] = set()
        self._epoch_boundaries = 0
        self._num_epochs_planned = 0
        self.cache_stats: dict = {
            "capacity_bytes": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "skipped_too_large": 0,
            "peak_bytes": 0,
            "admitted_records": 0,
            "rejected_records": 0,
            "purged_records": 0,
            "purged_bytes": 0,
        }
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
        from ..contract import contract_capture_layer_ids

        self.layer_ids = contract_capture_layer_ids(contract)
        self.hidden_size = int(contract["text"]["hidden_size"])
        self.image_token_id = int(contract["tokens"]["image_token_id"])
        self.merge_size = contract["image"].get("merge_size")
        self.fused_hidden_size = len(self.layer_ids) * int(contract["text"]["hidden_size"])
        # Rows per batched capture forward; 1 is the per-sample reference path.
        self.online_batch_size = max(1, int((cfg.get("train") or {}).get("online_batch_size", 1) or 1))
        # Optional bounded host-RAM retention of extracted records across
        # epochs (``train.online_feature_cache_gb``; 0 = disabled, the
        # reference behavior). Hits serve a row from RAM instead of re-running
        # the target and return the stored tensors bitwise; a miss (cold
        # start, eviction, oversized record) re-extracts through the reference
        # extraction path -- with online_batch_size > 1 a partially cached
        # group forwards only its missing rows, so the forward's shape can
        # differ from a fully uncached run's. The cache changes neither the
        # consumed rows nor their order; a hit reuses the first-extraction
        # bytes, so under a shuffling loader cached runs drift from
        # re-extracting ones at the bf16-rounding scale (measured in
        # docs/online-training.md).
        # Runtime-only, never persisted, LRU-evicted at a byte cap; it stays
        # out of every digest/plan gate, and a resume simply starts cold.
        cache_gb = (cfg.get("train") or {}).get("online_feature_cache_gb", 0) or 0
        try:
            cache_gb = float(cache_gb)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "train.online_feature_cache_gb must be a number of GiB (0 disables the cache), "
                f"got {cache_gb!r}."
            ) from exc
        if cache_gb < 0:
            raise ValueError("train.online_feature_cache_gb must be >= 0 (0 disables the cache).")
        self.cache_capacity_bytes = int(cache_gb * (1 << 30))
        # Retention policy inside the byte budget: ``lru`` (default, byte-for-
        # byte the shipped behavior) or ``pinned`` (static smallest-first
        # retention frozen at the first train-phase epoch end; see
        # note_epoch_boundary). Only meaningful when the cache is enabled.
        policy = (cfg.get("train") or {}).get("online_feature_cache_policy", "lru")
        if policy is None or policy == "":
            policy = "lru"
        if not isinstance(policy, str) or policy not in ("lru", "pinned"):
            raise ValueError(
                "train.online_feature_cache_policy must be 'lru' or 'pinned', "
                f"got {policy!r}."
            )
        self.cache_policy = policy
        planned_epochs = (cfg.get("train") or {}).get("num_epochs", 6)
        self._num_epochs_planned = int(planned_epochs) if planned_epochs is not None else 6
        self._record_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._record_cache_bytes = 0
        self._record_sizes = {}
        self._pinned_plan = None
        self._pinned_set = set()
        self._epoch_boundaries = 0
        self.cache_stats: dict = {
            "capacity_bytes": self.cache_capacity_bytes,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "skipped_too_large": 0,
            "peak_bytes": 0,
            "admitted_records": 0,
            "rejected_records": 0,
            "purged_records": 0,
            "purged_bytes": 0,
        }
        # Optional adaptive budget on top of the fixed cap
        # (``train.online_feature_cache_adaptive``): when enabled, the
        # controller senses the job's memory headroom and the payload is
        # trimmed to ``min(cap, headroom + retained - reserve)`` under pressure
        # (the headroom already excludes the cache's own bytes, so retained is
        # added back and only the missing safety margin is evicted), growing
        # back stepwise when headroom returns. Default off; with it off the
        # cache behaves exactly as before (the LRU store keeps its semantics -
        # only the byte budget it runs against becomes dynamic).
        adaptive = (cfg.get("train") or {}).get("online_feature_cache_adaptive", False)
        if not isinstance(adaptive, bool):
            raise ValueError(
                "train.online_feature_cache_adaptive must be a bool "
                f"(true enables the adaptive budget), got {adaptive!r}."
            )
        self._cache_budget_controller = (
            CacheBudgetController(self.cache_capacity_bytes)
            if adaptive and self.cache_capacity_bytes
            else None
        )
        target_cfg = cfg.get("target", {}) or {}
        self.target_dtype = str(target_cfg.get("dtype", "bfloat16"))
        self.attn_implementation = str(target_cfg.get("attn_implementation", "sdpa"))

        self.model, self.processor = load_target(cfg, contract)
        self.target_device = next(self.model.parameters()).device
        # Every CUDA device holding target weights (several with target.device_map).
        self.target_devices = sorted(
            {parameter.device for parameter in self.model.parameters() if parameter.device.type == "cuda"},
            key=lambda device: device.index,
        )
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
        for device in self.target_devices:
            torch.cuda.reset_peak_memory_stats(device)

    def _close_peak_window(self) -> None:
        # Tolerates partially-constructed instances (no target loaded yet).
        if not getattr(self, "_peak_open", False):
            return
        self._peak_open = False
        if not self.target_devices:
            return
        entry = self.stats.setdefault(self.phase, {"target_forwards": 0, "wall_seconds": 0.0})
        peaks = {str(device): torch.cuda.max_memory_allocated(device) / 1e9 for device in self.target_devices}
        entry["target_device_peak_gb"] = max(float(entry.get("target_device_peak_gb", 0.0)), max(peaks.values()))
        if len(peaks) > 1:
            recorded = entry.setdefault("target_device_peaks_gb", {})
            for name, peak in peaks.items():
                recorded[name] = max(float(recorded.get(name, 0.0)), peak)

    def _open_peak_window(self) -> None:
        if self.target_devices:
            self._reset_target_peak()
            self._peak_open = True

    def set_phase(self, phase: str) -> None:
        self._close_peak_window()
        self.phase = phase
        self._open_peak_window()

    def _sync_target(self) -> None:
        """Synchronize every target device so wall-clock attribution is honest.

        Timed regions bracket their own CUDA work with a synchronize on both
        sides: without it ``perf_counter`` deltas only measure kernel launch
        time and the reported preparation/forward split (and total) would
        depend on queue depth instead of actual work.
        """
        for device in self.target_devices:
            torch.cuda.synchronize(device)

    def _stats_entry(self) -> dict:
        return self.stats.setdefault(
            self.phase,
            {"target_forwards": 0, "wall_seconds": 0.0, "prepare_seconds": 0.0, "forward_seconds": 0.0},
        )

    def note_fast_forward(self, items: int = 1) -> None:
        """Count resume fast-forward items, which consume no target work."""
        entry = self._stats_entry()
        entry["fast_forward_items"] = int(entry.get("fast_forward_items", 0)) + max(0, int(items))

    def extract(self, index: int) -> dict:
        """One extraction of one row (render half + forward half, both timed).

        Identical boundary to ``prepare`` + ``extract_prepared``: the same work
        is charged the same way whether the caller fetches per sample or in
        capture groups (R2).
        """
        return self.extract_prepared([self.prepare(index)])[0]

    def prepare(self, index: int) -> dict:
        """Render one row without a target forward; timed into ``prepare_seconds``."""
        self._sync_target()
        started = time.perf_counter()
        prepared = prepare_sample_record(
            self.processor,
            self.rows[index],
            self.media_dir,
            seq_length=self.seq_length,
        )
        self._sync_target()
        elapsed = time.perf_counter() - started
        entry = self._stats_entry()
        entry["prepare_seconds"] = float(entry.get("prepare_seconds", 0.0)) + elapsed
        entry["wall_seconds"] += elapsed
        return prepared

    def extract_prepared(self, prepared_list: list[dict]) -> list[dict]:
        """One batched target forward for already-prepared rows.

        Counters record one target forward call and ``len(prepared_list)``
        samples, so a summary distinguishes forward calls from extracted rows
        once ``train.online_batch_size > 1``. The forward half (target forward
        + output finalisation) is timed into ``forward_seconds``; both timed
        halves sum into ``wall_seconds`` with the same boundary as
        :meth:`extract` (R2).
        """
        self._sync_target()
        started = time.perf_counter()
        records = finalize_prepared_records(
            self.model,
            prepared_list,
            layer_ids=self.layer_ids,
            image_token_id=self.image_token_id,
            merge_size=self.merge_size,
            dtype=self.dtype,
            store_final_hidden=self.store_final_hidden,
            tokenizer=getattr(self.processor, "tokenizer", None),
        )
        self._sync_target()
        elapsed = time.perf_counter() - started
        entry = self._stats_entry()
        entry["target_forwards"] += 1
        entry["target_samples"] = int(entry.get("target_samples", 0)) + len(prepared_list)
        entry["forward_seconds"] = float(entry.get("forward_seconds", 0.0)) + elapsed
        entry["wall_seconds"] += elapsed
        return records

    # -- bounded record cache ------------------------------------------------
    def cache_enabled(self) -> bool:
        """True when ``train.online_feature_cache_gb`` is set above zero."""
        return self.cache_capacity_bytes > 0

    def cache_lookup(self, index: int) -> dict | None:
        """Return the retained record for ``index`` (LRU touch) or None.

        The record is shared, not copied: callers must treat it as read-only.
        With the adaptive budget enabled the store is trimmed to the current
        budget first, so memory-pressure shrinks take effect on the next read.
        """
        if not self.cache_capacity_bytes:
            return None
        budget = self.cache_budget_bytes()
        if budget < self._record_cache_bytes:
            self._trim_record_cache(budget)
        record = self._record_cache.get(int(index))
        if record is None:
            self.cache_stats["misses"] += 1
            return None
        self._record_cache.move_to_end(int(index))
        self.cache_stats["hits"] += 1
        return record

    def cache_store(self, index: int, record: dict) -> None:
        """Retain a normalized record under the byte budget (LRU eviction).

        Records larger than the whole budget are not stored (they would evict
        every other entry for a single hit); placeholders must never be passed
        here. With the adaptive budget enabled the budget is refreshed and the
        existing store is trimmed to it first, so a store that is then refused
        (the record exceeds the budget, e.g. under new pressure) still leaves
        the cache inside the budget instead of holding stale payload until the
        next lookup.
        """
        if not self.cache_capacity_bytes:
            return
        index = int(index)
        if self.cache_policy == "pinned":
            # Frozen plan: only planned records are admitted; every other
            # record keeps re-extracting through the reference path.
            if self._pinned_plan is not None and index not in self._pinned_set:
                self.cache_stats["rejected_records"] += 1
                return
        budget = self.cache_budget_bytes()
        if budget < self._record_cache_bytes:
            self._trim_record_cache(budget)
        size = _record_nbytes(record)
        if self.cache_policy == "pinned":
            self._record_sizes[index] = size
        if size > budget:
            self.cache_stats["skipped_too_large"] += 1
            return
        previous = self._record_cache.pop(index, None)
        if previous is not None:
            self._record_cache_bytes -= _record_nbytes(previous)
        self._record_cache[index] = record
        self._record_cache_bytes += size
        self.cache_stats["stores"] += 1
        if self.cache_policy == "pinned" and self._pinned_plan is not None:
            self.cache_stats["admitted_records"] += 1
        self._trim_record_cache(budget)
        if self._record_cache_bytes > self.cache_stats["peak_bytes"]:
            self.cache_stats["peak_bytes"] = self._record_cache_bytes

    def cache_current_bytes(self) -> int:
        return self._record_cache_bytes

    def cache_budget_bytes(self) -> int:
        """Effective payload budget in bytes.

        Without the adaptive controller this is the configured cap; with it,
        the controller's current budget (``min(cap, job share, headroom +
        retained - reserve)``), refreshed at most once per controller
        interval and sized against the bytes this store currently retains.
        """
        if self._cache_budget_controller is None:
            return self.cache_capacity_bytes
        return self._cache_budget_controller.observe(retained_bytes=self._record_cache_bytes)

    def _trim_record_cache(self, budget: int) -> None:
        while self._record_cache_bytes > budget and self._record_cache:
            _evicted_index, evicted = self._record_cache.popitem(last=False)
            self._record_cache_bytes -= _record_nbytes(evicted)
            self.cache_stats["evictions"] += 1

    def note_epoch_boundary(self) -> dict | None:
        """Freeze the static pinned plan at the first train-phase epoch end.

        Called by the training loop when the loader iterator exhausts (a
        natural epoch end -- the end-of-run break does not fire it). The
        record sizes observed so far (one epoch's extractions, i.e. the whole
        working set under a full pass) are ranked and packed smallest-first
        into ``PINNED_PLAN_FILL_FRACTION`` of the configured cap; the rest of
        the cache is purged once, and from then on only planned records are
        admitted (``rejected_records`` counts the refusals). Nothing is
        planned when the policy is not ``pinned``, when the boundary happens
        in another phase (overfit-sanity / calibration slices), when
        ``train.num_epochs < 2`` (no later epoch could reuse the plan) or
        before any record size is known; the cache then keeps its LRU
        behavior unchanged.

        Returns the plan on the call that froze it (the trainer logs the
        freeze event), ``None`` on every other call.
        """
        if self.cache_policy != "pinned" or self._pinned_plan is not None:
            return None
        if self.phase != "train":
            return None
        if self._num_epochs_planned < 2:
            return None
        self._epoch_boundaries += 1
        if not self.cache_capacity_bytes or not self._record_sizes:
            return None
        budget = int(self.cache_capacity_bytes * PINNED_PLAN_FILL_FRACTION)
        pinned: list[int] = []
        used = 0
        for index, size in sorted(self._record_sizes.items(), key=lambda item: (item[1], item[0])):
            if used + size > budget:
                continue
            pinned.append(index)
            used += size
        self._pinned_set = set(pinned)
        purged_records = 0
        purged_bytes = 0
        for cached_index in [key for key in self._record_cache if key not in self._pinned_set]:
            record = self._record_cache.pop(cached_index)
            purged_bytes += _record_nbytes(record)
            purged_records += 1
        self._record_cache_bytes -= purged_bytes
        self.cache_stats["purged_records"] += purged_records
        self.cache_stats["purged_bytes"] += purged_bytes
        # Records that survive the freeze count as the plan's initial
        # admissions; later admitted stores keep incrementing the same counter.
        self.cache_stats["admitted_records"] += len(self._record_cache)
        self._pinned_plan = {
            "frozen_at_epoch_boundary": self._epoch_boundaries,
            "fill_fraction": PINNED_PLAN_FILL_FRACTION,
            "budget_bytes": budget,
            "n_pinned": len(pinned),
            "planned_bytes": used,
            "sized_records": len(self._record_sizes),
            "purged_records": purged_records,
            "purged_bytes": purged_bytes,
        }
        return self._pinned_plan

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
            "online_batch_size": self.online_batch_size,
            "seq_length": self.seq_length,
            "storage_dtype": str(self.dtype).replace("torch.", ""),
            "stores_final_hidden": self.store_final_hidden,
            "draft_layer_ids": self.layer_ids,
            "capture_layer_ids": self.layer_ids,
            "hidden_size": self.hidden_size,
            "hidden_layout": "concat_capture_layers",
            "target_dtype": self.target_dtype,
            "attn_implementation": self.attn_implementation,
            "target_device": str(self.target_device),
            "target_devices": [str(device) for device in self.target_devices],
            "target_train_device": self.train_device,
            "digest": self.digest(),
        }

    def summary(self) -> dict:
        # Close an open peak window so the final phase's peak is recorded even
        # if ``close()`` never runs; idempotent afterwards.
        self._close_peak_window()
        total_forwards = sum(entry["target_forwards"] for entry in self.stats.values())
        total_samples = sum(int(entry.get("target_samples", 0)) for entry in self.stats.values())
        total_wall = sum(entry["wall_seconds"] for entry in self.stats.values())
        total_prepare = sum(float(entry.get("prepare_seconds", 0.0)) for entry in self.stats.values())
        total_forward = sum(float(entry.get("forward_seconds", 0.0)) for entry in self.stats.values())
        total_fast_forward = sum(int(entry.get("fast_forward_items", 0)) for entry in self.stats.values())
        lookups = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_block = {
            **dict(self.cache_stats),
            "enabled": self.cache_enabled(),
            "policy": self.cache_policy,
            "budget_bytes": self.cache_budget_bytes(),
            "current_bytes": self._record_cache_bytes,
            "hit_rate": (self.cache_stats["hits"] / lookups) if lookups else None,
            # One hit replaces exactly one reference extraction of one row.
            "saved_extractions": self.cache_stats["hits"],
            "note": (
                "opt-in bounded host-RAM retention of extracted records across epochs "
                "(train.online_feature_cache_gb; 0 = disabled). A cache hit replaces the "
                "target forward for that row; hits/misses are per lookup, evictions are "
                "byte-cap evictions, and the cache is runtime-only."
                + (
                    " With train.online_feature_cache_adaptive the byte budget additionally "
                    "tracks the job's memory headroom (see adaptive_budget); capacity_bytes "
                    "stays the configured hard cap."
                    if self._cache_budget_controller is not None
                    else ""
                )
                + (
                    " With train.online_feature_cache_policy=pinned the first train-phase "
                    "epoch end freezes a static smallest-first plan over the record sizes "
                    "observed in that epoch (see pinned/plan): planned records are admitted "
                    "thereafter and counted in admitted_records, non-planned stores are "
                    "refused (rejected_records), records outside the plan are purged once at "
                    "freeze (purged_records/purged_bytes), and pinned_records/pinned_bytes "
                    "track the planned records resident now. Misses still re-extract through "
                    "the reference path; saved_extractions equals hits."
                    if self.cache_policy == "pinned"
                    else ""
                )
            ),
        }
        if self._cache_budget_controller is not None:
            cache_block["adaptive_budget"] = self._cache_budget_controller.stats()
        if self.cache_policy == "pinned":
            cache_block["pinned"] = {
                "frozen": self._pinned_plan is not None,
                "pinned_records": len(self._record_cache) if self._pinned_plan is not None else 0,
                "pinned_bytes": self._record_cache_bytes if self._pinned_plan is not None else 0,
                "known_sizes": len(self._record_sizes),
                "plan": self._pinned_plan,
            }
        return {
            **self.manifest(),
            "target_resident_params": self.target_resident_params,
            "target_resident_bytes": self.target_resident_bytes,
            "extractions": {phase: dict(entry) for phase, entry in sorted(self.stats.items())},
            "target_forwards_total": total_forwards,
            "target_samples_total": total_samples,
            "extraction_wall_seconds_total": total_wall,
            "extraction_prepare_seconds_total": total_prepare,
            "extraction_forward_seconds_total": total_forward,
            "fast_forward_items_total": total_fast_forward,
            "feature_cache": cache_block,
            "target_forwards_note": (
                "target_forwards counts model forward calls; target_samples counts extracted rows. "
                "They differ when train.online_batch_size > 1 (one forward per capture group)."
            ),
            "extraction_wall_seconds_note": (
                "wall_seconds is the full extraction boundary for one row: render/decode + processor "
                "(prepare_seconds) plus the teacher-forced target forward and record finalisation "
                "(forward_seconds), with the target device synchronized around each timed stage. "
                "Per-sample reads (extract) and batched reads (fetch_batch) charge the same boundary, "
                "so their extraction_wall_seconds_total values are comparable. Counters are per phase "
                "(train / overfit sanity / calibration); the startup visual-path check runs a fixed 3 "
                "capture forwards outside these counters, and a resumed run answers fast-forwarded "
                "items with cheap placeholders that consume no target work (counted as "
                "fast_forward_items, never as wall_seconds)"
            ),
            "target_device_peak_note": (
                "target_device_peak_gb is the CUDA peak on the target's device during that phase "
                "(reset per phase; the largest per-device peak when target.device_map shards the target "
                "over several devices). When target.device == train.device the phase peak includes the "
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
        self._record_cache.clear()
        self._record_cache_bytes = 0


class OnlineBatchIterator:
    """Epoch iterator that yields extracted + collated batches for the trainer.

    Replaces the ``DataLoader`` on the online path when
    ``train.online_batch_size > 1``: it walks the same
    ``RandomSampler`` + ``BatchSampler`` sequence the per-sample ``DataLoader``
    would (same generator, ``drop_last=False``), so batch composition is
    identical to the per-sample path for a given seed, but every batch is
    extracted with one batched target forward via ``dataset.fetch_batch``.
    ``test_online_batch_iterator_reproduces_dataloader_order`` pins the
    equivalence against a real ``DataLoader``.

    ``capture_batch`` (``train.online_batch_size``) caps how many rows share
    one target forward: a training batch larger than it is split into
    consecutive capture groups (each fetched with its own padded forward and
    concatenated back in order), so the effective group size is
    ``min(online_batch_size, micro_batch_size)``.
    """

    def __init__(self, dataset: "OnlineFeatureDataset", collate, *, batch_size: int, shuffle: bool,
                 generator, capture_batch: int | None = None, fetch_batch=None):
        self.dataset = dataset
        self.collate = collate
        self.batch_size = max(1, int(batch_size))
        self.capture_batch = max(1, int(capture_batch) if capture_batch else self.batch_size)
        self.shuffle = bool(shuffle)
        self.generator = generator
        self.num_samples = len(dataset)
        # Sampler indices address ``dataset``; ``fetch_batch`` defaults to the
        # dataset's own batched extraction and is overridden by the trainer
        # when a Subset wrapper needs its indices mapped back to base indices.
        self.fetch_batch = fetch_batch if fetch_batch is not None else dataset.fetch_batch

    def __len__(self) -> int:
        return max(1, math.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

        if self.shuffle:
            # The DataLoader's iterator draws one int64 base seed from the
            # generator before the sampler draws its permutations (a torch
            # internal, pinned by the parity test); consume the same draw so
            # the epoch order matches the per-sample path exactly.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            sampler = RandomSampler(self.dataset, generator=self.generator)
        else:
            sampler = SequentialSampler(self.dataset)
        for indices in BatchSampler(sampler, self.batch_size, drop_last=False):
            indices = list(indices)
            records: list[dict] = []
            for start in range(0, len(indices), self.capture_batch):
                records.extend(self.fetch_batch(indices[start : start + self.capture_batch]))
            yield self.collate(records)
