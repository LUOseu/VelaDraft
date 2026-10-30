"""In-training objective validation on frozen held-out splits (online feature mode).

Configured under ``validation``::

    validation:
      enabled: true
      rows: <target_data.jsonl with regen answers>   # sharegpt rows with a ``regen`` block
      meta: <meta.json next to the media dir>
      splits: {dev: <ids jsonl>, test: <ids jsonl>}   # rows whose ``id`` is listed
      every_steps: 2500
      extra_steps: [18750]
      at_start: true
      at_epoch_end: true
      batch_size: 2
      seed: 20261030

``dev`` is evaluated at update 0, every ``every_steps`` updates, every
``extra_steps`` entry, every epoch end and at the end of the run; ``test``
only at the end. Each point is evaluated once per (split, step), also across
an interrupted-and-resumed run (existing records are read back).

The objective is the module's own training objective (same forward, same
loss), computed under ``no_grad`` in training mode with the training RNG
forked away: every batch draws its anchors from a fixed per-batch seed, so
values are comparable across steps and arms, and training's random stream is
untouched. For a staged recipe the stage of the last completed update is
evaluated, and at a stage boundary both adjacent stages.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..data.sharegpt import load_sharegpt
from ..utils import append_jsonl, log, read_jsonl
from .features import collate_features
from .instrument import participating_devices
from .trainer_core import NoValidAnchorsError


class ObjectiveValidator:
    def __init__(self, cfg: dict, contract: dict, module, source, device, out_path: Path, *, total_steps: int):
        vcfg = cfg.get("validation") or {}
        self.module = module
        self.source = source
        self.device = torch.device(device)
        self.out_path = Path(out_path)
        self.total_steps = int(total_steps)
        self.every = int(vcfg.get("every_steps") or 0)
        self.extra = {int(step) for step in vcfg.get("extra_steps") or []}
        self.at_start = bool(vcfg.get("at_start", True))
        self.at_epoch_end = bool(vcfg.get("at_epoch_end", True))
        self.batch_size = int(vcfg.get("batch_size", 2))
        self.seed = int(vcfg.get("seed", 20261030))
        self.image_token_id = int(contract["tokens"]["image_token_id"])
        self.spatial = bool(
            getattr(module, "uses_multimodal_position_ids", getattr(module.draft_model, "spatial_rope_enabled", False))
        )
        rows, self.media_dir = load_sharegpt(vcfg["rows"], meta_path=vcfg.get("meta"))
        by_id = {str(row["id"]): row for row in rows}
        self.rows: dict[str, list[dict]] = {}
        self.split_sizes: dict[str, dict] = {}
        for split, ids_path in (vcfg.get("splits") or {}).items():
            ids = [str(record["id"]) for record in read_jsonl(ids_path)]
            kept = [by_id[row_id] for row_id in ids if row_id in by_id]
            self.rows[split] = kept
            self.split_sizes[split] = {"split_rows": len(ids), "rows_with_target_answer": len(kept)}
        if "dev" not in self.rows:
            raise ValueError("validation.splits must name a 'dev' split.")
        self.done: set[tuple[str, int]] = set()
        if self.out_path.exists():
            self.done.update((record["split"], int(record["step"])) for record in read_jsonl(self.out_path))
        log(f"validation: {self.split_sizes}; every {self.every} updates, extra {sorted(self.extra)}")

    def _stages(self, step: int) -> list[int | None]:
        recipe = getattr(self.module, "train_recipe", None)
        if recipe is None or recipe.name == "controlled_ce":
            return [None]
        last = recipe.stage_index(max(step - 1, 0), self.total_steps)
        upcoming = recipe.stage_index(step, self.total_steps)
        return sorted({int(last), int(upcoming)})

    def __call__(self, step: int, *, epoch_end: bool, final: bool) -> None:
        reasons = []
        if step == 0 and self.at_start:
            reasons.append("start")
        if self.every and step > 0 and step % self.every == 0:
            reasons.append("every")
        if step in self.extra:
            reasons.append("extra")
        if epoch_end and self.at_epoch_end:
            reasons.append("epoch_end")
        if final:
            reasons.append("final")
        if not reasons:
            return
        for split in ["dev"] + (["test"] if final and "test" in self.rows else []):
            if (split, step) in self.done:
                continue
            self.done.add((split, step))
            for record in self.evaluate(split, step):
                record["reasons"] = reasons
                append_jsonl(self.out_path, record)

    def evaluate(self, split: str, step: int) -> list[dict]:
        module = self.module
        rows = self.rows[split]
        stages = self._stages(step)
        saved_stage = getattr(module, "_recipe_stage_index", None)
        acc = {stage: _Accumulator() for stage in stages}
        started = time.perf_counter()
        cuda_devices = [d for d in participating_devices(self.device, self.source) if d.type == "cuda"]
        self.source.set_phase("validation")
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
                for batch_index, start in enumerate(range(0, len(rows), self.batch_size)):
                    records = self.source.extract_rows(rows[start : start + self.batch_size], self.media_dir)
                    batch = collate_features(records, self.image_token_id)
                    batch = {
                        key: value.to(self.device) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                        if key != "lengths"
                    }
                    for stage in stages:
                        if stage is not None:
                            module._recipe_stage_index = stage
                        torch.manual_seed(self.seed + batch_index)
                        try:
                            metrics = module(
                                input_ids=batch["input_ids"],
                                hidden_states=batch["hidden"],
                                loss_mask=batch["loss_mask"],
                                image_mask=batch["image_mask"],
                                multimodal_position_ids=batch["position3"] if self.spatial else None,
                                teacher_hidden=batch.get("final_hidden"),
                            )
                        except NoValidAnchorsError:
                            acc[stage].skipped += 1
                            continue
                        acc[stage].add(metrics, getattr(module, "last_workload", {}), len(records))
        finally:
            if stages != [None]:
                module._recipe_stage_index = saved_stage
            self.source.set_phase("train")
        wall = time.perf_counter() - started
        out = []
        for stage in stages:
            record = acc[stage].result()
            record.update(
                {
                    "split": split,
                    "step": int(step),
                    "recipe_stage": stage,
                    "rows": len(rows),
                    **self.split_sizes[split],
                    "batch_size": self.batch_size,
                    "anchor_seed": self.seed,
                    "wall_s": wall,
                }
            )
            out.append(record)
            log(
                f"validation {split} step {step} stage {stage}: objective={record['objective_mean']:.5f} "
                f"acc={record['accuracy']:.4f} al={record['accept_len']:.3f} ({wall:.0f}s)"
            )
        return out


class _Accumulator:
    def __init__(self):
        self.batches = 0
        self.skipped = 0
        self.samples = 0
        self.loss_sum = 0.0
        self.weighted_loss = 0.0
        self.weight = 0.0
        self.valid = 0
        self.correct = 0
        self.accept_sum = 0.0
        self.blocks = 0
        self.extra: dict[str, float] = {}
        self.workload: dict[str, int] = {}

    def add(self, metrics, workload: dict, samples: int) -> None:
        loss = float(metrics.loss.detach())
        weight = float(metrics.loss_weight)
        self.batches += 1
        self.samples += samples
        self.loss_sum += loss
        self.weighted_loss += loss * weight
        self.weight += weight
        self.valid += int(metrics.valid_tokens)
        self.correct += int(metrics.correct_tokens)
        self.accept_sum += float(metrics.accept_len_sum)
        self.blocks += int(metrics.valid_blocks)
        for key, value in (metrics.extra_metrics or {}).items():
            self.extra[key] = self.extra.get(key, 0.0) + float(value)
        for key, value in (workload or {}).items():
            self.workload[key] = self.workload.get(key, 0) + int(value)

    def result(self) -> dict:
        batches = max(self.batches, 1)
        return {
            "objective_mean": self.loss_sum / batches,
            "objective_weighted": self.weighted_loss / self.weight if self.weight else None,
            "components_mean": {key: value / batches for key, value in self.extra.items()},
            "accuracy": self.correct / self.valid if self.valid else 0.0,
            "accept_len": self.accept_sum / self.blocks if self.blocks else 0.0,
            "supervised_positions": self.valid,
            "batches": self.batches,
            "skipped_batches": self.skipped,
            "samples": self.samples,
            "workload": self.workload,
        }
