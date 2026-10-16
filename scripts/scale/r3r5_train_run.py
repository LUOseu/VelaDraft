#!/usr/bin/env python
"""Drive one R3/R5 training run: link the shared corpus, launch the trainer on
the training GPU under a guard, evaluate every milestone snapshot on the eval
GPU with the fixed 64-prompt dev subset, prune losing snapshots, and keep an
ETA/disk/GPU progress ledger. One manager process; writes everything under
``<run-dir>/driver/`` and ``<run-dir>/eval/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[2]
PY = "/home/lyh/miniforge3/envs/c1vllm/bin/python"
TARGET = "/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct"


def now() -> str:
    return time.strftime("%F %H:%M:%S")


def _tag_suffix(ckpt: str) -> str:
    if ckpt == "ckpt_final.pt":
        return "_final"
    digits = ckpt.replace("ckpt_step", "").replace(".pt", "").lstrip("0")
    return f"_step{digits}"


class Driver:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.shared_dir = Path(args.shared_dir)
        self.log_dir = self.run_dir / "driver"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log = self.run_dir / "train" / "train_log.jsonl"
        self.evaluated: dict[str, float] = {}
        self.best_ckpt: str | None = None
        self.best_al: float | None = None
        self.eval_proc: subprocess.Popen | None = None
        self.eval_meta: dict | None = None
        self.extra_results: dict[str, dict] = {}
        self.pending_extra: dict[str, str] = {}
        self.proc: subprocess.Popen | None = None
        self.started = time.time()
        # Plan length depends on the corpus, which is only linked into the run
        # dir in run() (link_shared); deferring the plan keeps __init__ from
        # reading a file that does not exist yet.
        self.total_steps = 0
        self.boundary_step: int | None = None
        self.last_step = 0
        self.last_step_time = None
        self.rate_s_per_step: float | None = None
        self.samples: list[tuple[float, int]] = []

    # ---------- planning ----------
    def _config(self) -> dict:
        import yaml

        return yaml.safe_load(Path(self.args.config).read_text())

    def _plan_steps(self) -> int:
        from vlm_dflash.dflash.train import compute_total_steps

        cfg = self._config()
        with (self.run_dir / "regen" / "target_data.jsonl").open() as fh:
            rows = sum(1 for line in fh if line.strip())
        return compute_total_steps(rows, cfg.get("train", {}))

    def _boundary_step(self) -> int | None:
        from vlm_dflash.objectives import resolve_recipe

        cfg = self._config()
        recipe = resolve_recipe(cfg, "dfly")
        if recipe is None:
            return None
        boundary = recipe.split_step(self.total_steps)
        return boundary if 0 < boundary < self.total_steps else None

    # ---------- setup ----------
    def link_shared(self) -> None:
        for name in ("data", "regen"):
            src = self.shared_dir / name
            dst = self.run_dir / name
            if dst.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for path in sorted(src.rglob("*")):
                if path.is_dir():
                    (dst / path.relative_to(src)).mkdir(parents=True, exist_ok=True)
                    continue
                target = dst / path.relative_to(src)
                if target.exists():
                    continue
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copyfile(path, target)
        frozen = json.loads((self.run_dir / "regen" / "freeze_record.json").read_text())
        rows = sum(1 for _ in (self.run_dir / "regen" / "target_data.jsonl").open())
        if rows != frozen["num_frozen"]:
            raise SystemExit(f"frozen corpus mismatch: file rows={rows} record={frozen['num_frozen']}")
        print(f"[{now()}] linked shared corpus: {rows} frozen rows", flush=True)

    def gpu_free(self, index: int, min_free_mib: int) -> tuple[int, int]:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        util, free = (int(part.strip()) for part in out.split(",")[:2])
        return util, free

    def guard(self, index: int, min_free_mib: int, attempts: int = 20, sleep: int = 30) -> None:
        for attempt in range(attempts):
            util, free = self.gpu_free(index, min_free_mib)
            if util <= 5 and free >= min_free_mib:
                print(f"[{now()}] guard OK gpu={index} util={util}% free={free}MiB", flush=True)
                return
            print(f"[{now()}] guard wait gpu={index} util={util}% free={free}MiB", flush=True)
            time.sleep(sleep)
        raise SystemExit(f"guard failed for gpu {index}")

    # ---------- training ----------
    def launch_training(self, resume: bool = False, extra_sets: list[str] | None = None) -> None:
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(self.args.train_gpu),
            PYTHONPATH=str(REPO),
            PYTHONUNBUFFERED="1",
            VLM_DFLASH_TARGET=TARGET,
            # The 3B target + live extraction peaks at ~39 GiB on a 48 GiB card;
            # expandable segments keeps the reserved-but-unallocated pool small.
            PYTORCH_CUDA_ALLOC_CONF=os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
        )
        cmd = [PY, str(REPO / "scripts" / "train_dflash.py"), "--config", self.args.config,
               "--run-dir", str(self.run_dir)]
        if resume:
            cmd.append("--resume")
        for item in extra_sets or []:
            cmd += ["--set", item]
        stdout = open(self.log_dir / "train_stdout.log", "a")
        self.proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT)
        print(f"[{now()}] launched trainer pid={self.proc.pid} gpu={self.args.train_gpu} ", flush=True)

    # ---------- monitoring helpers ----------
    def step(self) -> int:
        if not self.train_log.exists():
            return 0
        last = 0
        with self.train_log.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    last = int(json.loads(line)["step"])
                except Exception:
                    continue
        return last

    def progress(self, event: str, **fields) -> None:
        row = {"time": now(), "event": event, **fields}
        with (self.log_dir / "progress.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    # ---------- evals ----------
    def snapshots(self) -> list[str]:
        train_dir = self.run_dir / "train"
        return sorted(p.name for p in train_dir.glob("ckpt_step*.pt"))

    def eval_gpu_busy(self) -> bool:
        if self.args.eval_gpu is None:
            return False
        _, free = self.gpu_free(self.args.eval_gpu, 0)
        return free < self.args.eval_min_free_mib

    def start_eval(self, ckpt: str, tag: str | None = None, prompts: int | None = None,
                   split: str | None = None) -> None:
        if tag is None:
            if ckpt == "ckpt_final.pt":
                tag = "pfinal"
            else:
                tag = "p" + ckpt.replace("ckpt_step", "").replace(".pt", "").lstrip("0")
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(self.args.eval_gpu),
            PYTHONPATH=str(REPO),
            PYTHONUNBUFFERED="1",
            VLM_DFLASH_TARGET=TARGET,
        )
        cmd = [
            PY, str(REPO / "scripts" / "eval_dflash.py"),
            "--config", self.args.config,
            "--run-dir", str(self.run_dir),
            "--num-prompts", str(prompts or self.args.eval_prompts),
            "--verification", "sequential",
            "--tag", tag,
            "--set", f"eval.checkpoints=[{ckpt}]",
        ]
        if split:
            cmd += ["--prompt-source-split", split]
        log = open(self.log_dir / f"eval_{tag}.log", "a")
        self.eval_proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.eval_meta = {"checkpoint": ckpt, "tag": tag, "started": time.time(),
                          "prompts": prompts or self.args.eval_prompts, "split": split or "val.jsonl"}
        print(f"[{now()}] eval started: {ckpt} -> tag {tag}", flush=True)

    def finish_eval(self) -> None:
        proc, meta = self.eval_proc, dict(self.eval_meta or {})
        self.eval_proc, self.eval_meta = None, None
        rc = proc.wait()
        result_path = self.run_dir / "eval" / f"eval_result_{meta['tag']}.json"
        row = {"time": now(), "checkpoint": meta["checkpoint"], "tag": meta["tag"], "rc": rc,
               "wall_seconds": round(time.time() - meta["started"], 1)}
        periodic = meta.get("prompts") == self.args.eval_prompts and meta.get("split") == "val.jsonl"
        if rc == 0 and result_path.exists():
            result = json.loads(result_path.read_text())
            run = result["runs"][meta["checkpoint"]]
            row.update(
                mean_al=run["mean_al"],
                acceptance_rate=run["acceptance_rate"],
                per_slot_acceptance=run["per_slot_acceptance"],
                target_forward_calls=run.get("target_forward_calls"),
                tokens_per_target_forward=run.get("tokens_per_target_forward"),
                token_exact=result["equivalence_check"]["runs"][meta["checkpoint"]]["token_exact"],
                split=meta.get("split"),
                num_prompts=meta.get("prompts"),
            )
            if periodic:
                self.evaluated[meta["checkpoint"]] = float(run["mean_al"])
                if self.best_al is None or float(run["mean_al"]) > self.best_al:
                    self.best_al, self.best_ckpt = float(run["mean_al"]), meta["checkpoint"]
            else:
                self.extra_results[meta["tag"]] = row
        else:
            row["error"] = "eval failed"
        with (self.log_dir / "periodic_summary.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{now()}] eval done: {meta['checkpoint']} rc={rc} row={json.dumps(row)[:200]}", flush=True)
        self.progress("eval_done", **row)
        self.prune()

    def prune(self) -> None:
        boundary_ckpt = None
        if self.boundary_step is not None:
            boundary_ckpt = f"ckpt_step{self.boundary_step:07d}.pt"
        snapshots = self.snapshots()
        keep = set()
        if snapshots:
            keep.add(snapshots[-1])
        if self.best_ckpt:
            keep.add(self.best_ckpt)
        if boundary_ckpt and boundary_ckpt in snapshots:
            keep.add(boundary_ckpt)
        # candidates still queued for dev confirmation / test evaluation
        for name in list(self.pending_extra.values()):
            if name in snapshots:
                keep.add(name)
        removed = []
        for name in snapshots:
            if name not in keep:
                path = self.run_dir / "train" / name
                try:
                    path.unlink()
                    removed.append(name)
                except OSError:
                    pass
        if removed:
            print(f"[{now()}] pruned snapshots: {removed} (kept {sorted(keep)})", flush=True)
            self.progress("pruned", removed=removed, kept=sorted(keep))

    def pending_eval(self) -> str | None:
        for name in self.snapshots():
            if name in self.evaluated:
                continue
            return name
        return None

    # ---------- post-training: dev confirmation + report-only test ----------
    def wait_for_eval_slot(self, timeout_seconds: float = 5400.0) -> bool:
        deadline = time.time() + timeout_seconds
        while self.eval_gpu_busy():
            if time.time() > deadline:
                return False
            time.sleep(30)
        return True

    def run_eval_blocking(self, ckpt: str, tag: str, prompts: int, split: str | None = None) -> dict | None:
        if not self.wait_for_eval_slot():
            self.progress("eval_skipped", checkpoint=ckpt, tag=tag, reason="eval_gpu_busy_timeout")
            print(f"[{now()}] eval skipped (gpu busy): {ckpt} tag={tag}", flush=True)
            return None
        self.start_eval(ckpt, tag=tag, prompts=prompts, split=split)
        self.finish_eval()
        return self.extra_results.get(tag)

    def confirm_and_test(self) -> None:
        """Pick the run's best checkpoint on the full dev split, then test it.

        Selection uses the full 256-row dev split (never the test split): the
        periodic 64-row evals only order the candidates. The test split is
        evaluated afterwards for reporting only.
        """
        import hashlib

        ranked = sorted(self.evaluated.items(), key=lambda item: -item[1])
        candidates: list[str] = []
        skipped_pruned: list[str] = []
        for ckpt, _ in ranked:
            if len(candidates) >= self.args.confirm_top:
                break
            if (self.run_dir / "train" / ckpt).exists():
                candidates.append(ckpt)
            else:
                skipped_pruned.append(ckpt)
        confirm: dict[str, dict] = {}
        for ckpt in candidates:
            tag = "dev" + _tag_suffix(ckpt)
            row = self.run_eval_blocking(ckpt, tag, self.args.confirm_prompts)
            if row is not None:
                confirm[ckpt] = row
        best = None
        if confirm:
            best = max(confirm, key=lambda ckpt: confirm[ckpt]["mean_al"])
        elif self.best_ckpt:
            best = self.best_ckpt
        selection = {
            "time": now(),
            "selection_split": "val.jsonl",
            "selection_num_prompts": self.args.confirm_prompts,
            "periodic_64_ranking": {ckpt: al for ckpt, al in ranked},
            "skipped_missing_candidates": skipped_pruned,
            "confirmed_full_dev": {ckpt: row["mean_al"] for ckpt, row in confirm.items()},
            "best_ckpt": best,
            "best_full_dev_al": None if best not in confirm else confirm[best]["mean_al"],
            "note": "test split is evaluated for reporting only and never selects the checkpoint",
        }
        if self.args.test_prompts:
            for ckpt in dict.fromkeys([best, "ckpt_final.pt"]):
                if not ckpt or not (self.run_dir / "train" / ckpt).exists():
                    continue
                row = self.run_eval_blocking(ckpt, "test" + _tag_suffix(ckpt), self.args.test_prompts,
                                             split=self.args.test_split)
                if row is not None:
                    selection.setdefault("test_report", {})[ckpt] = {
                        "split": self.args.test_split,
                        "num_prompts": self.args.test_prompts,
                        "mean_al": row["mean_al"],
                        "acceptance_rate": row.get("acceptance_rate"),
                        "per_slot_acceptance": row.get("per_slot_acceptance"),
                        "target_forward_calls": row.get("target_forward_calls"),
                        "tokens_per_target_forward": row.get("tokens_per_target_forward"),
                    }
        (self.log_dir / "selection.json").write_text(json.dumps(selection, indent=2))
        self.progress("selection", **{k: v for k, v in selection.items() if k != "periodic_64_ranking"})
        print(f"[{now()}] SELECTION best={best} full_dev={selection['best_full_dev_al']}", flush=True)

    # ---------- loop ----------
    def oom_detected(self) -> bool:
        log_path = self.log_dir / "train_stdout.log"
        if not log_path.exists():
            return False
        tail = log_path.read_text(errors="ignore")[-20000:]
        return "OutOfMemoryError" in tail or "out of memory" in tail.lower()

    def monitor_until_exit(self) -> None:
        archived_config = False
        last_eta_log = 0.0
        while True:
            rc = self.proc.poll()
            step = self.step()
            wall = time.time() - self.started
            if step and step != self.last_step:
                self.samples.append((time.time(), step))
                self.last_step = step
                recent = [s for s in self.samples if s[0] > time.time() - 600]
                if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
                    self.rate_s_per_step = (recent[-1][0] - recent[0][0]) / (recent[-1][1] - recent[0][1])
            if not archived_config and (self.run_dir / "config.json").exists():
                shutil.copyfile(self.run_dir / "config.json", self.run_dir / "config.train.json")
                archived_config = True
            # snapshot eval scheduling
            if self.eval_proc is not None:
                if self.eval_proc.poll() is not None:
                    self.finish_eval()
            else:
                pending = self.pending_eval()
                if pending and not self.eval_gpu_busy():
                    self.start_eval(pending)
                elif pending:
                    self.progress("eval_deferred", checkpoint=pending, reason="eval_gpu_busy")
            # ETA / disk logging
            if time.time() - last_eta_log > 120:
                last_eta_log = time.time()
                dh = shutil.disk_usage("/home").free / 1e9
                try:
                    gpu_util, gpu_free = self.gpu_free(self.args.train_gpu, 0)
                except Exception:
                    gpu_util, gpu_free = -1, -1
                eta = None
                if self.rate_s_per_step and step:
                    eta = (self.total_steps - step) * self.rate_s_per_step / 3600.0
                payload = {
                    "step": step, "total_steps": self.total_steps,
                    "elapsed_min": round(wall / 60, 1),
                    "s_per_step": None if self.rate_s_per_step is None else round(self.rate_s_per_step, 3),
                    "eta_hours": None if eta is None else round(eta, 2),
                    "home_free_gb": round(dh, 2),
                    "train_gpu_util": gpu_util, "train_gpu_free_mib": gpu_free,
                    "evaluated": self.evaluated, "best_al": self.best_al, "best_ckpt": self.best_ckpt,
                }
                print(f"[{now()}] progress {json.dumps(payload)}", flush=True)
                self.progress("tick", **payload)
                if dh < 4.0:
                    # ckpt_init.pt (the fresh-start snapshot) is only needed to
                    # start over; the run is underway, so trade it for headroom
                    # and keep its hash in the ledger.
                    init_ckpt = self.run_dir / "train" / "ckpt_init.pt"
                    if init_ckpt.exists():
                        from vlm_dflash.utils import sha256_file

                        digest = sha256_file(init_ckpt)
                        try:
                            init_ckpt.unlink()
                            self.progress("pruned_ckpt_init", sha256=digest, home_free_gb=round(dh, 2))
                            print(f"[{now()}] pruned ckpt_init.pt (sha256={digest[:12]})", flush=True)
                        except OSError:
                            pass
                if dh < 3.0:
                    self.progress("DISK_ALARM", home_free_gb=round(dh, 2))
                    print(f"[{now()}] DISK ALARM home_free_gb={dh:.2f}", flush=True)
            if rc is not None:
                break
            time.sleep(20)
        return

    def finish_after_training(self) -> None:
        # training exited; drain every pending milestone eval before the final one
        while True:
            pending = self.pending_eval()
            if self.eval_proc is not None:
                self.finish_eval()
                continue
            if not pending:
                break
            if self.eval_gpu_busy():
                time.sleep(30)
                continue
            self.start_eval(pending)
        # training finished cleanly: drain evals, confirm best, report-only test
        self.progress("train_finished", rc=0, step=self.last_step)
        # final evaluation of ckpt_final on the fixed dev subset
        final = "ckpt_final.pt"
        if self.eval_proc is None and final not in self.evaluated:
            deadline = time.time() + 1800
            while self.eval_gpu_busy() and time.time() < deadline:
                time.sleep(30)
            self.start_eval(final)
            self.finish_eval()
        self.confirm_and_test()
        self.progress(
            "run_complete", evaluated=self.evaluated, best_ckpt=self.best_ckpt, best_al=self.best_al,
            wall_min=round((time.time() - self.started) / 60, 1),
        )
        print(f"[{now()}] RUN COMPLETE evaluated={json.dumps(self.evaluated)}", flush=True)



    def training_attempts(self) -> list[tuple[list[str], bool, str]]:
        """Recovery ladder: (extra --set overrides, resume, label).

        An OOM on a card that later loses a neighbour is transient, so the first
        retry resumes the *same* geometry from the periodic train-state (the
        checkpoint's plan digest pins the geometry: a changed micro-batch is a
        different run by design and cannot resume). Only if the spec geometry
        keeps failing do the reduced micro-batch variants start fresh -- they
        keep the effective batch at 8 (micro 4 x accum 2), so the optimizer sees
        the same batch, the same step count and the same loss normalization.
        """
        spec: list[str] = [item.strip() for item in (self.args.initial_sets or "").split(",") if item.strip()]
        if not self.args.oom_fallback:
            return [(spec, False, "spec")]
        small = ["train.micro_batch_size=4", "train.grad_accumulation_steps=2"]
        smaller = small + ["train.online_batch_size=4"]
        return [
            (spec, False, "spec"),
            (spec, True, "spec-resume"),
            (small, False, "mbs4-accum2"),
            (smaller, False, "mbs4-accum2-online4"),
        ]

    def committed_state_step(self) -> int:
        """Step of the newest *committed* resumable state, or 0 if there is none.

        The trainer only writes a resumable state at the save cadence (0.5 epoch)
        and at the stage boundary, so the mere existence of ``ckpt_last.pt`` means
        real optimiser progress was committed and can be resumed; the step that
        the (reconciled) train log reports is used for the log message.
        """
        try:
            from vlm_dflash.config import load_config

            cfg = load_config(self.args.config)
            state_dir = (cfg.get("train") or {}).get("resume_state_dir")
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"[{now()}] committed_state_step: config unavailable ({exc})", flush=True)
            return 0
        if not state_dir or not (Path(str(state_dir)) / "ckpt_last.pt").exists():
            return 0
        return self.step()

    def load_previous_evals(self) -> None:
        """Re-adopt milestone evals already on disk.

        A driver relaunched after the whole process tree was killed (the host
        OOM killer SIGKILLs the trainer, which the driver reads as a non-OOM
        crash, so it stops by design) must not pay for those evals again, and
        more importantly must still know which snapshot was the best one --
        ``prune()`` deletes every snapshot that is not the best, the boundary or
        the newest, so forgetting the best would delete it on the next prune.
        """
        path = self.log_dir / "periodic_summary.jsonl"
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("rc") != 0 or row.get("mean_al") is None:
                continue
            if row.get("num_prompts") != self.args.eval_prompts or row.get("split") != "val.jsonl":
                continue
            ckpt = row["checkpoint"]
            self.evaluated[ckpt] = float(row["mean_al"])
            if self.best_al is None or float(row["mean_al"]) > self.best_al:
                self.best_al, self.best_ckpt = float(row["mean_al"]), ckpt
        if self.evaluated:
            print(f"[{now()}] re-adopted {len(self.evaluated)} dev points; best={self.best_ckpt} "
                  f"al={self.best_al}", flush=True)

    def run(self) -> None:
        self.link_shared()
        self.total_steps = self._plan_steps()
        self.boundary_step = self._boundary_step()
        self.guard(self.args.train_gpu, self.args.train_min_free_mib)
        self.load_previous_evals()
        attempts = self.training_attempts()
        first = max(1, int(self.args.start_attempt))
        if first > 1:
            print(f"[{now()}] starting at attempt {first} (previously interrupted)", flush=True)
            self.progress("start_attempt", first_attempt=first, total_attempts=len(attempts))
        for index, (extra_sets, resume, label) in enumerate(attempts, start=1):
            if index < first:
                continue
            if not resume and self.committed_state_step() > 0:
                # The remaining ladder entries start from scratch. That was safe
                # when they were written (a crash before any save cadence leaves
                # nothing to lose), but the host OOM killer SIGKILLs the trainer
                # without writing OOM text to the log, so the driver reads it as
                # a plain crash. Starting fresh here would silently throw away
                # hours of committed optimisation, so refuse and stop instead.
                self.progress("fresh_restart_refused", attempt=index, label=label,
                              committed_step=self.committed_state_step())
                print(f"[{now()}] attempt {index} ({label}) would restart from scratch while "
                      f"committed state at step {self.committed_state_step()} exists; refusing",
                      flush=True)
                return
            self.launch_training(resume=resume, extra_sets=extra_sets)
            self.progress("train_launched", total_steps=self.total_steps, boundary_step=self.boundary_step,
                          train_gpu=self.args.train_gpu, eval_gpu=self.args.eval_gpu,
                          attempt=index, label=label, extra_sets=extra_sets, resume=resume)
            print(f"[{now()}] attempt {index}/{len(attempts)} label={label} resume={resume} "
                  f"sets={extra_sets}", flush=True)
            self.monitor_until_exit()
            rc = self.proc.returncode
            if rc == 0:
                break
            oom = self.oom_detected()
            self.progress("attempt_failed", rc=rc, step=self.last_step, attempt=index, label=label, oom=oom)
            print(f"[{now()}] attempt {index} FAILED rc={rc} oom={oom} step={self.last_step}", flush=True)
            if not oom or index == len(attempts):
                print(f"[{now()}] TRAIN FAILED rc={rc}; giving up", flush=True)
                return
            self.proc = None
            self.last_step = 0
            self.samples = []
            time.sleep(90)
        self.finish_after_training()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--train-gpu", type=int, required=True)
    parser.add_argument("--eval-gpu", type=int, default=None)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--confirm-top", type=int, default=2,
                        help="how many top periodic candidates get a full-dev confirmation eval")
    parser.add_argument("--confirm-prompts", type=int, default=256)
    parser.add_argument("--test-prompts", type=int, default=256, help="0 disables the report-only test eval")
    parser.add_argument("--test-split", default="test.jsonl")
    parser.add_argument("--initial-sets", default="",
                        help="comma-separated --set overrides applied from the first attempt "
                             "(known-in-advance geometry fallbacks, e.g. after a smoke OOM)")
    parser.add_argument("--oom-fallback", action="store_true",
                        help="on a CUDA OOM in the first attempt, relaunch once with micro_batch_size=4 "
                             "grad_accumulation_steps=2 (same effective batch and loss normalization)")
    parser.add_argument("--start-attempt", type=int, default=1,
                        help="1-based index into the recovery ladder to start at; use 2 to resume a "
                             "run whose driver was killed mid-training (attempt 2 = spec-resume)")
    parser.add_argument("--train-min-free-mib", type=int, default=40000)
    parser.add_argument("--eval-min-free-mib", type=int, default=20000)
    args = parser.parse_args()
    Driver(args).run()


if __name__ == "__main__":
    main()
