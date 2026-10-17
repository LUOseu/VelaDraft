"""Deterministic vLLM launch config + GPU flock helper."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

LOCK_ROOT = "/tmp/veladraft_gpu_locks"


class LaunchError(RuntimeError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("model", "served_model_name", "dtype", "max_model_len", "gpu_memory_utilization"):
        if key not in config:
            raise LaunchError(f"config is missing required key {key!r}")
    return config


def gpu_uuid(index: int) -> str:
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    if not out:
        raise LaunchError(f"no GPU UUID for index {index}")
    return out


def gpu_free_mib(index: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).strip()
    return int(out)


def gpu_compute_processes(index: int) -> list[str]:
    uuid = gpu_uuid(index)
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    return [line for line in out.splitlines() if line.startswith(uuid)]


def acquire_gpu_lock(index: int) -> int:
    """Acquire the per-UUID flock and return an inheritable fd.

    The lock coordinates only jobs that follow the same convention.  It does
    not replace the explicit compute-process and free-memory checks.
    """

    uuid = gpu_uuid(index)
    Path(LOCK_ROOT).mkdir(parents=True, exist_ok=True)
    lock_path = Path(LOCK_ROOT) / f"{uuid}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise LaunchError(f"GPU {index} ({uuid}) is locked by another compliant job: {lock_path}") from exc
    os.set_inheritable(fd, True)
    return fd


def build_server_command(config: dict[str, Any], *, port: int, gpu_index: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config["model"]),
        "--served-model-name",
        str(config["served_model_name"]),
        "--dtype",
        str(config["dtype"]),
        "--tensor-parallel-size",
        str(int(config.get("tensor_parallel_size", 1))),
        "--gpu-memory-utilization",
        str(float(config["gpu_memory_utilization"])),
        "--max-model-len",
        str(int(config["max_model_len"])),
        "--host",
        str(config.get("host", "127.0.0.1")),
        "--port",
        str(int(port)),
        "--limit-mm-per-prompt",
        json.dumps(config.get("limit_mm_per_prompt", {"image": 1}), separators=(",", ":")),
        "--mm-processor-kwargs",
        json.dumps(config.get("mm_processor_kwargs", {}), separators=(",", ":")),
        "--uvicorn-log-level",
        str(config.get("uvicorn_log_level", "info")),
    ]
    if config.get("enable_prefix_caching", False):
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")
    if config.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    if config.get("max_num_seqs"):
        cmd.extend(["--max-num-seqs", str(int(config["max_num_seqs"]))])
    if config.get("max_num_batched_tokens"):
        cmd.extend(["--max-num-batched-tokens", str(int(config["max_num_batched_tokens"]))])
    if config.get("speculative_config"):
        cmd.extend(["--speculative-config", json.dumps(config["speculative_config"], separators=(",", ":"))])
    if config.get("async_scheduling") is False:
        cmd.append("--no-async-scheduling")
    elif config.get("async_scheduling") is True:
        cmd.append("--async-scheduling")
    return cmd


def serve(config_path: str | Path, *, port: int | None = None, gpu_index: int | None = None) -> None:
    config = load_config(config_path)
    index = int(gpu_index if gpu_index is not None else config.get("gpu_index", 0))
    port = int(port if port is not None else config.get("port", 8100))
    processes = gpu_compute_processes(index)
    if processes:
        raise LaunchError(
            f"GPU {index} already has compute processes; refusing to launch:\n" + "\n".join(processes)
        )
    free_mib = gpu_free_mib(index)
    if free_mib < int(config.get("min_free_memory_mib", 20000)):
        raise LaunchError(f"GPU {index} has only {free_mib} MiB free; refusing to launch")
    acquire_gpu_lock(index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ["PYTHONUNBUFFERED"] = "1"
    if config.get("attention_backend"):
        os.environ["VLLM_ATTENTION_BACKEND"] = str(config["attention_backend"])
    cmd = build_server_command(config, port=port, gpu_index=index)
    os.execv(sys.executable, cmd)
