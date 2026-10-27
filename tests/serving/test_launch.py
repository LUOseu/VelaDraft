from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from vlm_dflash.serving import launch
from vlm_dflash.serving.launch import build_server_command, load_config


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "LOCK_ROOT", str(tmp_path))
    monkeypatch.setattr(launch, "gpu_uuid", lambda index: f"GPU-{index}")
    monkeypatch.setattr(launch, "gpu_compute_processes", lambda index: [])
    monkeypatch.setattr(launch, "gpu_free_mib", lambda index: 30000)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,9")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def assert_unlocked(index=2):
    fd = launch.acquire_gpu_lock(index)
    assert os.get_inheritable(fd)
    os.close(fd)


def test_ar_and_ngram_configs_build_commands():
    ar = load_config("configs/platform_extensions/serving/ar.json")
    ngram = load_config("configs/platform_extensions/serving/ngram.json")
    ar_cmd = build_server_command(ar, port=8100, gpu_index=0)
    ngram_cmd = build_server_command(ngram, port=8101, gpu_index=0)
    assert "--speculative-config" not in ar_cmd
    assert "--speculative-config" in ngram_cmd
    spec = ngram_cmd[ngram_cmd.index("--speculative-config") + 1]
    assert '"method":"ngram"' in spec
    assert "--no-enable-prefix-caching" in ar_cmd
    assert ar_cmd[ar_cmd.index("--port") + 1] == "8100"
    assert ngram_cmd[ngram_cmd.index("--port") + 1] == "8101"


def test_guard_checks_inside_lock_and_pins_physical_gpu(gpu, monkeypatch):
    probes = []

    def check_processes(index):
        with pytest.raises(launch.LaunchError, match="locked"):
            launch.acquire_gpu_lock(index)
        probes.append("processes")
        return []

    def check_memory(index):
        with pytest.raises(launch.LaunchError, match="locked"):
            launch.acquire_gpu_lock(index)
        probes.append("memory")
        return 30000

    monkeypatch.setattr(launch, "gpu_compute_processes", check_processes)
    monkeypatch.setattr(launch, "gpu_free_mib", check_memory)
    with launch.gpu_guard(2) as uuid:
        assert uuid == "GPU-2"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        with pytest.raises(launch.LaunchError, match="locked"):
            with launch.gpu_guard(2):
                pytest.fail("a second guard must not enter")
    assert probes == ["processes", "memory"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,9"
    assert_unlocked()


@pytest.mark.parametrize("failure", ["busy", "memory", "probe", "body"])
def test_guard_releases_on_all_failures(gpu, monkeypatch, failure):
    if failure == "busy":
        monkeypatch.setattr(launch, "gpu_compute_processes", lambda index: ["GPU-2, 123, busy"])
    elif failure == "memory":
        monkeypatch.setattr(launch, "gpu_free_mib", lambda index: 100)
    elif failure == "probe":
        def fail_probe(index):
            raise OSError("nvidia-smi failed")
        monkeypatch.setattr(launch, "gpu_compute_processes", fail_probe)
    with pytest.raises((launch.LaunchError, OSError, RuntimeError)):
        with launch.gpu_guard(2):
            if failure == "body":
                raise RuntimeError("initialization failed")
            pytest.fail("unsafe GPU must be rejected")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,9"
    assert_unlocked()


def test_guard_restores_unset_visibility(gpu, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    with launch.gpu_guard(2):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "CUDA_DEVICE_ORDER" not in os.environ


def test_serve_keeps_lock_through_exec_and_releases_on_failure(gpu, monkeypatch):
    def fail_exec(executable, command):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        with pytest.raises(launch.LaunchError, match="locked"):
            launch.acquire_gpu_lock(2)
        raise OSError("exec failed")

    monkeypatch.setattr(launch.os, "execv", fail_exec)
    with pytest.raises(OSError, match="exec failed"):
        launch.serve("configs/platform_extensions/serving/ar.json", gpu_index=2)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,9"
    assert_unlocked()


@pytest.mark.parametrize("key", ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"])
def test_rejects_multi_gpu_topology_before_access(gpu, monkeypatch, key):
    config = load_config("configs/platform_extensions/serving/ar.json")
    config[key] = 2
    monkeypatch.setattr(launch, "load_config", lambda path: config)
    with pytest.raises(launch.LaunchError, match=key):
        launch.serve("unused")
    with pytest.raises(launch.LaunchError, match=key):
        launch.build_server_command(config, port=8100, gpu_index=2)


def test_guard_rejects_already_initialized_cuda(gpu, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    with pytest.raises(launch.LaunchError, match="already initialized"):
        with launch.gpu_guard(2):
            pytest.fail("changing visibility after initialization is unsafe")
    assert_unlocked()


@pytest.mark.parametrize("visibility", ["9", "", "-1", "GPU-9"])
def test_guard_never_widens_existing_visibility(gpu, monkeypatch, visibility):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    with pytest.raises(launch.LaunchError, match="refusing to widen"):
        with launch.gpu_guard(2):
            pytest.fail("selection outside existing visibility is unsafe")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visibility
    assert_unlocked()


def test_guard_accepts_matching_uuid_visibility(gpu, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-2")
    with launch.gpu_guard(2) as uuid:
        assert uuid == "GPU-2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-2"


def test_lock_closes_fd_when_setting_inheritance_fails(gpu, monkeypatch):
    opened = []
    real_open = os.open

    def track_open(*args):
        fd = real_open(*args)
        opened.append(fd)
        return fd

    def fail_inheritance(fd, enabled):
        raise OSError("cannot inherit fd")

    monkeypatch.setattr(launch.os, "open", track_open)
    monkeypatch.setattr(launch.os, "set_inheritable", fail_inheritance)
    with pytest.raises(OSError, match="cannot inherit fd"):
        launch.acquire_gpu_lock(2)
    with pytest.raises(OSError):
        os.fstat(opened[0])
