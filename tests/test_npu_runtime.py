"""Backend-neutral runtime tests (C01/C02): no real NPU required.

The fake ``torch_npu`` cases prove call routing and failure semantics only;
they are not evidence that an Ascend device or any fused operator works.
"""

from __future__ import annotations

import importlib
import random
import sys
import types
from contextlib import contextmanager

import pytest
import torch

from vlm_dflash import runtime
from vlm_dflash.config import validate_runtime_config


class _FakeNpu:
    def __init__(self, *, available: bool = True, count: int = 1):
        self.available = bool(available)
        self.count = int(count)
        self.set_devices: list[torch.device] = []
        self.seeds: list[int] = []
        self.rng = [[1, 2, 3], [4, 5, 6]][: self.count]
        self.restored = None

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def set_device(self, device):
        self.set_devices.append(torch.device(device))

    def synchronize(self, device):
        self.set_devices.append(("sync", torch.device(device)))

    def manual_seed_all(self, seed):
        self.seeds.append(int(seed))

    def get_rng_state_all(self):
        return [state[:] for state in self.rng]

    def set_rng_state_all(self, states):
        self.restored = [state[:] for state in states]


def _install_fake_npu(monkeypatch, fake: _FakeNpu):
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    real_import = importlib.import_module
    monkeypatch.setattr(
        runtime,
        "importlib",
        types.SimpleNamespace(
            import_module=lambda name: sys.modules["torch_npu"] if name == "torch_npu" else real_import(name)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_fusion_attention=None))


def test_device_type_does_not_require_the_backend():
    assert runtime.device_type("cpu") == "cpu"
    assert runtime.device_type("cuda:3") == "cuda"
    assert runtime.device_type("npu:2") == "npu"
    assert runtime.device_type("tpu:0") == "tpu"
    with pytest.raises(runtime.DeviceResolutionError, match="Unrecognised device"):
        runtime.device_type("not-a-device")


def test_cpu_device_resolution_and_capabilities():
    assert runtime.resolve_device("cpu") == torch.device("cpu")
    assert runtime.pin_memory_supported("cpu") is False
    assert runtime.compile_supported("npu:0") is False
    assert runtime.max_memory_allocated(torch.device("cpu")) is None
    assert runtime.reset_peak_memory_stats(torch.device("cpu")) is False
    assert runtime.device_memory_info(torch.device("cpu")) is None


def test_cpu_wall_clock_timing_is_deterministic_enough():
    run = runtime.TimingRun.start(torch.device("cpu"))
    sum(range(1000))
    run.stop()
    assert run.elapsed_seconds() >= 0.0
    assert run.start_event is None


def test_explicit_npu_without_torch_npu_fails_without_fallback(monkeypatch):
    def missing(name):
        if name == "torch_npu":
            raise ImportError("simulated: no torch_npu in this environment")
        return importlib.import_module(name)

    monkeypatch.setattr(runtime, "importlib", types.SimpleNamespace(import_module=missing))
    with pytest.raises(runtime.DeviceResolutionError, match="torch_npu could not be imported"):
        runtime.resolve_device("npu:0")
    with pytest.raises(ValueError, match="torch_npu/CANN is unavailable"):
        validate_runtime_config({"train": {"device": "npu:0"}})


def test_fake_npu_resolution_never_returns_cpu_or_cuda(monkeypatch):
    fake = _FakeNpu()
    _install_fake_npu(monkeypatch, fake)
    assert runtime.npu_available() is True
    # A fake module cannot make torch itself understand ``torch.device('npu')``.
    # The resolver must still fail loudly rather than fall back to CPU/CUDA.
    with pytest.raises(runtime.DeviceResolutionError, match="did not register the 'npu' device type"):
        runtime.ensure_runtime_device("npu:0", context="train.device")
    assert fake.set_devices == []


def test_fake_npu_unavailable_is_an_actionable_error(monkeypatch):
    fake = _FakeNpu(available=False, count=0)
    _install_fake_npu(monkeypatch, fake)
    assert runtime.npu_available() is False
    with pytest.raises(runtime.DeviceResolutionError, match="torch.npu.is_available\\(\\) is False"):
        runtime.ensure_runtime_device("npu:0")
    assert fake.set_devices == []


def test_rng_round_trip_covers_python_torch_and_numpy():
    pytest.importorskip("numpy")
    import numpy as np

    runtime.seed_all(1234, torch.device("cpu"))
    first = (random.random(), float(torch.rand(())), float(np.random.rand()))
    state = runtime.capture_rng_state()
    second = (random.random(), float(torch.rand(())), float(np.random.rand()))
    runtime.restore_rng_state(state)
    replayed = (random.random(), float(torch.rand(())), float(np.random.rand()))
    assert replayed == second
    assert first != second


def test_fake_npu_rng_state_is_saved_and_restored(monkeypatch):
    fake = _FakeNpu()
    _install_fake_npu(monkeypatch, fake)
    state = runtime.capture_rng_state()
    assert state["npu"] == fake.rng
    runtime.restore_rng_state(state)
    assert fake.restored == fake.rng
    runtime.seed_all(7, "npu:0")
    assert fake.seeds == [7]


def test_flex_attention_is_rejected_before_an_npu_run(monkeypatch):
    _install_fake_npu(monkeypatch, _FakeNpu())
    with pytest.raises(ValueError, match="flex_attention is not enabled on Ascend NPU"):
        validate_runtime_config(
            {"train": {"device": "npu:0"}, "drafter": {"attention_backend": "flex_attention"}}
        )


def test_environment_report_is_import_safe_without_npu():
    report = runtime.environment_report()
    assert report["torch"]
    assert report["npu"]["available"] is False
    assert report["cuda"]["available"] is False


def test_peak_memory_apis_report_unavailable_as_none_not_zero(monkeypatch):
    def boom(_device):
        raise RuntimeError("no accelerator")

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", boom)
    monkeypatch.setattr(torch.cuda, "memory_allocated", boom)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", boom)
    device = torch.device("cuda", 0)
    assert runtime.max_memory_allocated(device) is None
    assert runtime.memory_allocated(device) is None
    assert runtime.reset_peak_memory_stats(device) is False


def test_accelerator_devices_include_parameters_and_buffers():
    class _FakeDevice:
        def __init__(self, kind: str, index: int):
            self.type = kind
            self.index = index

    class _FakeTensor:
        def __init__(self, device):
            self.device = device

    class _FakeModel:
        def parameters(self):
            return iter([_FakeTensor(_FakeDevice("npu", 0))])

        def buffers(self):
            return iter([_FakeTensor(_FakeDevice("npu", 1))])

    devices = runtime.accelerator_devices(_FakeModel())
    assert [(device.type, device.index) for device in devices] == [("npu", 0), ("npu", 1)]


def test_event_timer_records_both_boundaries_on_selected_device(monkeypatch):
    selected = torch.device("privateuseone", 0)
    current = [torch.device("privateuseone", 1)]
    recorded = []

    @contextmanager
    def device_context(device):
        previous = current[0]
        current[0] = device
        try:
            yield
        finally:
            current[0] = previous

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            recorded.append(current[0])

        def elapsed_time(self, other):
            return 250.0

    api = types.SimpleNamespace(Event=Event, device=device_context)
    monkeypatch.setattr(runtime, "_api_for", lambda device: api)
    monkeypatch.setattr(runtime, "synchronize", lambda device: None)
    timer = runtime.TimingRun.start(selected)
    # elapsed_seconds() must also stop an open event timer.
    assert timer.elapsed_seconds() == 0.25
    assert recorded == [selected, selected]
    assert current[0].index == 1

