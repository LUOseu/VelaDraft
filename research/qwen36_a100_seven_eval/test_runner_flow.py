"""CPU tests of the engine stage and the CLI orchestration with fake vLLM / torch / GPU lease."""

import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from research.qwen36_a100_seven_eval import cli, lane, publish, settings, stage
from research.qwen36_a100_seven_eval.test_runner import (
    CONFIG,
    FakeApi,
    manifest_for,
    sessions_two_topics,
    write_shard,
)


class FakeLLM:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shutdowns = 0
        FakeLLM.instances.append(self)
        self.llm_engine = types.SimpleNamespace(
            engine_core=types.SimpleNamespace(shutdown=self.shutdown)
        )

    def shutdown(self):
        self.shutdowns += 1

    def collective_rpc(self, name, timeout=None):
        if name == "backend_metadata":
            spec = self.kwargs.get("speculative_config")
            return [
                {"speculative_tokens": spec["num_speculative_tokens"] if spec else None}
            ]
        return [{"method": "meta"}]


class FakeLease:
    fail_validate = False

    def __init__(self, index, path):
        self.path = Path(path)
        self.uuid = "GPU-test"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def validate(self, timed_windows=None):
        if FakeLease.fail_validate:
            raise ValueError("foreign app inside a timed window")
        return {"status": "complete", "windows": len(timed_windows or [])}


def install_fakes(monkeypatch, *, tokens=3, finish="stop", prompt_ids=(1, 2)):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        get_device_name=lambda i: "NVIDIA A100-SXM4-80GB"
    )
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace()
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: "processor"
    )
    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = lambda **kw: kw
    for name, module in {
        "torch": torch,
        "PIL": pil,
        "transformers": transformers,
        "vllm": vllm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    import vlm_dflash.eval.native_resources as resources
    import vlm_dflash.eval.native_runtime as runtime

    monkeypatch.setattr(resources, "GPULease", FakeLease)
    calls = []

    def request(llm, processor, params, torch_, image, session, turn, reference, plan):
        calls.append((session["id"], turn))
        n = params.get("max_tokens") if params.get("ignore_eos") else tokens
        return {
            "id": session["id"],
            "turn": turn,
            "tokens": [7] * n,
            "prompt_token_ids": list(prompt_ids),
            "text": "t",
            "finish_reason": "length" if params.get("ignore_eos") else finish,
            "wall_seconds": 0.5,
            "prompt": "p",
            "messages": [],
        }

    monkeypatch.setattr(runtime, "request", request)
    monkeypatch.setattr(
        runtime,
        "metrics",
        lambda llm: {"rounds": 1.0, "proposed": 3.0, "accepted": 2.0},
    )
    monkeypatch.setattr(
        runtime,
        "metric_delta",
        lambda a, b: {"status": "complete", "rounds": 1, "proposed": 3, "accepted": 2},
    )
    monkeypatch.setattr(stage.importlib.metadata, "version", lambda name: "0.32.0")
    monkeypatch.setenv("LANE_GPU", "0")
    FakeLease.fail_validate = False
    FakeLLM.instances.clear()
    return calls


def setup(tmp_path, methods=("dflash",)):
    run = tmp_path / "run"
    settings.save_json(run / "manifest.json", manifest_for(tmp_path, methods))
    settings.save_json(run / "sessions.json", sessions_two_topics())
    return run


def args(run, method, stage_name, protocol="natural"):
    return types.SimpleNamespace(
        run_dir=str(run), method=method, stage=stage_name, protocol=protocol
    )


def test_reference_then_sd_stage_writes_shards_and_resumes(tmp_path, monkeypatch):
    run = setup(tmp_path)
    calls = install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    ref = stage.load_reference(run, sessions_two_topics())
    assert ref[("s2", 1)]["prompt"] == "p"  # reference rows keep the full prompt
    assert stage.run_stage(args(run, "dflash", "sd_r0")) == 0
    llm = FakeLLM.instances[-1]
    assert (
        llm.kwargs["speculative_config"]["num_speculative_tokens"] == 3
        and llm.shutdowns == 1
    )
    shard = stage.shard_path(run, "dflash", "natural", "sd_r0", settings.CATEGORIES[5])
    row = json.loads(shard.read_text().splitlines()[0])
    assert "prompt" not in row and row["output_tokens"] == 3 and row["arm"] == "sd"
    meta = settings.read_json(run / "stages" / "dflash" / "natural__sd_r0.json")
    assert meta["status"] == "complete" and meta["resources"]["status"] == "complete"
    assert meta["gpu_name"].startswith("NVIDIA A100")
    before = len(calls)
    assert (
        stage.run_stage(args(run, "dflash", "sd_r0")) == 0 and len(calls) == before
    )  # nothing left


def test_stats_and_fixed512_stages(tmp_path, monkeypatch):
    run = setup(tmp_path, ("dflash2",))
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    assert stage.run_stage(args(run, "dflash2", "stats")) == 0
    row = json.loads(
        stage.shard_path(
            run, "dflash2", "natural", "stats", settings.CATEGORIES[0]
        ).read_text()
    )
    assert row["official_counters"]["accepted"] == 2
    assert FakeLLM.instances[-1].kwargs["disable_log_stats"] is False
    assert stage.run_stage(args(run, "dflash2", "ar_r0", "fixed512")) == 0
    row = json.loads(
        stage.shard_path(
            run, "dflash2", "fixed512", "ar_r0", settings.CATEGORIES[0]
        ).read_text()
    )
    assert row["output_tokens"] == 512 and row["finish_reason"] == "length"


def test_prompt_drift_and_wrong_k_fail_the_stage(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    install_fakes(monkeypatch, prompt_ids=(9,))
    assert stage.run_stage(args(run, "dflash", "sd_r0")) == 1
    meta = settings.read_json(run / "stages" / "dflash" / "natural__sd_r0.json")
    assert "prompt ids differ" in meta["error"] and FakeLLM.instances[-1].shutdowns == 1
    install_fakes(monkeypatch)
    monkeypatch.setattr(
        FakeLLM,
        "collective_rpc",
        lambda self, name, timeout=None: [{"speculative_tokens": 2}],
    )
    assert stage.run_stage(args(run, "dflash", "sd_r1")) == 1
    assert (
        "K=2"
        in settings.read_json(run / "stages" / "dflash" / "natural__sd_r1.json")[
            "error"
        ]
    )


def test_contaminated_resources_withdraw_shards(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    FakeLease.fail_validate = True
    assert stage.run_stage(args(run, "dflash", "ar_r0")) == 1
    assert stage.missing_topics(
        run, sessions_two_topics(), "dflash", "natural", "ar_r0"
    )
    assert list((run / "raw" / "mmspec" / "dflash").glob("*.invalid*"))
    meta = settings.read_json(run / "stages" / "dflash" / "natural__ar_r0.json")
    assert meta["status"] == "invalid_resources"


def test_stage_guards(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    with pytest.raises(ValueError, match="ar_default"):
        stage.run_stage(args(run, "ar_default", "sd_r0"))
    monkeypatch.setattr(stage.importlib.metadata, "version", lambda name: "0.31.0")
    assert stage.run_stage(args(run, "ar", "reference")) == 1
    assert (
        stage.main(["--run-dir", str(run), "--method", "ar", "--stage", "reference"])
        == 1
    )


def test_stock_ar_runs_without_audit_worker(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0

    def no_worker(self, name, timeout=None):
        raise AttributeError(name)

    monkeypatch.setattr(FakeLLM, "collective_rpc", no_worker)
    assert stage.run_stage(args(run, "ar_default", "ar_r0")) == 0
    assert (
        "unavailable"
        in settings.read_json(run / "stages" / "ar_default" / "natural__ar_r0.json")[
            "worker"
        ]
    )


def test_lane_run_stage_invokes_hook_and_module(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        lane.subprocess,
        "run",
        lambda cmd, **kw: (
            seen.append((cmd, kw.get("env", {}))) or types.SimpleNamespace(returncode=0)
        ),
    )
    log = tmp_path / "lane.log"
    assert (
        lane.run_stage(tmp_path, 3, "dflash", "natural", "sd_r0", log, "free_gpu.sh")
        == 0
    )
    assert seen[0][0] == ["free_gpu.sh", "3"]
    cmd, env = seen[1]
    assert cmd[-6:] == [
        "--method",
        "dflash",
        "--protocol",
        "natural",
        "--stage",
        "sd_r0",
    ]
    assert (
        env["CUDA_VISIBLE_DEVICES"] == "3" and env["VELADRAFT_GPU_OWNERSHIP"] == "pid"
    )


# ---------------------------------------------------------------- cli orchestration
def fake_freeze(monkeypatch, tmp_path):
    from research.qwen36_a100_seven_eval import prepare

    monkeypatch.setattr(cli.data_mod, "fetch", lambda repo, commit, dest: dest)
    monkeypatch.setattr(
        cli.data_mod,
        "build_sessions",
        lambda clone, suite: (
            sessions_two_topics(),
            {"formal": suite == "mmspec-full", "sessions": 2},
        ),
    )
    monkeypatch.setattr(
        prepare,
        "target_identity",
        lambda path, rev, cache: {
            "path": str(path),
            "revision": rev,
            "stop_token_ids": [1],
        },
    )
    monkeypatch.setattr(
        prepare,
        "freeze_checkpoints",
        lambda methods, config, cache: {m: {"path": f"/e/{m}"} for m in methods},
    )
    monkeypatch.setattr(
        cli,
        "gpus_info",
        lambda idx: [
            {"index": str(i), "name": "NVIDIA A100-SXM4-80GB", "memory_used_mib": "0"}
            for i in idx
        ],
    )
    monkeypatch.setattr(cli, "git", lambda *a: "" if a[0] == "status" else "c0ffee" * 7)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda name: "0.32.0")


class FakeProc:
    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0


def test_create_run_freezes_manifest_and_links_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path / "root"))
    fake_freeze(monkeypatch, tmp_path)
    config = settings.apply_overrides(settings.load_config(CONFIG), methods=["dflash"])
    run = cli.create_run(config, "mmspec-smoke")
    manifest = settings.read_json(run / "manifest.json")
    assert manifest["methods"] == ["dflash"] and manifest["code"]["commit"].startswith(
        "c0ffee"
    )
    assert cli.latest_run(tmp_path / "root", "mmspec-smoke") == run.resolve()
    monkeypatch.setattr(
        cli, "git", lambda *a: " M file" if a[0] == "status" else "c0ffee" * 7
    )
    with pytest.raises(ValueError, match="uncommitted"):
        cli.create_run(config, "mmspec-smoke")


def test_main_runs_lanes_publisher_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path / "root"))
    fake_freeze(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "preflight",
        lambda config, suite, publish_: {
            "ok": True,
            "checks": [{"name": "x", "ok": True, "detail": "", "action": None}],
        },
    )
    launched = []
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda cmd, **kw: launched.append((cmd, kw.get("env"))) or FakeProc(cmd),
    )
    code = cli.main(
        [
            "--config",
            str(CONFIG),
            "--suite",
            "mmspec-smoke",
            "--methods",
            "dflash",
            "--gpus",
            "0,1",
            "--publish",
        ]
    )
    assert code == 3  # nothing ran, so the run is partial
    lanes = [c for c, _ in launched if "research.qwen36_a100_seven_eval.lane" in c]
    pubs = [
        (c, e) for c, e in launched if "research.qwen36_a100_seven_eval.publish" in c
    ]
    assert len(lanes) == 2 and len(pubs) == 1 and "HF_HUB_OFFLINE" not in pubs[0][1]
    run = cli.latest_run(tmp_path / "root", "mmspec-smoke")
    status = settings.read_json(run / "status.json")
    assert status["status"] == "partial" and status["jobs"] == 4
    # Resume: same commit continues the run; a changed commit or method list is refused.
    (run / "claims" / "dflash__natural").mkdir(parents=True)
    assert (
        cli.main(
            [
                "--config",
                str(CONFIG),
                "--suite",
                "mmspec-smoke",
                "--methods",
                "dflash",
                "--resume",
            ]
        )
        == 3
    )
    assert not (run / "claims" / "dflash__natural").exists()
    with pytest.raises(SystemExit, match="methods"):
        cli.main(
            [
                "--config",
                str(CONFIG),
                "--suite",
                "mmspec-smoke",
                "--methods",
                "dfly",
                "--resume",
            ]
        )
    monkeypatch.setattr(cli, "git", lambda *a: "" if a[0] == "status" else "beef" * 10)
    with pytest.raises(SystemExit, match="commit changed"):
        cli.main(
            [
                "--config",
                str(CONFIG),
                "--suite",
                "mmspec-smoke",
                "--methods",
                "dflash",
                "--resume",
            ]
        )


def test_main_stops_on_failed_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(
        cli,
        "preflight",
        lambda config, suite, publish_: {
            "ok": False,
            "checks": [
                {
                    "name": "hf_write",
                    "ok": False,
                    "detail": "no token",
                    "action": "log in",
                }
            ],
        },
    )
    assert cli.main(["--config", str(CONFIG)]) == 2
    assert "-> log in" in capsys.readouterr().out


def test_publish_main_once(tmp_path, monkeypatch, capsys):
    run = setup(tmp_path)
    settings.save_json(
        run / "status.json", {"status": "complete", "suite": "mmspec-smoke"}
    )
    write_shard(
        run, "dflash", "natural", "ar_r0", settings.CATEGORIES[0], [{"id": "s1"}]
    )
    api = FakeApi()
    api.fail_once = False
    real = publish.Publisher

    class Bound(real):
        def __init__(self, r, repo, path):
            super().__init__(r, repo, path, api=api)

    monkeypatch.setattr(publish, "Publisher", Bound)
    assert (
        publish.main(
            ["--run-dir", str(run), "--repo", "o/r", "--path", "eval/x", "--once"]
        )
        == 0
    )
    assert '"uploaded"' in capsys.readouterr().out
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert (
        publish.loop(run, "o/r", "eval/x", 0) == 0
    )  # complete status and nothing left: exits


# ---------------------------------------------------------------- prepare
def test_hash_cache_and_target_identity(tmp_path):
    from research.qwen36_a100_seven_eval import prepare

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text("{}")
    (target / "generation_config.json").write_text(json.dumps({"eos_token_id": [5, 6]}))
    cache = tmp_path / "cache.json"
    ident = prepare.target_identity(target, "rev", cache)
    assert ident["stop_token_ids"] == [5, 6] and set(ident["files"]) == {
        "config.json",
        "generation_config.json",
    }
    assert len(settings.read_json(cache)) == 2
    (target / "generation_config.json").write_text(json.dumps({"eos_token_id": 9}))
    assert prepare.target_identity(target, "rev", cache)["stop_token_ids"] == [9]
    with pytest.raises(FileNotFoundError):
        prepare.target_identity(tmp_path / "missing", "rev", cache)


def test_checkpoint_local_hit_download_and_mismatch(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import prepare

    local = tmp_path / "local.pt"
    local.write_bytes(b"weights")
    entry = {
        "repo": "o/m",
        "revision": "abc",
        "size": 7,
        "sha256": settings.sha256(local),
    }
    cache = tmp_path / "c.json"
    assert prepare.fetch_checkpoint("dflash", entry, [local], tmp_path, cache) == local
    downloaded = tmp_path / "dl.pt"
    downloaded.write_bytes(b"weights")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name, revision, local_dir: str(downloaded)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert (
        prepare.fetch_checkpoint(
            "dflash", entry, [tmp_path / "absent.pt"], tmp_path, cache
        )
        == downloaded
    )
    downloaded.write_bytes(b"other!!")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        prepare.fetch_checkpoint("dflash", entry, [], tmp_path, cache)


def raw_export(root, method, arch_config):
    raw = root / "raw" / method
    raw.mkdir(parents=True)
    (raw / "config.json").write_text(json.dumps(arch_config))
    (raw / "export_receipt.json").write_text("{}")
    (raw / "model.safetensors").write_bytes(b"w")
    return raw


def test_export_dispatch_per_method(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import prepare

    ran = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda cmd, **kw: ran.append(cmd))
    raw_export(tmp_path, "dflash", {"architectures": ["DFlashDraftModel"]})
    final = prepare.export("dflash", tmp_path / "ckpt.pt", tmp_path)
    assert (final / "model.safetensors").samefile(
        tmp_path / "raw/dflash/model.safetensors"
    ) and not ran
    assert (
        prepare.export("dflash", tmp_path / "ckpt.pt", tmp_path) == final
    )  # idempotent
    seen = {}
    for method, module in (
        ("jetspec", "prepare_native"),
        ("dspark", "prepare_dspark"),
        ("dfly", "prepare_custom"),
    ):
        fake = types.ModuleType(module)

        def fake_prepare(*a, m=method):
            seen[m] = a
            dest = Path(a[-1])
            dest.mkdir(parents=True)
            (dest / "config.json").write_text("{}")
            (dest / "model.safetensors").write_bytes(b"w")

        fake.prepare = fake_prepare
        monkeypatch.setitem(
            sys.modules, f"research.qwen36_seven_20261102.{module}", fake
        )
        prepare.export(method, tmp_path / "ckpt.pt", tmp_path / method)
        assert ran[-1][-2:] == [
            str(tmp_path / "ckpt.pt"),
            str(tmp_path / method / "raw" / method),
        ]
    assert (
        seen["jetspec"][0] == "jetspec"
        and len(seen["dspark"]) == 2
        and seen["dfly"][0] == "dfly"
    )


def test_freeze_checkpoints_records_pins(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import prepare

    final = tmp_path / "final"
    final.mkdir()
    (final / "config.json").write_text("{}")
    (final / "model.safetensors").write_bytes(b"w")
    monkeypatch.setattr(
        prepare, "fetch_checkpoint", lambda m, e, c, d, cache: tmp_path / "ck.pt"
    )
    monkeypatch.setattr(prepare, "export", lambda m, ck, root: final)
    config = {
        "checkpoints": {
            "work_dir": str(tmp_path / "w"),
            "local_root": str(tmp_path / "l"),
        }
    }
    out = prepare.freeze_checkpoints(["dflash", "dfly"], config, tmp_path / "c.json")
    pinned = settings.checkpoint_manifest()
    assert out["dfly"]["revision"] == pinned["dfly"]["revision"] and out["dflash"][
        "path"
    ] == str(final)


def test_no_raw_results_visible_before_resource_validation(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    original = FakeLease.validate

    def validate(self, timed_windows=None):
        assert not list((run / "raw").glob("**/*.jsonl"))
        assert not any(k.startswith("raw/") for k in publish.candidates(run))
        return original(self, timed_windows)

    monkeypatch.setattr(FakeLease, "validate", validate)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    assert list((run / "raw").glob("**/*.jsonl"))


def test_resume_rejects_changed_physical_gpu(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    assert stage.run_stage(args(run, "dflash", "ar_r0")) == 0
    settings.save_json(
        run / "bindings/dflash__natural.json", {"index": 0, "gpu_id": "other-card"}
    )
    assert stage.run_stage(args(run, "dflash", "sd_r0")) == 1
    assert not list((run / "raw/mmspec/dflash").glob("*sd_r0*.jsonl"))


def test_resource_lease_entry_failure_is_reported(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)

    def occupied(self):
        raise ValueError("occupied")

    monkeypatch.setattr(FakeLease, "__enter__", occupied)
    monkeypatch.setattr(
        FakeLease, "__exit__", lambda *a: pytest.fail("lease never entered")
    )
    assert stage.run_stage(args(run, "ar", "reference")) == 1
    assert (
        "occupied"
        in settings.read_json(run / "stages/_reference/natural__reference.json")[
            "error"
        ]
    )


def test_relative_root_symlink_and_launcher_lock(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("Q36_A100_ROOT", "./work")
    fake_freeze(monkeypatch, tmp_path)
    config = settings.apply_overrides(settings.load_config(CONFIG), methods=["dflash"])
    run = cli.create_run(config, "mmspec-smoke")
    assert cli.latest_run(tmp_path / "work", "mmspec-smoke") == run
    with (run / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active launcher"):
            cli.run_all(run, [0], False)


def test_resume_missing_and_invalid_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "preflight", lambda *a: {"ok": True, "checks": []})
    with pytest.raises(SystemExit, match="no existing run"):
        cli.main(["--config", str(CONFIG), "--resume"])
    for bad in ("../escape", ".", ".."):
        with pytest.raises(SystemExit):
            cli.main(["--config", str(CONFIG), "--resume", "--run-id", bad])
    for gpus in ("0,0", "-1"):
        with pytest.raises(ValueError, match="GPU indices"):
            cli.main(["--config", str(CONFIG), "--gpus", gpus])


def test_incomplete_or_modified_export_is_rejected(tmp_path):
    from research.qwen36_a100_seven_eval import prepare

    final = tmp_path / "vllm032/dflash"
    final.mkdir(parents=True)
    (final / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        prepare.export("dflash", tmp_path / "unused", tmp_path)
    (final / "model.safetensors").write_bytes(b"w")
    settings.save_json(final / "ready.json", {"config.json": "wrong"})
    with pytest.raises(ValueError, match="receipt"):
        prepare.export("dflash", tmp_path / "unused", tmp_path)


def test_preflight_checks_all_requested_gpus_and_hf(tmp_path, monkeypatch):
    config = settings.load_config(CONFIG)
    config["run"].update(root=str(tmp_path), gpus=[0, 1])
    config["engine"]["cuda_home"] = str(tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/nvcc").touch()
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda n: "0.32.0")
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda *a, **kw: (
            "0, A100, GPU-a, 80000, 0, 570\n1, A100, GPU-b, 80000, 900, 570\n"
        ),
    )
    assert len(cli.gpus_info([0, 1])) == 2
    monkeypatch.setattr(publish, "check_write_access", lambda repo: {"ok": True})
    report = cli.preflight(config, "mmspec-full", True)
    checks = {c["name"]: c for c in report["checks"]}
    assert (
        checks["gpus"]["ok"]
        and not checks["gpus_free"]["ok"]
        and checks["hf_write"]["ok"]
    )
    config["run"]["gpus"] = [0, 1, 2]
    assert not next(
        c
        for c in cli.preflight(config, "mmspec-full", False)["checks"]
        if c["name"] == "gpus"
    )["ok"]
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("missing")),
    )
    monkeypatch.setattr(
        publish,
        "check_write_access",
        lambda *a: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    report = cli.preflight(config, "mmspec-full", True)
    assert not next(c for c in report["checks"] if c["name"] == "hf_write")["ok"]


def test_launcher_cleans_up_running_process_groups(tmp_path, monkeypatch):
    import subprocess

    run = setup(tmp_path)
    killed = []
    process = types.SimpleNamespace(pid=123, poll=lambda: None)
    waits = []

    def wait(timeout=None):
        waits.append(timeout)
        if timeout:
            raise subprocess.TimeoutExpired("lane", timeout)

    process.wait = wait

    def start(run, gpus, publish_, fd, children):
        children.append(process)
        raise RuntimeError("launch failed")

    monkeypatch.setattr(cli, "_run_all", start)
    monkeypatch.setattr(cli.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(RuntimeError, match="launch failed"):
        cli.run_all(run, [0], False)
    assert len(killed) == 2 and waits == [30, None]


def test_launcher_reports_active_progress_and_upload_failure(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import summarize

    run = setup(tmp_path)
    settings.save_json(run / "failed/old", {})
    polls = iter([None, 0, 0, 0, 0])
    process = types.SimpleNamespace(returncode=0, poll=lambda: next(polls, 0))
    publisher = types.SimpleNamespace(poll=lambda: 1, wait=lambda: 1, returncode=1)
    processes = iter([process, publisher])
    settings.save_json(run / "status.json", {"status": "partial"})

    def launch(*a, **k):
        assert settings.read_json(run / "status.json")["status"] == "running"
        return next(processes)

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.time, "monotonic", lambda: 1000)
    monkeypatch.setattr(
        summarize,
        "summarize",
        lambda r: {"status": "complete", "failed_jobs": [], "methods": {}},
    )
    assert cli.run_all(run, [0], True) == 4
    assert not (run / "failed/old").exists()


def test_verify_frozen_input_hashes(tmp_path):
    from research.qwen36_a100_seven_eval import prepare

    target = tmp_path / "target"
    target.mkdir()
    config = target / "config.json"
    config.write_text("{}")
    manifest = {
        "target": {
            "path": str(target),
            "files": {"config.json": settings.sha256(config)},
        },
        "exports": {},
    }
    prepare.verify_inputs(tmp_path, manifest)
    config.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="target snapshot"):
        prepare.verify_inputs(tmp_path, manifest)
    manifest["target"]["files"] = {}
    manifest["exports"] = {"dflash": {"path": str(target), "config_sha256": "wrong"}}
    with pytest.raises(ValueError, match="draft export"):
        prepare.verify_inputs(tmp_path, manifest)


def test_hooked_method_and_shutdown_failure(tmp_path, monkeypatch):
    run = setup(tmp_path, ("dfly",))
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, "ar", "reference")) == 0
    hooked = []
    original = stage.importlib.import_module
    monkeypatch.setattr(
        stage.importlib,
        "import_module",
        lambda name: (
            types.SimpleNamespace(install=lambda: hooked.append(name))
            if name == "fake_adapter"
            else original(name)
        ),
    )
    real_settings = stage.inference_settings()
    real_settings["methods"]["dfly"]["install_before_engine"] = "fake_adapter.install"
    monkeypatch.setattr(stage, "inference_settings", lambda: real_settings)
    assert stage.run_stage(args(run, "dfly", "sd_r0")) == 0 and hooked == [
        "fake_adapter"
    ]
    monkeypatch.setattr(
        FakeLLM,
        "shutdown",
        lambda self: (_ for _ in ()).throw(RuntimeError("shutdown")),
    )
    assert stage.run_stage(args(run, "dfly", "sd_r1")) == 1
    assert not list((run / "raw/mmspec/dfly").glob("*sd_r1*.jsonl"))
