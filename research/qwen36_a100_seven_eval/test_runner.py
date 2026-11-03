"""CPU tests for the seven-method MMSpec runner: config, data scope, job plan, resume, summary, publisher."""

import json
import subprocess
import types
from pathlib import Path

import pytest

from research.qwen36_a100_seven_eval import (
    cli,
    data,
    lane,
    publish,
    settings,
    stage,
    summarize,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs/eval/qwen36_a100_seven.yaml"


# ---------------------------------------------------------------- settings
def test_stage_order_alternates_arms():
    assert settings.stage_names(3) == [
        "ar_r0",
        "sd_r0",
        "sd_r1",
        "ar_r1",
        "ar_r2",
        "sd_r2",
    ]
    with pytest.raises(ValueError):
        settings.stage_names(0)


def test_config_expands_environment_and_freezes_k(monkeypatch, tmp_path):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path / "root"))
    config = settings.load_config(CONFIG)
    assert config["run"]["root"] == str(tmp_path / "root")
    assert (
        config["protocol"]["draft_length"] == 3 and config["engine"]["vllm"] == "0.32.0"
    )
    with pytest.raises(ValueError, match="K=3"):
        settings.apply_overrides(config, draft_length=4)
    with pytest.raises(ValueError, match="unknown methods"):
        settings.apply_overrides(config, methods=["medusa"])
    assert settings.apply_overrides(config, methods=["dflash"])["methods"] == ["dflash"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG.read_text().replace("draft_length: 3", "draft_length: 4"))
    with pytest.raises(ValueError, match="K=3"):
        settings.load_config(bad)


def test_expand_requires_undefaulted_variables(monkeypatch):
    monkeypatch.delenv("Q36_UNSET_FOR_TEST", raising=False)
    with pytest.raises(ValueError, match="Q36_UNSET_FOR_TEST"):
        settings.expand("${Q36_UNSET_FOR_TEST}")
    assert settings.expand({"a": ["${Q36_UNSET_FOR_TEST:-x}", 1]}) == {"a": ["x", 1]}


# ---------------------------------------------------------------- data
def mmspec_rows():
    rows = []
    turns = [1] * 530 + [2] * 18 + [3] * 51 + [4] * 1
    for i, n in enumerate(turns):
        topic = settings.CATEGORIES[i // 100]
        rows.append(
            {
                "id": f"mmspec_{i:03d}",
                "image": f"{i}.jpg",
                "turns": [f"q{i}-{t}" for t in range(n)],
                "topic": topic,
                "category": "default",
            }
        )
    return rows


def test_scope_matches_paper_and_rejects_drift():
    rows = mmspec_rows()
    assert data.verify_scope(rows)["turns"] == 723
    with pytest.raises(ValueError, match="600 sessions|599 sessions"):
        data.verify_scope(rows[:-1])
    broken = json.loads(json.dumps(rows))
    broken[0]["turns"] = ["  "]
    with pytest.raises(ValueError, match="empty turn"):
        data.verify_scope(broken)


def test_smoke_takes_two_per_topic_with_multi_turn_history():
    smoke = data.smoke_rows(mmspec_rows())
    assert len(smoke) == 12
    assert all(
        len(r["turns"]) >= 2 for r in smoke if r["topic"] == "multi-turn conversation"
    )


def make_clone(tmp_path):
    repo = tmp_path / "MMSpec"
    images = repo / "dataset/MMSpec/test/images"
    images.mkdir(parents=True)
    rows = mmspec_rows()
    for r in rows:
        (images / r["image"]).write_bytes(r["id"].encode())
    (repo / "dataset/MMSpec/test/mmspec.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
    ):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    return repo


def test_build_sessions_full_and_smoke(tmp_path):
    repo = make_clone(tmp_path)
    sessions, manifest = data.build_sessions(repo, "mmspec-full")
    assert len(sessions) == 600 and manifest["turns"] == 723 and manifest["formal"]
    assert sessions[0]["image_sha256"] == settings.sha256(Path(sessions[0]["image"]))
    smoke, m2 = data.build_sessions(repo, "mmspec-smoke")
    assert len(smoke) == 12 and not m2["formal"]
    with pytest.raises(ValueError):
        data.build_sessions(repo, "testmini")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert data.fetch("unused", head, repo) == repo


# ---------------------------------------------------------------- stage helpers
def manifest_for(tmp_path, methods=("dflash",)):
    return {
        "run_id": "r1",
        "suite": "mmspec-smoke",
        "methods": list(methods),
        "data": {"formal": False},
        "target": {"path": "/t", "stop_token_ids": [1, 2]},
        "exports": {m: {"path": f"/e/{m}"} for m in methods},
        "protocol": {
            "protocols": ["natural", "fixed512"],
            "draft_length": 3,
            "repeats": 3,
            "max_new_tokens": 1024,
            "fixed_tokens": 512,
            "max_model_len": 8192,
            "min_pixels": 1,
            "max_pixels": 2,
            "seed": 43,
            "warmup_sessions_per_topic": 1,
            "acceptance_stats": True,
            "ar_default": True,
            "system_prompt": "s",
            "chat_template_kwargs": {},
        },
        "engine": {
            "gpu_memory_utilization": 0.85,
            "max_num_batched_tokens": 4096,
            "gdn_decode_kernel": "cuda",
            "moe_policy": "grouped",
            "tuned_config_folder": "",
        },
        "site": {},
        "versions": {"vllm": "0.32.0"},
        "publish": {"repo": "o/r", "path": "eval/x"},
    }


def test_engine_kwargs_per_arm():
    m = manifest_for(None, ("dspark", "dfly"))
    sd = stage.engine_kwargs(m, "dspark", "sd")
    assert (
        sd["speculative_config"]["method"] == "dflash"
        and sd["speculative_config"]["num_speculative_tokens"] == 3
    )
    assert sd["compilation_config"]["cudagraph_capture_sizes"] == [4]
    assert (
        stage.engine_kwargs(m, "dfly", "sd")["speculative_config"]["method"] == "dflash"
    )
    ar = stage.engine_kwargs(m, "dfly", "ar")
    assert "speculative_config" not in ar and ar["attention_config"] == {
        "backend": "FLASH_ATTN"
    }
    stock = stage.engine_kwargs(m, "ar_default", "ar")
    assert "attention_config" not in stock and "worker_extension_cls" not in stock
    assert (
        stage.engine_kwargs(m, "dfly", "sd", stats=True)["disable_log_stats"] is False
    )


def test_environment_per_arm():
    m = manifest_for(None, ("dfly",))
    sd = stage.environment_for(m, "dfly", "sd", "natural")
    assert sd["set"]["Q36_SWEEP_K"] == "3" and sd["set"]["Q36_MOE_POLICY"] == "grouped"
    assert "VLLM_TUNED_CONFIG_FOLDER" in sd["unset"]
    m["engine"]["tuned_config_folder"] = "/moe"
    assert (
        stage.environment_for(m, "dfly", "sd", "natural")["set"][
            "VLLM_TUNED_CONFIG_FOLDER"
        ]
        == "/moe"
    )
    assert (
        stage.environment_for(m, "dfly", "ar", "natural")["set"]["Q36_V032_ARM"] == "ar"
    )
    assert (
        "Q36_MOE_POLICY"
        in stage.environment_for(m, "ar_default", "ar", "natural")["unset"]
    )


def test_sampling_protocols():
    made = []
    fake = lambda **kw: made.append(kw) or kw
    m = manifest_for(None)
    assert stage.sampling(m, "natural", fake)["stop_token_ids"] == [1, 2]
    fixed = stage.sampling(m, "fixed512", fake)
    assert (
        fixed["ignore_eos"]
        and fixed["max_tokens"] == 512
        and "stop_token_ids" not in fixed
    )
    with pytest.raises(ValueError):
        stage.sampling(m, "other", fake)


def test_compact_keeps_audit_hashes():
    row = {
        "id": "a",
        "turn": 0,
        "tokens": [1, 2],
        "prompt_token_ids": [5, 6, 7],
        "prompt": "p",
        "messages": [],
        "wall_seconds": 1.0,
    }
    out = stage.compact(row)
    assert (
        "prompt" not in out and out["prompt_tokens"] == 3 and out["output_tokens"] == 2
    )
    assert len(out["prompt_token_ids_sha256"]) == 64


# ---------------------------------------------------------------- lane / resume
def sessions_two_topics():
    return [
        {"id": "s1", "questions": ["a"], "topic": settings.CATEGORIES[0]},
        {"id": "s2", "questions": ["b", "c"], "topic": settings.CATEGORIES[5]},
    ]


def write_shard(run, owner, protocol, stage_name, topic, rows):
    path = stage.shard_path(run, owner, protocol, stage_name, topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake_runner(run, sessions, *, fail=()):
    calls = []

    def run_stage(run_, gpu, method, protocol, stage_name, log, hook):
        calls.append((method, protocol, stage_name))
        if (method, stage_name) in fail:
            return 1
        proto = "natural" if stage_name in ("reference", "stats") else protocol
        for topic in {s["topic"] for s in sessions}:
            rows = []
            for s in [s for s in sessions if s["topic"] == topic]:
                for t in range(len(s["questions"])):
                    wall = 2.0 if stage_name.startswith("ar") else 1.0
                    rows.append(
                        {
                            "id": s["id"],
                            "turn": t,
                            "topic": topic,
                            "stage": stage_name,
                            "wall_seconds": wall,
                            "gpu_id": "gpu-a",
                            "output_tokens": 10,
                            "tokens": [1] * 10,
                            "finish_reason": "stop",
                            "text": "x",
                            "prompt_token_ids": [1],
                            "official_counters": {
                                "status": "complete",
                                "accepted": 6,
                                "proposed": 9,
                                "rounds": 3,
                            },
                        }
                    )
            write_shard(
                run_, stage.owner_of(method, stage_name), proto, stage_name, topic, rows
            )
        return 0

    return run_stage, calls


def setup_run(tmp_path, methods=("dflash",)):
    run = tmp_path / "run"
    run.mkdir()
    settings.save_json(run / "manifest.json", manifest_for(tmp_path, methods))
    settings.save_json(run / "sessions.json", sessions_two_topics())
    return run


def test_lane_runs_reference_then_jobs_and_resume_skips_done(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, calls = fake_runner(run, sessions_two_topics())
    monkeypatch.setattr(lane, "run_stage", runner)
    assert lane.lane(run, 0, poll=0) == 0
    assert calls[0] == ("ar", "natural", "reference")
    assert sorted(p.name for p in (run / "done").iterdir()) == sorted(
        [
            "dflash__natural",
            "dflash__fixed512",
            "ar_default__natural",
            "ar_default__fixed512",
        ]
    )
    assert ("dflash", "natural", "stats") in calls and (
        "dflash",
        "fixed512",
        "stats",
    ) not in calls
    assert not any(c[0] == "ar_default" and c[2].startswith("sd") for c in calls)
    calls.clear()
    assert lane.lane(run, 1, poll=0) == 0 and calls == []


def test_lane_marks_failed_stage_and_partial_resume(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, calls = fake_runner(run, sessions_two_topics(), fail={("dflash", "sd_r1")})
    monkeypatch.setattr(lane, "run_stage", runner)
    lane.lane(run, 0, poll=0)
    failed = settings.read_json(run / "failed" / "dflash__natural")
    assert failed["failed_stage"] == "sd_r1"
    assert calls.count(("dflash", "natural", "sd_r1")) == lane.STAGE_ATTEMPTS
    sessions = sessions_two_topics()
    assert lane.stage_done(run, sessions, "dflash", "natural", "sd_r0")
    (run / "failed" / "dflash__natural").unlink()
    assert stage.missing_topics(run, sessions, "dflash", "natural", "sd_r1") == [
        settings.CATEGORIES[0],
        settings.CATEGORIES[5],
    ]


def test_reference_failure_stops_lane(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, _ = fake_runner(run, sessions_two_topics(), fail={("ar", "reference")})
    monkeypatch.setattr(lane, "run_stage", runner)
    assert lane.lane(run, 0, poll=0) == 2 and (run / "failed" / "_reference").exists()
    assert lane.lane(run, 1, poll=0) == 2


def test_load_reference_requires_all_topics(tmp_path):
    sessions = sessions_two_topics()
    with pytest.raises(FileNotFoundError):
        stage.load_reference(tmp_path, sessions)
    for topic in (settings.CATEGORIES[0], settings.CATEGORIES[5]):
        write_shard(
            tmp_path,
            "_reference",
            "natural",
            "reference",
            topic,
            [{"id": topic, "turn": 0}],
        )
    assert len(stage.load_reference(tmp_path, sessions)) == 2


# ---------------------------------------------------------------- summary
def test_summary_recomputes_from_raw(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, _ = fake_runner(run, sessions_two_topics())
    monkeypatch.setattr(lane, "run_stage", runner)
    lane.lane(run, 0, poll=0)
    result = summarize.summarize(run)
    m = result["methods"]["dflash__natural"]
    assert m["status"] == "complete" and m["pooled"]["speedup_e2e"] == pytest.approx(
        2.0
    )
    assert (
        m["pooled"]["per_token_ratio"] == pytest.approx(2.0)
        and m["pooled"]["token_exact_rate"] == 1.0
    )
    assert m["pooled"]["speedup_e2e_ci95"] == [pytest.approx(2.0), pytest.approx(2.0)]
    assert (
        m["acceptance"]["accepted_per_round"] == 2
        and m["acceptance"]["tokens_per_round_with_bonus_upper_bound"] == 3
    )
    assert m["quality"]["status"] == "pending"
    assert "speedup_vs_stock_ar_pooled" not in m
    assert result["status"] == "complete"
    summarize.write_csv(result, tmp_path / "s.csv")
    assert "dflash" in (tmp_path / "s.csv").read_text()


def test_bootstrap_is_seeded_and_requires_pairs():
    ar = {"a": [2.0, 2.0], "b": [4.0]}
    sd = {"a": [1.0, 1.0], "b": [1.0]}
    assert summarize.bootstrap(ar, sd) == summarize.bootstrap(ar, sd)
    with pytest.raises(ValueError):
        summarize.bootstrap(ar, {"a": [1.0]})


def test_acceptance_unavailable_without_counters():
    assert (
        summarize.acceptance([{"official_counters": None}])["status"] == "unavailable"
    )


# ---------------------------------------------------------------- publisher
class FakeApi:
    def __init__(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        self.head = "h0"
        self.files = {}
        self.fail_once = True

    def repo_info(self, repo, revision=None):
        return types.SimpleNamespace(sha=self.head)

    def hf_hub_download(self, repo, path, revision=None):
        from huggingface_hub.errors import EntryNotFoundError

        if path not in self.files:
            raise EntryNotFoundError(path)
        local = Path(self.directory.name) / "download"
        local.write_bytes(self.files[path])
        return str(local)

    def create_commit(self, repo, operations, parent_commit, commit_message):
        if self.fail_once:
            self.fail_once = False
            self.head = "h1"
            raise RuntimeError("parent moved")
        assert parent_commit == self.head
        for op in operations:
            source = op.path_or_fileobj
            payload = Path(source).read_bytes() if isinstance(source, str) else source
            self.files[op.path_in_repo] = payload
        self.head = "h2"
        return types.SimpleNamespace(oid="h2")

    def get_paths_info(self, repo, paths, revision=None):
        return [
            types.SimpleNamespace(path=p, size=len(self.files[p]), lfs=None)
            for p in paths
            if p in self.files
        ]


def test_publisher_retries_and_protects_raw_shards(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    settings.save_json(
        run / "status.json", {"status": "running", "suite": "mmspec-smoke"}
    )
    write_shard(
        run, "dflash", "natural", "ar_r0", settings.CATEGORIES[0], [{"id": "s1"}]
    )
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    api = FakeApi()
    pub = publish.Publisher(run, "o/r", "eval/x", api=api)
    result = pub.cycle()
    assert result["uploaded"] >= 3 and result["attempts"] == 2
    assert (
        any(p.endswith(".jsonl") for p in api.files)
        and "eval/x/INDEX.json" in api.files
    )
    assert all(p.startswith("eval/x/") for p in api.files)
    assert pub.cycle() == {"uploaded": 0}
    shard = stage.shard_path(run, "dflash", "natural", "ar_r0", settings.CATEGORIES[0])
    shard.write_text('{"id":"changed"}\n')
    with pytest.raises(ValueError, match="immutable"):
        pub.changed()
    reloaded = publish.Publisher(run, "o/r", "eval/x", api=api)
    assert reloaded.state["uploaded"]


def test_write_access_check():
    api = types.SimpleNamespace(
        whoami=lambda: {
            "name": "o",
            "orgs": [],
            "auth": {"accessToken": {"role": "write"}},
        },
        repo_info=lambda repo: types.SimpleNamespace(sha="x"),
    )
    assert publish.check_write_access("o/r", api=api)["ok"]
    api.whoami = lambda: {
        "name": "z",
        "orgs": [],
        "auth": {"accessToken": {"role": "read"}},
    }
    assert not publish.check_write_access("o/r", api=api)["ok"]


# ---------------------------------------------------------------- cli
def test_cli_preflight_reports_actions(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "gpus_info",
        lambda idx: [
            {"index": "0", "name": "NVIDIA A800-SXM4-80GB", "memory_used_mib": "0"}
        ],
    )
    code = cli.main(["--config", str(CONFIG), "--preflight", "--suite", "mmspec-smoke"])
    out = capsys.readouterr().out
    assert code == 2 and "[NEEDS ACTION] gpus" in out and "[NEEDS ACTION] target" in out


def test_cli_rejects_wrong_k(monkeypatch, tmp_path):
    monkeypatch.setenv("Q36_A100_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="K=3"):
        cli.main(["--config", str(CONFIG), "--preflight", "--draft-length", "2"])


def test_missing_matched_shards_are_not_complete(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, _ = fake_runner(run, sessions_two_topics())
    monkeypatch.setattr(lane, "run_stage", runner)
    lane.lane(run, 0, poll=0)
    for arm in ("ar_r2", "sd_r2"):
        stage.shard_path(run, "dflash", "natural", arm, settings.CATEGORIES[5]).unlink()
    result = summarize.summarize(run)
    assert result["status"] == "partial"
    assert result["methods"]["dflash__natural"]["missing_pairs"] == 2


def test_duplicate_and_cross_gpu_pairs_are_rejected(tmp_path):
    run = setup_run(tmp_path)
    row = {"id": "s1", "turn": 0, "stage": "ar_r0", "gpu_id": "a"}
    with pytest.raises(ValueError, match="duplicate"):
        summarize.keyed([row, row])
    write_shard(run, "dflash", "natural", "ar_r0", settings.CATEGORIES[0], [row])
    write_shard(
        run,
        "dflash",
        "natural",
        "sd_r0",
        settings.CATEGORIES[0],
        [dict(row, stage="sd_r0", gpu_id="b")],
    )
    with pytest.raises(ValueError, match="same physical GPU"):
        summarize.summarize(run)


def test_missing_stats_and_stock_ar_prevent_completion(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, _ = fake_runner(run, sessions_two_topics())
    monkeypatch.setattr(lane, "run_stage", runner)
    lane.lane(run, 0, poll=0)
    path = stage.shard_path(run, "dflash", "natural", "stats", settings.CATEGORIES[0])
    saved = path.read_bytes()
    path.unlink()
    assert summarize.summarize(run)["status"] == "partial"
    path.write_bytes(saved)
    stage.shard_path(
        run, "ar_default", "natural", "ar_r0", settings.CATEGORIES[0]
    ).unlink()
    assert summarize.summarize(run)["status"] == "partial"
    counters = {"status": "complete", "accepted": 1, "proposed": 3, "rounds": 1}
    assert (
        summarize.acceptance([{"official_counters": counters}, {}])["status"]
        == "partial"
    )


def test_dataset_dirty_and_wrong_revision_rejected(tmp_path):
    repo = make_clone(tmp_path)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    with pytest.raises(ValueError, match="pinned commit"):
        data.fetch("unused", "different", repo)
    (repo / "untracked").write_text("dirty")
    with pytest.raises(ValueError, match="local changes"):
        data.fetch("unused", head, repo)


def test_publication_redacts_site_details_and_preserves_remote_index(tmp_path):
    run = setup_run(tmp_path)
    manifest = settings.read_json(run / "manifest.json")
    manifest["site"] = {
        "environment": {"SECRET": "private-secret"},
        "pre_stage_hook": "/private/command",
    }
    manifest["gpus"] = [{"uuid": "private-uuid"}]
    settings.save_json(run / "manifest.json", manifest)
    settings.save_json(
        run / "stages/m/stage.json",
        {
            "status": "failed",
            "traceback": "private-secret",
            "environment": manifest["site"],
        },
    )
    api = FakeApi()
    api.fail_once = False
    api.files["eval/x/INDEX.json"] = json.dumps(
        {"runs": {"older": {"status": "complete"}}}
    ).encode()
    pub = publish.Publisher(run, "o/r", "eval/x", api=api)
    pub.cycle()
    combined = b"\n".join(api.files.values())
    assert (
        b"private-secret" not in combined
        and b"private-uuid" not in combined
        and b'/t"' not in combined
    )
    assert "older" in json.loads(api.files["eval/x/INDEX.json"])["runs"]
    with pytest.raises(ValueError, match="destination"):
        publish.Publisher(run, "o/other", "eval/x", api=api)
    with pytest.raises(ValueError, match="prefix"):
        publish.Publisher(run, "o/r", "eval/../outside", api=api)


def test_index_network_failure_never_overwrites_other_runs(tmp_path):
    run = setup_run(tmp_path)
    api = FakeApi()
    api.fail_once = False

    def outage(*a, **k):
        raise ConnectionError("offline")

    api.hf_hub_download = outage
    with pytest.raises(ConnectionError):
        publish.Publisher(run, "o/r", "eval/x", api=api, max_attempts=1).cycle()
    assert not api.files


def test_remote_content_verification_detects_same_size_corruption(tmp_path):
    run = setup_run(tmp_path)
    api = FakeApi()
    api.fail_once = False
    download = api.hf_hub_download

    def corrupted(repo, path, revision=None):
        local = Path(download(repo, path, revision))
        local.write_bytes(b"x" * local.stat().st_size)
        return local

    api.hf_hub_download = corrupted
    pub = publish.Publisher(run, "o/r", "eval/x", api=api, max_attempts=1)
    with pytest.raises(ValueError, match="remote content"):
        pub.cycle()
    assert not pub.state["uploaded"]


def test_partial_run_publisher_exits(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    settings.save_json(run / "status.json", {"status": "partial"})
    pub = types.SimpleNamespace(cycle=lambda: {"uploaded": 0}, changed=dict)
    monkeypatch.setattr(publish, "Publisher", lambda *a: pub)
    monkeypatch.setattr(
        publish.time, "sleep", lambda *a: pytest.fail("must exit on partial")
    )
    assert publish.loop(run, "o/r", "eval/x", 0) == 0


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.pop("target"),
        lambda c: c["protocol"].update(repeats=2),
        lambda c: c["protocol"].update(protocols=["natural"]),
        lambda c: c["protocol"].update(max_model_len=0),
        lambda c: c["protocol"].update(fixed_tokens=500),
        lambda c: c["protocol"].update(min_pixels=99999999),
    ],
)
def test_invalid_protocol_rejected(tmp_path, edit):
    import yaml

    config = settings.load_config(CONFIG)
    edit(config)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        settings.load_config(path)


def test_output_override_and_summary_cli(tmp_path, capsys):
    config = settings.apply_overrides(
        settings.load_config(CONFIG), output_root=tmp_path
    )
    assert config["run"]["root"] == str(tmp_path)
    run = setup_run(tmp_path)
    assert summarize.percentiles([]) is None
    assert summarize.main(["--run-dir", str(run)]) == 0
    assert "partial" in capsys.readouterr().out
    assert (run / "summary.csv").exists()


def test_lane_respects_bound_indices_and_live_claims(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    runner, _ = fake_runner(run, sessions_two_topics())
    monkeypatch.setattr(lane, "run_stage", runner)
    runner(run, 0, "ar", "natural", "reference", None, None)
    settings.save_json(run / "bindings/dflash__natural.json", {"index": 1})
    (run / "claims/dflash__fixed512").mkdir(parents=True)
    assert lane.main(["--run-dir", str(run), "--gpu", "0"]) == 0
    assert not (run / "done/dflash__natural").exists()
    assert not (run / "done/dflash__fixed512").exists()


def test_reference_waits_for_other_lane(tmp_path, monkeypatch):
    run = setup_run(tmp_path)
    sessions = sessions_two_topics()
    runner, _ = fake_runner(run, sessions)
    (run / "claims/_reference").mkdir(parents=True)
    monkeypatch.setattr(
        lane.time,
        "sleep",
        lambda _: runner(run, 0, "ar", "natural", "reference", None, None),
    )
    assert lane.ensure_reference(run, 1, sessions, tmp_path / "log", None, 0)


def test_fetch_local_clone_at_pinned_revision(tmp_path):
    repo = make_clone(tmp_path)
    old = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-qm",
            "next",
        ],
        check=True,
    )
    clone = data.fetch(str(repo), old, tmp_path / "clone")
    assert (
        subprocess.check_output(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True
        ).strip()
        == old
    )


def test_remote_raw_is_immutable_even_without_local_publish_state(tmp_path):
    run = setup_run(tmp_path)
    write_shard(
        run, "dflash", "natural", "ar_r0", settings.CATEGORIES[0], [{"id": "new"}]
    )
    api = FakeApi()
    api.fail_once = False
    rel = "raw/mmspec/dflash/natural__ar_r0__general_vqa.jsonl"
    api.files[f"eval/x/runs/r1/{rel}"] = b'{"id":"old"}\n'
    with pytest.raises(ValueError, match="remote immutable"):
        publish.Publisher(run, "o/r", "eval/x", api=api, max_attempts=1).cycle()
    assert api.files[f"eval/x/runs/r1/{rel}"] == b'{"id":"old"}\n'


def test_custom_export_uses_explicit_directories(tmp_path):
    from research.qwen36_seven_20261102.prepare_custom import prepare

    src = tmp_path / "source"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"dflash_config": {"spatial_rope_enabled": True}})
    )
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "export_receipt.json").write_text("{}")
    dst = tmp_path / "destination"
    prepare("domino", src, dst)
    assert (src / "model.safetensors").samefile(dst / "model.safetensors")
    assert (
        settings.read_json(dst / "config.json")["veladraft_custom_method"] == "domino"
    )
