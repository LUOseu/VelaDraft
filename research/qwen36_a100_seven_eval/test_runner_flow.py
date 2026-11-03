"""CPU tests of the engine stage and the CLI orchestration with fake vLLM / torch / GPU lease."""
import json
from pathlib import Path
import sys
import types

import pytest

from research.qwen36_a100_seven_eval import cli, lane, publish, settings, stage
from research.qwen36_a100_seven_eval.test_runner import CONFIG, FakeApi, manifest_for, sessions_two_topics, write_shard


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shutdowns = 0
        FakeLLM.instances.append(self)
        self.llm_engine = types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=self.shutdown))

    def shutdown(self):
        self.shutdowns += 1

    def collective_rpc(self, name, timeout=None):
        if name == 'backend_metadata':
            spec = self.kwargs.get('speculative_config')
            return [dict(speculative_tokens=spec['num_speculative_tokens'] if spec else None)]
        return [dict(method='meta')]


class FakeLease:
    fail_validate = False

    def __init__(self, index, path):
        self.path = Path(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def validate(self, timed_windows=None):
        if FakeLease.fail_validate:
            raise ValueError('foreign app inside a timed window')
        return dict(status='complete', windows=len(timed_windows or []))


def install_fakes(monkeypatch, *, tokens=3, finish='stop', prompt_ids=(1, 2)):
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(get_device_name=lambda i: 'NVIDIA A100-SXM4-80GB')
    pil = types.ModuleType('PIL')
    pil.Image = types.SimpleNamespace()
    transformers = types.ModuleType('transformers')
    transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: 'processor')
    vllm = types.ModuleType('vllm')
    vllm.LLM = FakeLLM
    vllm.SamplingParams = lambda **kw: kw
    for name, module in dict(torch=torch, PIL=pil, transformers=transformers, vllm=vllm).items():
        monkeypatch.setitem(sys.modules, name, module)
    import vlm_dflash.eval.native_resources as resources
    import vlm_dflash.eval.native_runtime as runtime
    monkeypatch.setattr(resources, 'GPULease', FakeLease)
    calls = []

    def request(llm, processor, params, torch_, image, session, turn, reference, plan):
        calls.append((session['id'], turn))
        n = params.get('max_tokens') if params.get('ignore_eos') else tokens
        return dict(id=session['id'], turn=turn, tokens=[7] * n, prompt_token_ids=list(prompt_ids), text='t',
                    finish_reason='length' if params.get('ignore_eos') else finish, wall_seconds=0.5,
                    prompt='p', messages=[])

    monkeypatch.setattr(runtime, 'request', request)
    monkeypatch.setattr(runtime, 'metrics', lambda llm: dict(rounds=1.0, proposed=3.0, accepted=2.0))
    monkeypatch.setattr(runtime, 'metric_delta', lambda a, b: dict(status='complete', rounds=1, proposed=3, accepted=2))
    monkeypatch.setattr(stage.importlib.metadata, 'version', lambda name: '0.32.0')
    monkeypatch.setenv('LANE_GPU', '0')
    FakeLease.fail_validate = False
    FakeLLM.instances.clear()
    return calls


def setup(tmp_path, methods=('dflash',)):
    run = tmp_path / 'run'
    settings.save_json(run / 'manifest.json', manifest_for(tmp_path, methods))
    settings.save_json(run / 'sessions.json', sessions_two_topics())
    return run


def args(run, method, stage_name, protocol='natural'):
    return types.SimpleNamespace(run_dir=str(run), method=method, stage=stage_name, protocol=protocol)


def test_reference_then_sd_stage_writes_shards_and_resumes(tmp_path, monkeypatch):
    run = setup(tmp_path)
    calls = install_fakes(monkeypatch)
    assert stage.run_stage(args(run, 'ar', 'reference')) == 0
    ref = stage.load_reference(run, sessions_two_topics())
    assert ref[('s2', 1)]['prompt'] == 'p'  # reference rows keep the full prompt
    assert stage.run_stage(args(run, 'dflash', 'sd_r0')) == 0
    llm = FakeLLM.instances[-1]
    assert llm.kwargs['speculative_config']['num_speculative_tokens'] == 3 and llm.shutdowns == 1
    shard = stage.shard_path(run, 'dflash', 'natural', 'sd_r0', settings.CATEGORIES[5])
    row = json.loads(shard.read_text().splitlines()[0])
    assert 'prompt' not in row and row['output_tokens'] == 3 and row['arm'] == 'sd'
    meta = settings.read_json(run / 'stages' / 'dflash' / 'natural__sd_r0.json')
    assert meta['status'] == 'complete' and meta['resources']['status'] == 'complete'
    assert meta['gpu_name'].startswith('NVIDIA A100')
    before = len(calls)
    assert stage.run_stage(args(run, 'dflash', 'sd_r0')) == 0 and len(calls) == before  # nothing left


def test_stats_and_fixed512_stages(tmp_path, monkeypatch):
    run = setup(tmp_path, ('dflash2',))
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, 'ar', 'reference')) == 0
    assert stage.run_stage(args(run, 'dflash2', 'stats')) == 0
    row = json.loads(stage.shard_path(run, 'dflash2', 'natural', 'stats', settings.CATEGORIES[0]).read_text())
    assert row['official_counters']['accepted'] == 2
    assert FakeLLM.instances[-1].kwargs['disable_log_stats'] is False
    assert stage.run_stage(args(run, 'dflash2', 'ar_r0', 'fixed512')) == 0
    row = json.loads(stage.shard_path(run, 'dflash2', 'fixed512', 'ar_r0', settings.CATEGORIES[0]).read_text())
    assert row['output_tokens'] == 512 and row['finish_reason'] == 'length'


def test_prompt_drift_and_wrong_k_fail_the_stage(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, 'ar', 'reference')) == 0
    install_fakes(monkeypatch, prompt_ids=(9,))
    assert stage.run_stage(args(run, 'dflash', 'sd_r0')) == 1
    meta = settings.read_json(run / 'stages' / 'dflash' / 'natural__sd_r0.json')
    assert 'prompt ids differ' in meta['error'] and FakeLLM.instances[-1].shutdowns == 1
    install_fakes(monkeypatch)
    monkeypatch.setattr(FakeLLM, 'collective_rpc', lambda self, name, timeout=None: [dict(speculative_tokens=2)])
    assert stage.run_stage(args(run, 'dflash', 'sd_r1')) == 1
    assert 'K=2' in settings.read_json(run / 'stages' / 'dflash' / 'natural__sd_r1.json')['error']


def test_contaminated_resources_withdraw_shards(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, 'ar', 'reference')) == 0
    FakeLease.fail_validate = True
    assert stage.run_stage(args(run, 'dflash', 'ar_r0')) == 1
    assert stage.missing_topics(run, sessions_two_topics(), 'dflash', 'natural', 'ar_r0')
    assert list((run / 'raw' / 'mmspec' / 'dflash').glob('*.invalid*'))
    meta = settings.read_json(run / 'stages' / 'dflash' / 'natural__ar_r0.json')
    assert meta['status'] == 'invalid_resources'


def test_stage_guards(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    with pytest.raises(ValueError, match='ar_default'):
        stage.run_stage(args(run, 'ar_default', 'sd_r0'))
    monkeypatch.setattr(stage.importlib.metadata, 'version', lambda name: '0.31.0')
    assert stage.run_stage(args(run, 'ar', 'reference')) == 1
    assert stage.main(['--run-dir', str(run), '--method', 'ar', '--stage', 'reference']) == 1


def test_stock_ar_runs_without_audit_worker(tmp_path, monkeypatch):
    run = setup(tmp_path)
    install_fakes(monkeypatch)
    assert stage.run_stage(args(run, 'ar', 'reference')) == 0

    def no_worker(self, name, timeout=None):
        raise AttributeError(name)
    monkeypatch.setattr(FakeLLM, 'collective_rpc', no_worker)
    assert stage.run_stage(args(run, 'ar_default', 'ar_r0')) == 0
    assert 'unavailable' in settings.read_json(run / 'stages' / 'ar_default' / 'natural__ar_r0.json')['worker']


def test_lane_run_stage_invokes_hook_and_module(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(lane.subprocess, 'run', lambda cmd, **kw: seen.append((cmd, kw.get('env', {}))) or
                        types.SimpleNamespace(returncode=0))
    log = tmp_path / 'lane.log'
    assert lane.run_stage(tmp_path, 3, 'dflash', 'natural', 'sd_r0', log, 'free_gpu.sh') == 0
    assert seen[0][0] == ['free_gpu.sh', '3']
    cmd, env = seen[1]
    assert cmd[-6:] == ['--method', 'dflash', '--protocol', 'natural', '--stage', 'sd_r0']
    assert env['CUDA_VISIBLE_DEVICES'] == '3' and env['VELADRAFT_GPU_OWNERSHIP'] == 'count'


# ---------------------------------------------------------------- cli orchestration
def fake_freeze(monkeypatch, tmp_path):
    from research.qwen36_a100_seven_eval import prepare
    monkeypatch.setattr(cli.data_mod, 'fetch', lambda repo, commit, dest: dest)
    monkeypatch.setattr(cli.data_mod, 'build_sessions', lambda clone, suite: (
        sessions_two_topics(), dict(formal=suite == 'mmspec-full', sessions=2)))
    monkeypatch.setattr(prepare, 'target_identity', lambda path, rev, cache: dict(path=str(path), revision=rev,
                                                                               stop_token_ids=[1]))
    monkeypatch.setattr(prepare, 'freeze_checkpoints', lambda methods, config, cache: {
        m: dict(path=f'/e/{m}') for m in methods})
    monkeypatch.setattr(cli, 'gpus_info', lambda idx: [dict(index=str(i), name='NVIDIA A100-SXM4-80GB',
                                                            memory_used_mib='0') for i in idx])
    monkeypatch.setattr(cli, 'git', lambda *a: '' if a[0] == 'status' else 'c0ffee' * 7)
    monkeypatch.setattr(cli.importlib.metadata, 'version', lambda name: '0.32.0')


class FakeProc:
    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self):
        return 0


def test_create_run_freezes_manifest_and_links_latest(tmp_path, monkeypatch):
    monkeypatch.setenv('Q36_A100_ROOT', str(tmp_path / 'root'))
    fake_freeze(monkeypatch, tmp_path)
    config = settings.apply_overrides(settings.load_config(CONFIG), methods=['dflash'])
    run = cli.create_run(config, 'mmspec-smoke')
    manifest = settings.read_json(run / 'manifest.json')
    assert manifest['methods'] == ['dflash'] and manifest['code']['commit'].startswith('c0ffee')
    assert cli.latest_run(tmp_path / 'root', 'mmspec-smoke') == run.resolve()
    monkeypatch.setattr(cli, 'git', lambda *a: ' M file' if a[0] == 'status' else 'c0ffee' * 7)
    with pytest.raises(ValueError, match='uncommitted'):
        cli.create_run(config, 'mmspec-smoke')


def test_main_runs_lanes_publisher_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv('Q36_A100_ROOT', str(tmp_path / 'root'))
    fake_freeze(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, 'preflight', lambda config, suite, publish_: dict(ok=True, checks=[
        dict(name='x', ok=True, detail='', action=None)]))
    launched = []
    monkeypatch.setattr(cli.subprocess, 'Popen', lambda cmd, **kw: launched.append((cmd, kw.get('env'))) or FakeProc(cmd))
    code = cli.main(['--config', str(CONFIG), '--suite', 'mmspec-smoke', '--methods', 'dflash', '--gpus', '0,1',
                     '--publish'])
    assert code == 3  # nothing ran, so the run is partial
    lanes = [c for c, _ in launched if 'research.qwen36_a100_seven_eval.lane' in c]
    pubs = [(c, e) for c, e in launched if 'research.qwen36_a100_seven_eval.publish' in c]
    assert len(lanes) == 2 and len(pubs) == 1 and 'HF_HUB_OFFLINE' not in pubs[0][1]
    run = cli.latest_run(tmp_path / 'root', 'mmspec-smoke')
    status = settings.read_json(run / 'status.json')
    assert status['status'] == 'partial' and status['jobs'] == 4
    # Resume: same commit continues the run; a changed commit or method list is refused.
    (run / 'claims' / 'dflash__natural').mkdir(parents=True)
    assert cli.main(['--config', str(CONFIG), '--suite', 'mmspec-smoke', '--methods', 'dflash', '--resume']) == 3
    assert not (run / 'claims' / 'dflash__natural').exists()
    with pytest.raises(SystemExit, match='methods'):
        cli.main(['--config', str(CONFIG), '--suite', 'mmspec-smoke', '--methods', 'dfly', '--resume'])
    monkeypatch.setattr(cli, 'git', lambda *a: '' if a[0] == 'status' else 'beef' * 10)
    with pytest.raises(SystemExit, match='commit changed'):
        cli.main(['--config', str(CONFIG), '--suite', 'mmspec-smoke', '--methods', 'dflash', '--resume'])


def test_main_stops_on_failed_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('Q36_A100_ROOT', str(tmp_path / 'root'))
    monkeypatch.setattr(cli, 'preflight', lambda config, suite, publish_: dict(ok=False, checks=[
        dict(name='hf_write', ok=False, detail='no token', action='log in')]))
    assert cli.main(['--config', str(CONFIG)]) == 2
    assert '-> log in' in capsys.readouterr().out


def test_publish_main_once(tmp_path, monkeypatch, capsys):
    run = setup(tmp_path)
    settings.save_json(run / 'status.json', dict(status='complete', suite='mmspec-smoke'))
    write_shard(run, 'dflash', 'natural', 'ar_r0', settings.CATEGORIES[0], [dict(id='s1')])
    api = FakeApi()
    api.fail_once = False
    real = publish.Publisher

    class Bound(real):
        def __init__(self, r, repo, path):
            super().__init__(r, repo, path, api=api)
    monkeypatch.setattr(publish, 'Publisher', Bound)
    assert publish.main(['--run-dir', str(run), '--repo', 'o/r', '--path', 'eval/x', '--once']) == 0
    assert '"uploaded"' in capsys.readouterr().out
    monkeypatch.setattr(publish.time, 'sleep', lambda s: None)
    assert publish.loop(run, 'o/r', 'eval/x', 0) == 0  # complete status and nothing left: exits


# ---------------------------------------------------------------- prepare
def test_hash_cache_and_target_identity(tmp_path):
    from research.qwen36_a100_seven_eval import prepare
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'config.json').write_text('{}')
    (target / 'generation_config.json').write_text(json.dumps({'eos_token_id': [5, 6]}))
    cache = tmp_path / 'cache.json'
    ident = prepare.target_identity(target, 'rev', cache)
    assert ident['stop_token_ids'] == [5, 6] and set(ident['files']) == {'config.json', 'generation_config.json'}
    assert len(settings.read_json(cache)) == 2
    (target / 'generation_config.json').write_text(json.dumps({'eos_token_id': 9}))
    assert prepare.target_identity(target, 'rev', cache)['stop_token_ids'] == [9]
    with pytest.raises(FileNotFoundError):
        prepare.target_identity(tmp_path / 'missing', 'rev', cache)


def test_checkpoint_local_hit_download_and_mismatch(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import prepare
    local = tmp_path / 'local.pt'
    local.write_bytes(b'weights')
    entry = dict(repo='o/m', revision='abc', size=7, sha256=settings.sha256(local))
    cache = tmp_path / 'c.json'
    assert prepare.fetch_checkpoint('dflash', entry, [local], tmp_path, cache) == local
    downloaded = tmp_path / 'dl.pt'
    downloaded.write_bytes(b'weights')
    hub = types.ModuleType('huggingface_hub')
    hub.hf_hub_download = lambda repo, name, revision, local_dir: str(downloaded)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    assert prepare.fetch_checkpoint('dflash', entry, [tmp_path / 'absent.pt'], tmp_path, cache) == downloaded
    downloaded.write_bytes(b'other!!')
    with pytest.raises(ValueError, match='SHA-256 differs'):
        prepare.fetch_checkpoint('dflash', entry, [], tmp_path, cache)


def raw_export(root, method, arch_config):
    raw = root / 'raw' / method
    raw.mkdir(parents=True)
    (raw / 'config.json').write_text(json.dumps(arch_config))
    (raw / 'export_receipt.json').write_text('{}')
    (raw / 'model.safetensors').write_bytes(b'w')
    return raw


def test_export_dispatch_per_method(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import prepare
    ran = []
    monkeypatch.setattr(prepare.subprocess, 'run', lambda cmd, **kw: ran.append(cmd))
    raw_export(tmp_path, 'dflash', {'architectures': ['DFlashDraftModel']})
    final = prepare.export('dflash', tmp_path / 'ckpt.pt', tmp_path)
    assert (final / 'model.safetensors').samefile(tmp_path / 'raw/dflash/model.safetensors') and not ran
    assert prepare.export('dflash', tmp_path / 'ckpt.pt', tmp_path) == final  # idempotent
    seen = {}
    for method, module in (('jetspec', 'prepare_native'), ('dspark', 'prepare_dspark'), ('dfly', 'prepare_custom')):
        fake = types.ModuleType(module)
        fake.prepare = lambda *a, m=method: seen.setdefault(m, a) and Path(a[-1]).mkdir(parents=True)
        monkeypatch.setitem(sys.modules, f'research.qwen36_seven_20261102.{module}', fake)
        prepare.export(method, tmp_path / 'ckpt.pt', tmp_path / method)
        assert ran[-1][-2:] == [str(tmp_path / 'ckpt.pt'), str(tmp_path / method / 'raw' / method)]
    assert seen['jetspec'][0] == 'jetspec' and len(seen['dspark']) == 2 and seen['dfly'][0] == 'dfly'


def test_freeze_checkpoints_records_pins(tmp_path, monkeypatch):
    from research.qwen36_a100_seven_eval import prepare
    final = tmp_path / 'final'
    final.mkdir()
    (final / 'config.json').write_text('{}')
    (final / 'model.safetensors').write_bytes(b'w')
    monkeypatch.setattr(prepare, 'fetch_checkpoint', lambda m, e, c, d, cache: tmp_path / 'ck.pt')
    monkeypatch.setattr(prepare, 'export', lambda m, ck, root: final)
    config = dict(checkpoints=dict(work_dir=str(tmp_path / 'w'), local_root=str(tmp_path / 'l')))
    out = prepare.freeze_checkpoints(['dflash', 'dfly'], config, tmp_path / 'c.json')
    pinned = settings.checkpoint_manifest()
    assert out['dfly']['revision'] == pinned['dfly']['revision'] and out['dflash']['path'] == str(final)
