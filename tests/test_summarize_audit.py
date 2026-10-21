"""Fail-closed generation from immutable small audit artifacts."""
import copy
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_completed_artifacts_generate_scoped_profiles():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    ids = {r['id'] for r in evidence['profile_records']}
    assert 'dfly-compaction-npu' in ids
    assert 'dfly-stage-boundary-npu' in ids
    assert 'dfly-npu-export-cuda' not in ids
    assert 'slower' in report and 'FAILED' in report
    assert all(r['source_revision_kind'] == 'base_commit_plus_dirty_source_hashes' for r in evidence['profile_records'])


def test_failed_parity_cannot_be_promoted():
    from scripts.npu.summarize_audit import require_pass
    with pytest.raises(ValueError, match='parity'):
        require_pass({'status': 'FAILED'}, 'parity')


def test_failed_cross_backend_forward_never_promotes_combined_export():
    from scripts.npu.summarize_audit import shared_forward_failure
    row = shared_forward_failure({'status': 'FAILED'}, 'raw/continuation/cuda_shared_forward_comparison.json')
    assert row['id'] == 'dfly-npu-export-cuda'
    assert row['status'] == 'FAILED'
    assert 'artifact' not in row
    with pytest.raises(ValueError):
        shared_forward_failure({'status': 'PASS'}, 'raw.json')


@pytest.mark.parametrize('case', ['geometry', 'actual_chunks', 'pilots', 'compaction_switch'])
def test_contradictory_raw_claims_fail_closed(monkeypatch, case):
    from scripts.npu import summarize_audit as audit
    original = audit.read
    def mutated_read(path):
        value = original(path)
        if Path(path).name == 'shape-4096-512-chunked_checkpoint.json' and case != 'compaction_switch':
            value = copy.deepcopy(value)
            if case == 'geometry':
                value['geometry']['num_hidden_layers'] = 2
            elif case == 'pilots':
                value['warmup_updates'] = []
            else:
                for update in value['measured_updates']:
                    update['attention']['effective_backends'] = ['dense']
                    window = update['attention']['measurement_window']
                    window['effective_backends'] = ['dense']
                    window['chunks'] = window['completed_attention_calls']
        if Path(path).name == 'pair0_on.json' and case == 'compaction_switch':
            value = copy.deepcopy(value)
            value['compaction'] = False
        return value
    monkeypatch.setattr(audit, 'read', mutated_read)
    with pytest.raises(ValueError):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')


@pytest.mark.parametrize('method', ['dspark', 'dfly'])
def test_recipe_geometry_evidence_cannot_contradict_formal_claim(monkeypatch, method):
    from scripts.npu import summarize_audit as audit
    original = audit.read
    def changed(path):
        value = original(path)
        if method == 'dspark' and Path(path).name == 'dspark.json':
            value = copy.deepcopy(value)
            value['geometry']['hidden_size'] = 1024
        if method == 'dfly' and Path(path).name == 'continuous_checkpoint_metadata.json':
            value = copy.deepcopy(value)
            value['model_tensor_shapes']['layers.1.mlp.up_proj.weight'] = [4096, 2048]
        return value
    monkeypatch.setattr(audit, 'read', changed)
    with pytest.raises(ValueError, match='geometry'):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')


@pytest.mark.parametrize('method', ['dspark', 'dfly'])
def test_observed_recipe_coefficients_cannot_contradict_claim(monkeypatch, method):
    from scripts.npu import summarize_audit as audit
    original = audit.read
    def changed(path):
        value = original(path)
        if method == 'dspark' and Path(path).name == 'dspark.json':
            value = copy.deepcopy(value)
            value['continuous_observation']['observations'][0]['stage']['confidence_coef'] = 0
        if method == 'dfly' and Path(path).parent.name == 'resume-5' and Path(path).name == 'results.json':
            value = copy.deepcopy(value)
            value['results'][0]['observation']['observations'][1]['stage']['e2e_tv_coef'] = 0
        return value
    monkeypatch.setattr(audit, 'read', changed)
    with pytest.raises(ValueError, match='recipe'):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')


def test_portable_usability_is_separate_from_failed_numerical_parity():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    rows = {row['id']:row for row in evidence['profile_records']}
    row = rows['dfly-portable-usability-cuda']
    assert row['mode'] == 'inference'
    assert row['shape']['accepted_tokens'] == 1931
    assert row['samples'] == 64
    assert row['tau_progress_tokens_per_round'] == pytest.approx(2936/997)
    assert row['A_accepted_tokens_per_round'] == pytest.approx(1931/997)
    assert 'dfly-npu-export-cuda' not in rows
    assert any(row['id']=='dfly-npu-export-cuda' and row['status']=='FAILED' for row in evidence['failed_profile_records'])
    assert 'INVALID_PROTOCOL' in report


@pytest.mark.parametrize('field', ['backend', 'counts', 'exactness'])
def test_portable_usability_rejects_protocol_or_metric_drift(monkeypatch, field):
    from scripts.npu import summarize_audit as audit
    original = audit.read
    def changed(path):
        value = original(path)
        if Path(path).name == 'C_export64.json':
            value = copy.deepcopy(value)
            if field == 'backend':
                value['verify_sdpa_backend_switch_enabled'] = True
            elif field == 'counts':
                value['results']['portable']['accepted_tokens'] += 1
            else:
                value['results']['portable']['token_exact'] = False
        return value
    monkeypatch.setattr(audit, 'read', changed)
    with pytest.raises(ValueError, match='C '):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')


def test_fp32_diagnostic_requires_actual_fp32_metadata(monkeypatch):
    from scripts.npu import summarize_audit as audit
    original = audit.read
    def changed(path):
        value = original(path)
        if Path(path).name == 'C_shared_forward_fp32_comparison.json':
            value = copy.deepcopy(value)
            value['metadata']['npu']['student_logits_dtype'] = 'torch.bfloat16'
        return value
    monkeypatch.setattr(audit, 'read', changed)
    with pytest.raises(ValueError, match='FP32 actual dtype'):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')


def test_cuda_capacity_qualification_keeps_bounded_oom_and_original_plan():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    assert [row['allocator_cap_gib'] for row in evidence['cuda_capacity_qualifications']] == [22,25]
    assert all(row['oom'] for row in evidence['cuda_capacity_qualifications'])
    assert '不是整张A6000' in report


def test_cuda_capacity_qualification_rejects_changed_npu_plan(monkeypatch):
    from scripts.npu import summarize_audit as audit
    original = audit.read
    def changed(path):
        value = original(path)
        if Path(path).name == 'plan_verification.json':
            value = copy.deepcopy(value)
            value['worst_plan']['context'] = 512
        return value
    monkeypatch.setattr(audit, 'read', changed)
    with pytest.raises(ValueError, match='CUDA A frozen plan'):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')


def test_npu_compaction_reports_non_bitwise_cases_explicitly():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    assert evidence['npu_compaction_parity']['strict_bitwise_cases'] == 4
    assert evidence['npu_compaction_parity']['bf16_envelope_cases'] == 8
    assert '严格逐位一致4/8' in report


def test_compaction_stage_costs_use_measured_projection_counters():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    rows = [r for r in evidence['compaction_phase_work'] if r['backend']=='npu']
    assert len(rows)==2
    off,on=rows
    assert off['updates']==on['updates']==30
    assert off['student_projected_rows_forward'] > on['student_projected_rows_forward']
    assert off['teacher_projected_rows_forward'] > on['teacher_projected_rows_forward']
    assert off['forward_ms']>0 and off['backward_ms']>0 and off['update_ms']>0


def test_cuda_full_run_requires_complete_legs_and_successful_guard(tmp_path):
    import json
    import shutil
    from scripts.npu.summarize_audit import completed_cuda_compaction, read
    source = ROOT / 'docs/npu/audit-2026-10-21/raw/compaction-v2'
    # Fixture reuses already checked JSON structure; it is not CUDA evidence.
    for path in source.glob('*.json'):
        shutil.copy2(path,tmp_path/path.name)
    (tmp_path/'guard.json').write_text(json.dumps({'returncode':1}))
    with pytest.raises(ValueError, match='guard'):
        completed_cuda_compaction(tmp_path,source)
    (tmp_path/'guard.json').write_text(json.dumps({'returncode':0}))
    summary = completed_cuda_compaction(tmp_path,source)
    assert len(summary['pairs'])==3
    assert summary['bf16_envelope_cases']==8
    leg = read(tmp_path/'pair2_on.json')
    leg['steps']=leg['steps'][:-1]
    (tmp_path/'pair2_on.json').write_text(json.dumps(leg))
    with pytest.raises(ValueError, match='window'):
        completed_cuda_compaction(tmp_path,source)


def test_completed_cuda_compaction_is_promoted_separately_from_cap_failures():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    rows={r['id']:r for r in evidence['profile_records']}
    assert rows['dfly-compaction-cuda']['status']=='PASS'
    assert rows['dfly-compaction-cuda']['allocator_environment']['PYTORCH_ALLOC_CONF']=='expandable_segments:True'
    assert evidence['cuda_compaction']['strict_bitwise_cases']==6
    assert len([r for r in evidence['compaction_phase_work'] if r['backend']=='cuda'])==2
    assert len(evidence['cuda_capacity_qualifications'])==2
    assert 'CUDA严格逐位一致6/8' in report


def test_cuda_compaction_failed_guard_cannot_promote(monkeypatch):
    from scripts.npu import summarize_audit as audit
    original=audit.read
    def changed(path):
        value=original(path)
        if Path(path).parent.name=='gpu2_expand' and Path(path).name=='guard.json':
            value=copy.deepcopy(value); value['returncode']=1
        return value
    monkeypatch.setattr(audit,'read',changed)
    with pytest.raises(ValueError,match='guard'):
        audit.generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
