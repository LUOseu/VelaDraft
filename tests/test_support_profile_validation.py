"""Evidence-bound profile claims reject missing, stale and substituted records."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from scripts.npu.validate_support_matrix import validate_matrix


def _fixture(tmp_path):
    row = dict(id='dfly-stage2-npu', method='dfly', objective_profile='upstream_two_stage',
               stage='e2e_tv', backend='npu', mode='offline', status='PASS', dtype='bfloat16',
               model_geometry=dict(num_hidden_layers=5,hidden_size=2048,intermediate_size=11008,
                                   num_attention_heads=16,num_key_value_heads=2,head_dim=128,block_size=8),
               shape=dict(context=825,anchors=256,query_rows=2048,chunks=64,
                          valid_supervision_rows=256,padding_fraction=0.5),
               data_kind='natural_images', samples=8,updates=3,verification_mode='not_applicable',
               source_revision='a'*40)
    artifact=tmp_path/'evidence.json'
    artifact.write_text(json.dumps({'profile_records':[row]}))
    row=copy.deepcopy(row)
    row['artifact']={'path':'evidence.json','sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'record_id':row['id']}
    matrix=tmp_path/'support_matrix.json'
    matrix.write_text(json.dumps({'audit_profiles':[row]}))
    return matrix,row,artifact


def _write(matrix,row):
    matrix.write_text(json.dumps({'audit_profiles':[row]}))


def test_valid_bound_profile(tmp_path):
    matrix,_,_=_fixture(tmp_path)
    assert validate_matrix(matrix)==[]


@pytest.mark.parametrize('field',['dtype','model_geometry','shape','data_kind','samples','updates','verification_mode','source_revision','objective_profile','artifact'])
def test_pass_missing_context_rejected(tmp_path,field):
    matrix,row,_=_fixture(tmp_path);del row[field];_write(matrix,row)
    assert validate_matrix(matrix)


@pytest.mark.parametrize('field,value',[('source_revision','b'*40),('stage','cold_start_lk_dpace'),('objective_profile','default'),('backend','cuda'),('status','FAILED')])
def test_substituted_or_stale_record_rejected_even_with_updated_digest(tmp_path,field,value):
    matrix,row,artifact=_fixture(tmp_path)
    evidence=json.loads(artifact.read_text());evidence['profile_records'][0][field]=value
    artifact.write_text(json.dumps(evidence))
    row['artifact']['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();_write(matrix,row)
    assert validate_matrix(matrix)


def test_changed_artifact_digest_rejected(tmp_path):
    matrix,_,artifact=_fixture(tmp_path);artifact.write_text('{}')
    assert validate_matrix(matrix)


def test_missing_or_duplicate_profile_records_rejected(tmp_path):
    for count in [0,2]:
        matrix,row,artifact=_fixture(tmp_path)
        record=json.loads(artifact.read_text())['profile_records'][0]
        artifact.write_text(json.dumps({'profile_records':[record]*count}))
        row['artifact']['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();_write(matrix,row)
        assert validate_matrix(matrix)


def test_expected_revision_rejects_consistent_but_stale_evidence(tmp_path):
    matrix,_,_=_fixture(tmp_path)
    assert validate_matrix(matrix,expected_revision='b'*40)


@pytest.mark.parametrize('status',['PENDING','FAILED'])
def test_unverified_rows_need_reason_and_do_not_need_evidence(tmp_path,status):
    matrix,row,_=_fixture(tmp_path)
    row={k:row[k] for k in ['id','method','objective_profile','stage','backend','mode']}
    row['status']=status;_write(matrix,row)
    assert validate_matrix(matrix)
    row['reason']='No device available';_write(matrix,row)
    assert validate_matrix(matrix)==[]


def test_unknown_success_spelling_cannot_bypass_gate(tmp_path):
    matrix,row,_=_fixture(tmp_path);row['status']='PASSED';_write(matrix,row)
    assert validate_matrix(matrix)


def test_checked_in_matrix_profiles_are_valid():
    path=Path(__file__).resolve().parents[1]/'docs/npu/support_matrix.json'
    assert validate_matrix(path)==[]


@pytest.mark.parametrize('key,value',[('samples',True),('updates',0),('source_revision','HEAD'),
                                     ('model_geometry',{}),('shape',{})])
def test_invalid_context_rejected_even_when_artifact_agrees(tmp_path,key,value):
    matrix,row,artifact=_fixture(tmp_path)
    record=json.loads(artifact.read_text())['profile_records'][0]
    row[key]=record[key]=value
    artifact.write_text(json.dumps({'profile_records':[record]}))
    row['artifact']['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();_write(matrix,row)
    assert validate_matrix(matrix)


def test_optional_dirty_tree_digest_cannot_disagree(tmp_path):
    matrix,row,_=_fixture(tmp_path)
    row['source_tree_sha256']='b'*64;_write(matrix,row)
    assert validate_matrix(matrix)


def test_duplicate_matrix_id_rejected(tmp_path):
    matrix,row,_=_fixture(tmp_path)
    matrix.write_text(json.dumps({'audit_profiles':[row,row]}))
    assert validate_matrix(matrix)


def _inference_fixture(tmp_path, *, mode='inference', omit=None, accepted=1931):
    matrix, row, artifact = _fixture(tmp_path)
    row.update(mode=mode, updates=0 if mode=='inference' else 3,
               verification_mode='sequential_parity', samples=64,
               shape=dict(max_new_tokens=64, speculative_rounds=997, new_tokens=2936,
                          draft_tokens=6507, accepted_tokens=accepted))
    if omit:
        del row['shape'][omit]
    record = {k:v for k,v in row.items() if k!='artifact'}
    artifact.write_text(json.dumps({'profile_records':[record]}))
    row['artifact']['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    _write(matrix,row)
    return matrix


def test_inference_profile_uses_actual_decoding_counts(tmp_path):
    assert validate_matrix(_inference_fixture(tmp_path)) == []


@pytest.mark.parametrize('field', ['max_new_tokens','speculative_rounds','new_tokens','draft_tokens','accepted_tokens'])
def test_inference_profile_cannot_omit_decoding_counts(tmp_path, field):
    assert validate_matrix(_inference_fixture(tmp_path, omit=field))


def test_training_profile_cannot_reuse_inference_shape_schema(tmp_path):
    assert validate_matrix(_inference_fixture(tmp_path, mode='offline'))


@pytest.mark.parametrize('accepted', [2937,6508,-1,True])
def test_inference_acceptance_counts_must_be_consistent(tmp_path, accepted):
    assert validate_matrix(_inference_fixture(tmp_path, accepted=accepted))
