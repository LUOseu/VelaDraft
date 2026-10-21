"""Audit tensor probes must index large real-model matrices without rounding."""
import pytest
from scripts.npu.audit_cuda import probe_indices


@pytest.mark.parametrize('numel',[1,31,32,33,22544384,311164928])
def test_probe_indices_use_exact_integer_bounds(numel):
    index=probe_indices(numel,'cpu')
    assert len(index)==min(32,numel)
    assert index[0]==0
    assert index[-1]==numel-1
    assert bool((index>=0).all() and (index<numel).all())


@pytest.mark.parametrize('change',['none','student_logits','input_sha256'])
def test_shared_forward_gate_catches_head_only_or_identity_mismatch(tmp_path,monkeypatch,change):
    import copy
    import json
    import sys
    import torch
    from scripts.npu.audit_cuda import compare_shared_forward
    record={key:torch.ones(2,3,dtype=torch.bfloat16) for key in ['draft_hidden','post_correction_hidden','student_logits']}
    record.update(weights_sha256='a',input_sha256='b',dtype='torch.bfloat16',student_logits_dtype='torch.bfloat16',
                  source_sha256={'trainer':'c'},anchor_table=(torch.tensor([[1]]),torch.tensor([[True]])),loss=.25)
    other=copy.deepcopy(record)
    if change=='student_logits':other['student_logits'][0,0]=1.5
    elif change=='input_sha256':other['input_sha256']='different'
    left,right,out=tmp_path/'npu.pt',tmp_path/'cuda.pt',tmp_path/'report.json'
    torch.save(other,left);torch.save(record,right)
    monkeypatch.setattr(sys,'argv',['audit_cuda.py','compare-forward','--npu',str(left),'--cuda',str(right),'--output',str(out)])
    assert compare_shared_forward()==(0 if change=='none' else 1)
    result=json.loads(out.read_text())
    if change=='student_logits':
        assert result['checks']['draft_hidden']['passed']
        assert result['checks']['post_correction_hidden']['passed']
        assert result['checks']['student_logits']['violating_elements']==1
        assert result['status']=='FAILED'
