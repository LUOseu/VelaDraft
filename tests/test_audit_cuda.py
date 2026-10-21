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


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
def test_shared_forward_respects_configured_compute_dtype(tmp_path,monkeypatch,dtype):
    import json
    import sys
    import torch
    from pathlib import Path
    from scripts.npu.audit_common import build_module
    from scripts.npu.audit_cuda import shared_forward
    from scripts.npu.seven_method_smoke import _make_target,_contract,_config_for,_SyntheticFeatures,VOCAB,HIDDEN
    target=_make_target(tmp_path/'target')
    contract=_contract(target)
    config=_config_for('dfly',1)
    config['train']['draft_dtype']=dtype
    bundle={'contract':contract,'manifest':{'capture_layer_ids':[0,1],'hidden_size':HIDDEN},
            'records':[_SyntheticFeatures()[0]],'embed_weight':torch.randn(VOCAB,HIDDEN),'lm_head_weight':None}
    module,_=build_module(bundle,config,'cpu')
    import scripts.npu.audit_common as common
    original_make_batch=common.make_batch
    batch_dtypes=[]
    def record_batch_dtype(*args,**kwargs):
        batch=original_make_batch(*args,**kwargs)
        batch_dtypes.append(batch['hidden_states'].dtype)
        return batch
    monkeypatch.setattr(common,'make_batch',record_batch_dtype)
    monkeypatch.setitem(sys.modules,'audit_common',common)
    torch.save({'model':module.draft_model.state_dict()},tmp_path/'checkpoint.pt')
    torch.save(bundle,tmp_path/'bundle.pt')
    (tmp_path/'config.json').write_text(json.dumps(config))
    output=tmp_path/'forward.pt'
    monkeypatch.setattr(sys,'argv',['audit_cuda.py','shared-forward','--repo',str(Path(__file__).resolve().parents[1]),
        '--bundle',str(tmp_path/'bundle.pt'),'--config',str(tmp_path/'config.json'),
        '--checkpoint',str(tmp_path/'checkpoint.pt'),'--output',str(output),'--device','cpu'])
    shared_forward()
    record=torch.load(output,weights_only=False)
    expected=getattr(torch,dtype)
    assert batch_dtypes==[expected]
    assert record['draft_hidden'].dtype==expected
    assert record['post_correction_hidden'].dtype==expected
    assert record['student_logits'].dtype==expected


def test_export_verifier_explicitly_selects_production_parity_backend():
    from scripts.npu.audit_cuda import configure_export_verifier
    from vlm_dflash.dflash.draft_model import set_verify_sdpa_backend_switch,verify_sdpa_backend_switch_enabled
    original=verify_sdpa_backend_switch_enabled()
    set_verify_sdpa_backend_switch(True)
    try:
        algorithm=configure_export_verifier()
        assert algorithm.name=='dfly'
        assert verify_sdpa_backend_switch_enabled() is False
    finally:
        set_verify_sdpa_backend_switch(original)
