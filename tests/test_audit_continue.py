import torch,json

def test_continuation_export_uses_actual_steps_and_fresh_optimizer(tmp_path):
    from scripts.npu.audit_continue import continue_and_export
    from scripts.npu.audit_common import build_module
    from scripts.npu.seven_method_smoke import _make_target,_contract,_config_for,_SyntheticFeatures
    bundle={'contract':_contract(_make_target(tmp_path/'target')),'manifest':{'capture_layer_ids':[0,1],'hidden_size':32},'embed_weight':torch.randn(128,32),'lm_head_weight':None,'records':[_SyntheticFeatures()[i] for i in range(8)]}
    bundle['contract']['mask']={'token_id':127}
    cfg=_config_for('dfly',10)
    model,_=build_module(bundle,cfg,'cpu')
    checkpoint=tmp_path/'source.pt';torch.save({'model':model.draft_model.state_dict()},checkpoint)
    result=continue_and_export(bundle,cfg,checkpoint,tmp_path/'run','cpu',steps=2)
    assert result['steps']==2 and result['initial_weight_sha256']!=result['final_weight_sha256']
    assert result['peak_accelerator_mem_gb'] is None
    assert (tmp_path/'run/export/model.safetensors').exists()
    assert json.loads((tmp_path/'run/train/train_summary.json').read_text())['observation']['observations'][0]['stage_index']==1


import pytest

@pytest.mark.parametrize('failure', ['gradient_norm', 'final_parameter'])
def test_continuation_rejects_nonfinite_final_update(tmp_path, monkeypatch, failure):
    from scripts.npu import audit_continue as audit
    from scripts.npu.audit_common import build_module
    from scripts.npu.seven_method_smoke import _make_target, _contract, _config_for, _SyntheticFeatures
    bundle = {'contract': _contract(_make_target(tmp_path / 'target')),
              'manifest': {'capture_layer_ids': [0, 1], 'hidden_size': 32},
              'embed_weight': torch.randn(128, 32), 'lm_head_weight': None,
              'records': [_SyntheticFeatures()[i] for i in range(8)]}
    bundle['contract']['mask'] = {'token_id': 127}
    cfg = _config_for('dfly', 10)
    model, _ = build_module(bundle, cfg, 'cpu')
    checkpoint = tmp_path / 'source.pt'
    torch.save({'model': model.draft_model.state_dict()}, checkpoint)
    real_step = audit.optimizer_step
    def poisoned_step(module, *args, **kwargs):
        norm = real_step(module, *args, **kwargs)
        if failure == 'gradient_norm':
            return float('nan')
        with torch.no_grad():
            next(module.draft_model.parameters()).view(-1)[0] = float('nan')
        return norm
    monkeypatch.setattr(audit, 'optimizer_step', poisoned_step)
    with pytest.raises(FloatingPointError, match='gradient norm' if failure == 'gradient_norm' else 'non-finite parameter'):
        audit.continue_and_export(bundle, cfg, checkpoint, tmp_path / 'run', 'cpu', steps=1)
    assert not (tmp_path / 'run' / 'train' / 'ckpt_final.pt').exists()
    assert not (tmp_path / 'run' / 'export').exists()
