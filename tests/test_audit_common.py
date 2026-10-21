import importlib.util
from pathlib import Path
import torch

def test_bundle_build_batch_and_master_step(tmp_path):
    from scripts.npu.audit_common import build_module, make_batch, create_optimizer, optimizer_step
    from scripts.npu.seven_method_smoke import _make_target, _contract, _config_for, _SyntheticFeatures
    contract=_contract(_make_target(tmp_path/'target'))
    bundle={'contract':contract,'manifest':{'capture_layer_ids':[0,1],'hidden_size':32},'embed_weight':torch.randn(128,32),'lm_head_weight':None,'records':[_SyntheticFeatures()[i] for i in range(2)]}
    cfg=_config_for('dfly',10)
    module,_=build_module(bundle,cfg,'cpu')
    batch=make_batch(bundle,[0,1],'cpu',torch.float32,[0,1])
    assert torch.equal(batch['teacher_hidden'][0],bundle['records'][0]['final_hidden'])
    before={n:p.detach().clone() for n,p in module.draft_model.named_parameters()}
    masters,opt,sched=create_optimizer(module,cfg,10)
    module(**batch).loss.backward()
    optimizer_step(module,masters,opt,sched,1.0)
    assert all(p.dtype==torch.float32 for p in masters.values())
    assert any(not torch.equal(p,before[n]) for n,p in module.draft_model.named_parameters())
