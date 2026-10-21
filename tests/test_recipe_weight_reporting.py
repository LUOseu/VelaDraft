import torch
import pytest

@pytest.mark.parametrize('values',[[.1,.2,.3,.4], [1/7]*1000, [.125,-.25,.375,-.5]])
def test_effective_weight_sum_ignores_interleaved_zero_padding(values):
    from vlm_dflash.dflash.trainer_core import _sum_nonzero_weights
    a=torch.tensor(values,dtype=torch.float32)
    padded=torch.zeros(len(values),3);padded[:,1]=a
    assert torch.equal(_sum_nonzero_weights(a),_sum_nonzero_weights(padded))

@pytest.mark.parametrize('compact',[False,True])
def test_reporting_fix_does_not_change_recipe_loss_or_gradients(monkeypatch,compact):
    from vlm_dflash.dflash import trainer_core
    from tests.test_recipe_row_compaction import _tiny_module,_batch,_run
    corrected=_run(_tiny_module(stage=1,recipe_row_compaction=compact),_batch())
    monkeypatch.setattr(trainer_core,'_sum_nonzero_weights',lambda x:x.sum())
    previous=_run(_tiny_module(stage=1,recipe_row_compaction=compact),_batch())
    assert corrected['loss']==previous['loss']
    assert corrected['valid_tokens']==previous['valid_tokens']
    assert all(torch.equal(corrected['grads'][k],previous['grads'][k]) for k in corrected['grads'])
