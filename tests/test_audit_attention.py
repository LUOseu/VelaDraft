import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('audit_attention', Path(__file__).parents[1] / 'scripts/npu/audit_attention.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_attention_baselines_match_chunked_output_and_gradients():
    from vlm_dflash.dflash.mask import DFlashMaskSpec
    mask = DFlashMaskSpec(torch.tensor([[2, 5]]), torch.ones(1, 2, dtype=torch.bool), 10, 4, query_chunk_size=3)
    torch.manual_seed(7)
    inputs = [torch.randn(1, h, s, 4) for h, s in [(4, 8), (2, 18), (2, 18)]]
    reference = None
    for mode in audit.MODES:
        tensors = [x.clone().requires_grad_() for x in inputs]
        output, info = audit.attention_for(mode)(*tensors, mask, dropout_p=0., scale=.5)
        output.square().sum().backward()
        actual = [output] + [x.grad for x in tensors]
        if reference is None:
            reference = actual
        else:
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        assert info['activation_checkpointing'] == (mode != 'dense')


def test_shape_protocol_is_fixed():
    assert audit.SHAPES == ((1024, 64), (4096, 256), (4096, 512))


def test_full_update_uses_fp32_master_gradients_and_moments():
    from types import SimpleNamespace
    module = SimpleNamespace(draft_model=torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16))
    module.draft_model.weight.data.fill_(1)
    masters, optimizer, scheduler = audit.create_audit_optimizer(module, 8)
    assert all(p.dtype == torch.float32 for p in masters.values())
    module.draft_model.weight.grad = torch.full_like(module.draft_model.weight, .25)
    seen = {}
    def observed_step(*args, **kwargs):
        seen['gradient'] = masters['weight'].grad.clone()
    optimizer.register_step_pre_hook(observed_step)
    from scripts.npu.audit_common import optimizer_step
    optimizer_step(module, masters, optimizer, scheduler)
    torch.testing.assert_close(seen['gradient'], torch.full((2, 2), .25))
    assert optimizer.state[masters['weight']]['exp_avg'].dtype == torch.float32
    assert module.draft_model.weight.grad is None
    torch.testing.assert_close(module.draft_model.weight, masters['weight'].to(torch.bfloat16))
