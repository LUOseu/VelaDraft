"""Synthetic BF16 training smoke; does not establish real-data or 8B readiness."""
import pytest
torch = pytest.importorskip('torch')
from tests.test_trainer_smoke import _tiny_module


def test_tiny_dflash_bf16_cuda_update():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA with BF16 required')
    torch.manual_seed(17)
    device = torch.device('cuda:0')
    module = _tiny_module(seed=17).to(device=device, dtype=torch.bfloat16)
    # The real trainer deliberately does not register frozen target modules;
    # unlike the draft, they must be placed explicitly.
    module.embed_tokens.to(device=device, dtype=torch.bfloat16)
    module.lm_head.to(device=device, dtype=torch.bfloat16)
    ids = torch.arange(48, device=device).remainder(200).add(10).unsqueeze(0)
    mask = torch.ones_like(ids, dtype=torch.float32)
    mask[:, :8] = 0
    with torch.no_grad():
        base = module.embed_tokens(ids).detach()
        hidden = torch.cat([base, torch.tanh(base)], dim=-1)
    optimizer = torch.optim.AdamW((p for p in module.parameters() if p.requires_grad), lr=1e-3)
    before = next(p.detach().clone() for p in module.parameters() if p.requires_grad)
    result = module(input_ids=ids, hidden_states=hidden, loss_mask=mask)
    assert torch.isfinite(result.loss)
    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    after = next(p.detach() for p in module.parameters() if p.requires_grad)
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
