"""Opt-in PyTorch fusion for the unquantized TP1 Qwen shared-expert gate.

Only official torch operations/compiler are used. The BF16 cast following the
dot product, sigmoid and product is preserved by emulate_precision_casts.
This does not assert numerical equivalence for arbitrary inputs.
"""
import os

_installed = False


def install():
    global _installed
    if _installed or os.environ.get('Q36_SHARED_GATE_FUSION') != '1':
        return
    import torch
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    def gate(x, weight, out):
        logits = (x.float() * weight.float()).sum(-1, keepdim=True).to(x.dtype)
        return torch.sigmoid(logits) * out

    compiled = torch.compile(gate, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False,
                                      'emulate_precision_casts': True})
    original = Qwen2MoeMLP.forward

    def forward(self, x):
        expert_gate = self.expert_gate
        supported = (
            expert_gate is not None
            and getattr(self.down_proj, 'tp_size', None) == 1
            and type(expert_gate.quant_method).__name__ == 'UnquantizedLinearMethod'
            and expert_gate.bias is None
            and tuple(expert_gate.weight.shape) == (1, 2048)
            and expert_gate.weight.dtype == torch.bfloat16
            and x.dtype == torch.bfloat16 and x.is_cuda
            and x.ndim == 2 and 0 < x.shape[0] <= 8
        )
        if not supported:
            return original(self, x)
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return compiled(x, expert_gate.weight, out)

    forward._q36_shared_gate_fusion = True
    Qwen2MoeMLP.forward = forward
    _installed = True


def metadata(model):
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    return {
        'requested': os.environ.get('Q36_SHARED_GATE_FUSION') == '1',
        'installed': bool(getattr(Qwen2MoeMLP.forward, '_q36_shared_gate_fusion', False)),
        'eligible_modules': [name for name, mod in model.named_modules()
                             if isinstance(mod, Qwen2MoeMLP)
                             and mod.expert_gate is not None
                             and getattr(mod.down_proj, 'tp_size', None) == 1
                             and tuple(mod.expert_gate.weight.shape) == (1, 2048)],
        'scope': 'BF16 unquantized TP1, shared gate only, query rows 1..8',
    }
