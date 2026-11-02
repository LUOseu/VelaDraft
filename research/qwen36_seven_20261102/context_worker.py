import hashlib
from pathlib import Path
from research.qwen36_seven_20261102 import context_bridge
context_bridge.install()
from research.qwen36_seven_20261102.worker import NativeSevenWorker


class ContextWorker(NativeSevenWorker):
    def method_metadata(self):
        value=super().method_metadata();spec=self.model_runner.speculator;model=spec.model.model
        value.update(context_method=model.context_method,
            context_buffer_width=spec.hidden_states.shape[-1],
            fusion_shape=list(model.layer_fusion_weights.shape),has_fc=hasattr(model,'fc'))
        if model.context_method=='dflare':
            value['dedicated_shapes']=[list(getattr(layer.self_attn,kind+'_proj_target').weight.shape)
                                       for layer in model.layers for kind in ('k','v')]
        else:
            value['fc_shape']=list(model.fc.weight.shape)
            value['correction_shapes']=[list(getattr(model.hidden_correction,name).weight.shape)
                                       for name in ('gate_proj','up_proj','down_proj')]
        for module in (context_bridge,context_bridge.flare_reference,context_bridge.fly_reference):
            path=Path(module.__file__);value['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return value
