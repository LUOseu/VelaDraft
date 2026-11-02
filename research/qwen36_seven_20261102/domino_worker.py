import hashlib
from pathlib import Path
from research.qwen36_seven_20261102 import domino_bridge
domino_bridge.install()
from research.qwen36_seven_20261102.worker import NativeSevenWorker


class DominoWorker(NativeSevenWorker):
    def method_metadata(self):
        value=super().method_metadata();spec=self.model_runner.speculator;model=spec.model
        value.update(native_input_anchor_sampling=spec.sample_from_anchor,
            head_shapes=[list(model.prefix_gru.weight_ih_l0.shape),list(model.prefix_gru.weight_hh_l0.shape),
                         list(model.embed_proj[0].weight.shape),list(model.embed_proj[2].weight.shape)],
            draft_sample_method=spec.speculative_config.draft_sample_method)
        for module in (domino_bridge,domino_bridge.reference):
            path=Path(module.__file__);value['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return value
