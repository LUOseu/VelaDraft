"""Opt-in native-components DSpark worker, keeping the original trained layout."""
import hashlib
from pathlib import Path
from research.qwen36_seven_20261102 import dspark_bridge
dspark_bridge.install()
from research.qwen36_seven_20261102.worker import NativeSevenWorker


class DSparkWorker(NativeSevenWorker):
    def method_metadata(self):
        from vllm.model_executor.models import qwen3_dspark
        from vllm.v1.worker.gpu.spec_decode.dspark import speculator
        value=super().method_metadata();spec=self.model_runner.speculator
        model=spec.model.model
        value.update(anchor_slot_bridge=True,native_input_anchor_sampling=spec.sample_from_anchor,
                     markov_shapes=[list(model.markov_head.markov_w1.weight.shape),
                                    list(model.markov_head.markov_w2.weight.shape)],
                     confidence_head_loaded=model.confidence_head is not None,
                     confidence_head_used=spec.use_confidence_head,
                     draft_sample_method=spec.speculative_config.draft_sample_method)
        for module in (dspark_bridge,qwen3_dspark,speculator):
            path=Path(module.__file__);value['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return value
