"""Audit public method implementations on the shared optimized target."""
import hashlib
from pathlib import Path
from research.qwen36_nextopt_20261102.worker import JointWorker


class NativeSevenWorker(JointWorker):
    def method_metadata(self):
        import inspect
        runner = self.model_runner
        spec = runner.speculator
        model = spec.model
        result = dict(speculator=type(spec).__module__+'.'+type(spec).__name__,
                      model=type(model).__module__+'.'+type(model).__name__,
                      queries=spec.num_query_per_req,
                      draft_causal=model.get_draft_attn_causal(),
                      graph_mode=str(spec.query_cudagraph_manager.cudagraph_mode),
                      graphs=[str(d) for d in spec.query_cudagraph_manager.graphs])
        if type(spec).__name__ == 'DFlash2Speculator':
            from vllm.model_executor.models import qwen3_dflash2
            from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator
            result.update(top_k=spec.top_k,
                conv_blocks=[(x.attention_conv.block_size, x.mlp_conv.block_size) for x in model.model.layers],
                predecessor_shape=list(model.model.candidate_selector.predecessor_codebook.shape),
                successor_shape=list(model.model.candidate_selector.successor_codebook.shape))
            sources=[qwen3_dflash2.__file__, speculator.__file__]
        else:
            sources=[inspect.getfile(type(model)), inspect.getfile(type(spec))]
        result['sources']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources}
        return result
