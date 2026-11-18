"""Replay the existing spatial-adapter context K/V operators, without new kernels."""
import os


def install():
    if os.environ.get('Q36_CONTEXT_KV_GRAPH') != '1':
        return
    import torch
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from research.qwen36_native_vllm.spatial_adapter import enabled, in_dummy_scope
    original = DFlashQwen3Model.precompute_and_store_context_kv
    if getattr(original, '_q36_context_graph', False):
        return

    def selected_cache(self, slots):
        values=[]
        for layer in self._attn_layers:
            cache=layer.kv_cache
            if cache.ndim != 5 or cache.shape[0] != 2:
                raise ValueError('Context graph audit requires the native FlashAttention K/V layout')
            physical=slots.clamp_min(0).long()
            values.append(cache[:,physical//cache.shape[2],physical%cache.shape[2]].clone())
        return values

    def forward(self, context_states, context_positions, context_slot_mapping=None):
        n=context_states.shape[0]
        if (not enabled(self.config) or in_dummy_scope() or context_slot_mapping is None
                or not 0 < n <= 8 or context_positions.shape != (3,n)
                or torch.cuda.is_current_stream_capturing()):
            return original(self,context_states,context_positions,context_slot_mapping)
        if not hasattr(self,'_q36_context_graphs'):
            self._q36_context_graphs={}
        # Run the original once to initialize its fused weight buffers if needed.
        if not hasattr(self,'_num_attn_layers'):
            original(self,context_states,context_positions,context_slot_mapping)
        pointers=tuple(layer.kv_cache.data_ptr() for layer in self._attn_layers)
        key=(tuple(context_states.shape),context_states.dtype,pointers)
        entry=self._q36_context_graphs.get(key)
        if entry is None:
            original(self,context_states,context_positions,context_slot_mapping)
            expected=selected_cache(self,context_slot_mapping)
            states=torch.empty_like(context_states);positions=torch.empty_like(context_positions);slots=torch.empty_like(context_slot_mapping)
            states.copy_(context_states);positions.copy_(context_positions);slots.copy_(context_slot_mapping)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):original(self,states,positions,slots)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):original(self,states,positions,slots)
            graph.replay()
            actual=selected_cache(self,slots)
            if not all(torch.equal(a,b) for a,b in zip(actual,expected)):
                raise ValueError('Captured context K/V differs from original adapter writes')
            entry=dict(graph=graph,states=states,positions=positions,slots=slots,replays=1,
                       oracle_checked=True,query_rows=n)
            self._q36_context_graphs[key]=entry
            return None
        entry['states'].copy_(context_states)
        entry['positions'].copy_(context_positions)
        entry['slots'].copy_(context_slot_mapping)
        entry['graph'].replay();entry['replays']+=1
        return None

    forward._q36_context_graph=True
    DFlashQwen3Model.precompute_and_store_context_kv=forward


def audit(model):
    entries=[]
    for name,mod in model.named_modules():
        for entry in getattr(mod,'_q36_context_graphs',{}).values():
            entries.append(dict(module=name,query_rows=entry['query_rows'],replays=entry['replays'],oracle_checked=entry['oracle_checked']))
    return entries
