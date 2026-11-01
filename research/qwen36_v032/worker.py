"""Version-specific audit extension; only the SD worker installs positions."""
import os
if os.environ.get('Q36_V032_ARM')=='sd':
    from research.qwen36_v032.spatial_adapter import install
    install()


class AuditWorker:
    def backend_metadata(self):
        import torch
        import vllm
        from research.qwen36_v032.spatial_adapter import source_hashes
        from vlm_dflash.eval.native_runtime import numeric_state
        runner=self.model_runner
        speculator=getattr(runner,'speculator',None)
        manager=getattr(speculator,'query_cudagraph_manager',None)
        context=self.vllm_config.compilation_config.static_forward_context
        layers=[dict(name=name,impl=type(layer.impl).__module__+'.'+type(layer.impl).__name__)
                for name,layer in sorted(context.items()) if hasattr(layer,'impl')]
        gdn=[dict(name=name,kernel=layer.gdn_decode_kernel,fused_spec_decode=bool(layer.enable_fused_gdn_spec_decode))
             for name,layer in sorted(context.items()) if hasattr(layer,'gdn_decode_kernel')]
        config=self.vllm_config.speculative_config
        return dict(vllm=str(vllm.__version__),torch=str(torch.__version__),runner=type(runner).__module__+'.'+type(runner).__name__,
                    use_v2=self.vllm_config.use_v2_model_runner,
                    numeric=numeric_state(),target_dtype=str(self.vllm_config.model_config.dtype),
                    target_quantization=self.vllm_config.model_config.quantization,
                    speculator=type(speculator).__module__+'.'+type(speculator).__name__ if speculator else None,
                    speculative_tokens=config.num_speculative_tokens if config else None,
                    target_noncausal=self.vllm_config.attention_config.use_non_causal,attention=layers,
                    gdn=gdn,
                    sources=source_hashes(),graph_config=str(self.vllm_config.compilation_config.cudagraph_mode),
                    draft_graph_manager=type(getattr(speculator,'query_cudagraph_manager',None)).__name__,
                    draft_graph_mode=str(getattr(manager,'cudagraph_mode',None)),
                    draft_graphs=[str(key) for key in getattr(manager,'graphs',{})],
                    adapter_installed=os.environ.get('Q36_V032_ARM')=='sd')
