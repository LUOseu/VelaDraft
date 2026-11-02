import os
from research.qwen36_v032.worker import AuditWorker
from research.qwen36_grouped_e2e_20261102.dispatch import install,metadata

if os.environ.get('Q36_MOE_POLICY')=='grouped':install()


class PolicyWorker(AuditWorker):
    def moe_policy_metadata(self):
        config=self.vllm_config
        assert config.parallel_config.tensor_parallel_size==1
        assert config.model_config.quantization is None
        assert str(config.model_config.dtype)=='torch.bfloat16'
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        modules=[m for m in self.model_runner.model.modules() if isinstance(m,MoERunner)]
        assert len(modules)==40 and all(m.routed_experts.expert_map is None for m in modules)
        return dict(metadata(),target_moe_layers=len(modules),native_expert_map_none=True)
