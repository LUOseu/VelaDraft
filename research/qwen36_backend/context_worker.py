import os
from research.qwen36_backend.context_graph import install,audit
install()
from research.qwen36_rootcause.e2e_stage import MoEAuditWorker


class ContextAuditWorker(MoEAuditWorker):
    def moe_config_snapshot(self):
        snapshot=super().moe_config_snapshot()
        snapshot['context_graph_requested']=os.environ.get('Q36_CONTEXT_KV_GRAPH')=='1'
        snapshot['shared_experts_stream_token_threshold']=os.environ.get('VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD','256')
        return snapshot

    def context_graph_audit(self):
        drafter=getattr(self.model_runner,'drafter',None)
        return audit(drafter.model) if drafter is not None else []
