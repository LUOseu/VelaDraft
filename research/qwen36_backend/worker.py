"""Install the opt-in backend before model load in each vLLM worker."""
from research.qwen36_backend.shared_gate import install, metadata
install()
from research.qwen36_rootcause.e2e_stage import MoEAuditWorker


class BackendAuditWorker(MoEAuditWorker):
    def moe_config_snapshot(self):
        snapshot = super().moe_config_snapshot()
        snapshot['shared_gate_fusion'] = metadata(self.model_runner.model)
        return snapshot
