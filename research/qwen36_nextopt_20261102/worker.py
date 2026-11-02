"""Audit the joint tuning table selected by the actual worker process."""
import hashlib
import os
from pathlib import Path
from research.qwen36_grouped_e2e_20261102.worker import PolicyWorker


class JointWorker(PolicyWorker):
    def backend_metadata(self):
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        value=super().backend_metadata()
        folder=os.environ.get('VLLM_TUNED_CONFIG_FOLDER')
        value['moe_tuning']=dict(folder=folder,
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(folder).glob('*.json')} if folder else {},
            configs={str(m):try_get_optimal_moe_config((256,1024,2048),(256,2048,512),8,None,m) for m in range(1,9)})
        return value
