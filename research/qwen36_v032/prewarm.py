import json
import os
from pathlib import Path
from vlm_dflash.eval.native_resources import GPULease

task=Path('/root/autodl-tmp/qwen36_backend_20261101');root=task/'v032_prewarm_cuda130';root.mkdir(exist_ok=False)
with GPULease(1,root/'resources.jsonl') as lease:
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    import torch
    import flashinfer
    probs=torch.full((2,8),.125,dtype=torch.float32,device='cuda')
    result=flashinfer.sampling.top_p_renorm_probs(probs,.9)
    torch.cuda.synchronize()
    assert torch.equal(result,probs)
    (root/'result.json').write_text(json.dumps({'status':'complete','sampler_kernel_gpu_executed':True,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_home':os.environ['CUDA_HOME']}))
(root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))
print('V032_SAMPLING_KERNEL_VERIFIED',flush=True)
