"""CPU-only exact51-tensor export; no target, draft, or CUDA model construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def verify_tensor_values(source,restored):
    import torch
    if set(source)!=set(restored):raise ValueError('Export changed tensor keys')
    result={}
    for key,tensor in source.items():
        other=restored[key]
        if tensor.shape!=other.shape or tensor.dtype!=other.dtype:raise ValueError('Export changed shape/dtype: '+key)
        data=tensor.contiguous().view(torch.uint8)
        if not torch.equal(data,other.contiguous().view(torch.uint8)):raise ValueError('Export changed bytes: '+key)
        result[key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':hashlib.sha256(data.numpy().tobytes()).hexdigest()}
    return result


def export(checkpoint,output):
    from . import contracts as c
    import torch
    from safetensors.torch import load_file,save_file
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Export requires CUDA_VISIBLE_DEVICES empty')
    checkpoint=Path(checkpoint);output=Path(output)
    c.require(not output.exists(),'Export destination must be new')
    c.require(c.sha(checkpoint)==c.CHECKPOINT_SHA,'Wrong canonical checkpoint SHA')
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    c.require(payload['algorithm']=='dflash2' and payload['step']==75000 and payload['head']=='target_lm_head','Wrong checkpoint metadata/head policy')
    c.validate_state(payload['model']);config=json.loads(json.dumps(payload['draft_config']));c.validate_config(config,source=True)
    config.update(architectures=[c.ARCH],algorithm='dflash2',source_checkpoint_sha256=c.CHECKPOINT_SHA)
    config['dflash_config'][c.MARKER]=True
    config['rope_parameters'].update(mrope_section=[24,20,20],mrope_interleaved=True)
    c.validate_config(config)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dflash2_export_',dir=output.parent) as temp:
        stage=Path(temp)/'hf';stage.mkdir()
        (stage/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        save_file({k:v.contiguous() for k,v in payload['model'].items()},str(stage/'model.safetensors'),metadata={'format':'pt','algorithm':'dflash2'})
        restored=load_file(str(stage/'model.safetensors'),device='cpu');c.validate_state(restored)
        tensors=verify_tensor_values(payload['model'],restored)
        c.require(not torch.cuda.is_initialized(),'Exporter initialized CUDA')
        receipt={'schema':'dflash2-native-export-v1','checkpoint_sha256':c.CHECKPOINT_SHA,'algorithm':'dflash2','step':75000,
            'tensor_count':51,'all51_values_bitwise_equal':True,'cuda_initialized':False,
            'model_constructed':False,'GPU_numerical_or_performance_gate_passed':False,
            'config_sha256':c.sha(stage/'config.json'),'weights_sha256':c.sha(stage/'model.safetensors'),'tensors':tensors}
        (stage/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');stage.rename(output)
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='/home/lyh/a6000_3b_v2_L300/runs/dflash2_L300_s42/train/ckpt_final.pt');parser.add_argument('--output',required=True)
    args=parser.parse_args();result=export(args.checkpoint,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'},indent=2))
