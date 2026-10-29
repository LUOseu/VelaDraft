"""Future strict40-value exporter. Large I/O requires a later explicit root gate."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from contracts import CHECKPOINT_SHA,MARKER,RAW_ARCH,validate_config,validate_state

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_tensor_values(source,restored):
    import torch
    if set(source)!=set(restored):raise ValueError('Export changed tensor keys')
    tensors={}
    for name,t in source.items():
        other=restored[name]
        if t.dtype!=other.dtype or t.shape!=other.shape:raise ValueError('Export changed shape/dtype: '+name)
        original=t.contiguous().view(torch.uint8);reloaded=other.contiguous().view(torch.uint8)
        if not torch.equal(original,reloaded):raise ValueError('Export changed tensor bytes: '+name)
        tensors[name]={'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(original.numpy().tobytes()).hexdigest()}
    return tensors

def export(checkpoint,output,*,allow_large_io=False):
    if not allow_large_io:raise RuntimeError('Real checkpoint I/O deferred; explicit root large-I/O authorization required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only export requires CUDA hidden')
    import torch
    from safetensors.torch import save_file,load_file
    checkpoint=Path(checkpoint);output=Path(output)
    if output.exists():raise FileExistsError('Destination must be new')
    if sha256(checkpoint)!=CHECKPOINT_SHA:raise ValueError('Wrong canonical Domino parent')
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if payload.get('algorithm')!='domino' or payload.get('step')!=75000:raise ValueError('Wrong algorithm/step metadata')
    state=payload['model'];validate_state(state)
    config=json.loads(json.dumps(payload['draft_config']));validate_config(config,source=True)
    config.update(algorithm='domino',architectures=[RAW_ARCH],source_checkpoint_sha256=CHECKPOINT_SHA)
    config['dflash_config'][MARKER]=True
    config['rope_parameters']['mrope_section']=[24,20,20];config['rope_parameters']['mrope_interleaved']=True
    validate_config(config)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.domino_export_',dir=output.parent) as td:
        stage=Path(td)/'artifact';stage.mkdir()
        (stage/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        save_file({name:t.contiguous() for name,t in state.items()},str(stage/'model.safetensors'),metadata={'format':'pt','algorithm':'domino'})
        restored=load_file(str(stage/'model.safetensors'),device='cpu');validate_state(restored)
        tensors=verify_tensor_values(state,restored)
        if torch.cuda.is_initialized():raise AssertionError('Exporter initialized CUDA')
        receipt={'schema':'domino-export-v1','source_sha256':CHECKPOINT_SHA,'tensor_count':40,'all40_values_bitwise_equal':True,'cuda_initialized':False,'model_constructed':False,'config_sha256':sha256(stage/'config.json'),'weights_sha256':sha256(stage/'model.safetensors'),'tensors':tensors,'gpu_eligible':False,'config_changes':['algorithm','architectures','source_checkpoint_sha256','dflash_config.'+MARKER,'rope_parameters.mrope_section','rope_parameters.mrope_interleaved']}
        (stage/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        stage.rename(output)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--allow-large-io',action='store_true');a=p.parse_args()
    print(json.dumps(export(a.checkpoint,a.output,allow_large_io=a.allow_large_io),indent=2))
