"""Screen public cuBLAS/cuBLASLt choices with real dense projection weights."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    from vlm_dflash.eval.native_resources import GPULease
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open
        model=Path(args.model);idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        def get(key):
            with safe_open(str(model/idx[key]),framework='pt') as f:return f.get_tensor(key).to('cuda')
        prefix='model.language_model.layers.0.'
        weights={'lm_head':get('lm_head.weight'),
                 'gdn_qkvz':torch.cat([get(prefix+'linear_attn.in_proj_qkv.weight'),get(prefix+'linear_attn.in_proj_z.weight')]),
                 'gdn_out':get(prefix+'linear_attn.out_proj.weight'),
                 'shared_up':torch.cat([get(prefix+'mlp.shared_expert.gate_proj.weight'),get(prefix+'mlp.shared_expert.up_proj.weight')]),
                 'shared_down':get(prefix+'mlp.shared_expert.down_proj.weight')}
        previous=torch.backends.cuda.preferred_blas_library();torch.manual_seed(42);checks=[];graphs={};alive=[]
        for name,w in weights.items():
            for m in (1,3,8):
                xs=[torch.randn(m,w.shape[1],device='cuda',dtype=w.dtype) for _ in range(4)]
                alive.extend(xs)  # CUDA graphs do not own external input tensors.
                torch.backends.cuda.preferred_blas_library('cublas');refs=[F.linear(x,w).clone() for x in xs]
                for backend in ('cublas','cublaslt'):
                    torch.backends.cuda.preferred_blas_library(backend)
                    for x,ref in zip(xs,refs):
                        value=F.linear(x,w)
                        checks.append(dict(name=name,m=m,backend=backend,bitwise=torch.equal(value,ref),
                                           max_abs=float((value.float()-ref.float()).abs().max()),
                                           close=torch.allclose(value,ref,rtol=.02,atol=.01)))
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for x in xs:F.linear(x,w)
                    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):alive.append([F.linear(x,w) for x in xs])
                    graphs[name,m,backend]=graph
        torch.backends.cuda.preferred_blas_library(previous)
        for key,g in graphs.items():
            print('WARM_GRAPH',key,flush=True)
            for _ in range(5):g.replay()
            torch.cuda.synchronize()
        torch.cuda.synchronize();rng=random.Random(42);times={k:[] for k in graphs}
        for repeat in range(30):
            order=list(graphs);rng.shuffle(order)
            for key in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(10):graphs[key].replay()
                b.record();b.synchronize();times[key].append(a.elapsed_time(b)/40)
        result={'status':'complete','gpu':torch.cuda.get_device_name(),'real_weights':True,'synthetic_inputs':True,'not_model_speedup':True,'checks':checks,
                'measurements':[dict(name=name,m=m,backend=backend,median_us=statistics.median(values)*1000) for (name,m,backend),values in times.items()]}
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['measurements'],indent=2),flush=True)
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))


if __name__=='__main__':main()
