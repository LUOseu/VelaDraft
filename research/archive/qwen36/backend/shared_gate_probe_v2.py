"""Compare original shared-expert gating with official torch.compile fusion; no handwritten kernel."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    from vlm_dflash.eval.native_resources import GPULease
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open
        model=Path('/dev/shm/qwen36_tp1_target');idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        weights=[]
        for layer in range(40):
            key=f'model.language_model.layers.{layer}.mlp.shared_expert_gate.weight'
            with safe_open(str(model/idx[key]),framework='pt') as f:weights.append(f.get_tensor(key).to(device='cuda',dtype=torch.bfloat16))
        def original(x,w,out):return torch.sigmoid(F.linear(x,w))*out
        def reduction(x,w,out):
            gate=(x.float()*w.float()).sum(-1,keepdim=True).to(x.dtype)
            return torch.sigmoid(gate)*out
        opts={'triton.cudagraphs':False,'emulate_precision_casts':True}
        methods={'original':original,'compiled_linear':torch.compile(original,fullgraph=True,dynamic=False,options=opts),
                 'compiled_reduction':torch.compile(reduction,fullgraph=True,dynamic=False,options=opts)}
        torch.manual_seed(20261101);checks=[];valid={name:True for name in methods}
        for m in (1,3,8):
            for scale in (.01,1.,10.):
                for layer,w in enumerate(weights):
                    x=torch.randn(m,2048,device='cuda',dtype=torch.bfloat16)*scale;out=torch.randn_like(x)*.1;expected=original(x,w,out)
                    for name,fn in methods.items():
                        actual=fn(x,w,out);err=float((actual.float()-expected.float()).abs().max());same=torch.equal(actual,expected)
                        close=torch.allclose(actual,expected,rtol=.015625,atol=.001)
                        valid[name]&=bool(close) and actual.dtype==expected.dtype
                        checks.append(dict(m=m,scale=scale,layer=layer,method=name,max_abs=err,bitwise=same,close=bool(close)))
        (root/'parity.json').write_text(json.dumps({'valid':valid,'checks':checks},indent=2))
        graphs={};alive=[]
        for m in (1,3,8):
            tensors=[(torch.randn(m,2048,device='cuda',dtype=torch.bfloat16),w,torch.randn(m,2048,device='cuda',dtype=torch.bfloat16)*.1) for w in weights]
            alive.extend(tensors)  # Retain graph input storage through every replay.
            for name,fn in methods.items():
                if not valid[name]:continue
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for vals in tensors:fn(*vals)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):alive.append([fn(*v) for v in tensors])
                graphs[m,name]=graph
        for graph in graphs.values():
            for _ in range(10):graph.replay()
        torch.cuda.synchronize();rng=random.Random(42);timings={k:[] for k in graphs}
        for repeat in range(30):
            order=list(graphs);rng.shuffle(order)
            for key in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(20):graphs[key].replay()
                b.record();b.synchronize();timings[key].append(a.elapsed_time(b)/(20*40))
        result={'status':'complete','real_gate_weights':40,'synthetic_inputs':True,'not_model_speedup':True,'precision_casts_emulated':True,'valid':valid,
                'results':[{ 'm':m,'method':name,'median_us_per_layer':statistics.median(v)*1000,'samples_ms':v,
                            'bitwise_cases':sum(x['bitwise'] for x in checks if x['method']==name),'cases':sum(x['method']==name for x in checks)} for (m,name),v in timings.items()]}
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({f'{m}_{name}':round(statistics.median(v)*1000,3) for (m,name),v in timings.items()}),flush=True)
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))

if __name__=='__main__':main()
