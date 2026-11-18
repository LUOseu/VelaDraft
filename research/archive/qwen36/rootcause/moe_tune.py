"""Small synthetic tuning probe using vLLM's official fused_experts + override_config APIs."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import statistics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--out',required=True)
    ap.add_argument('--tp-size',type=int,choices=(1,2),default=2);ap.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,3,4]);args=ap.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    from vlm_dflash.eval.native_resources import GPULease
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        import torch
        from vllm.model_executor.layers.fused_moe import override_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts,get_default_config,get_config_file_name,try_get_optimal_moe_config
        torch.manual_seed(20261101);dev='cuda';dtype=torch.bfloat16
        E,H,N,topk=256,2048,512//args.tp_size,8
        w1=torch.randn(E,2*N,H,device=dev,dtype=dtype)*.02
        w2=torch.randn(E,H,N,device=dev,dtype=dtype)*.02
        base=try_get_optimal_moe_config((E,2*N,H),(E,H,N),topk,None,1)
        configs={'default':base}
        for bn,bk,st in [(64,128,2),(32,128,2),(64,64,2),(128,64,2),(128,128,2),(32,64,2),(64,128,3)]:
            configs[f'n{bn}_k{bk}_s{st}']=dict(base,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,num_stages=st)
        samples={}
        for m in args.batch_sizes:
            for pattern in ('shared','disjoint','random'):
                cases=[]
                for rep in range(8):
                    x=torch.randn(m,H,device=dev,dtype=dtype)*.1
                    if pattern=='shared': ids=torch.arange(topk,device=dev,dtype=torch.int32)[None].expand(m,-1).contiguous()
                    elif pattern=='disjoint':ids=torch.arange(m*topk,device=dev,dtype=torch.int32).reshape(m,topk)
                    else:ids=torch.stack([torch.randperm(E,device=dev)[:topk].to(torch.int32) for _ in range(m)])
                    weights=torch.softmax(torch.randn(m,topk,device=dev),-1)
                    cases.append((x,weights,ids))
                samples[m,pattern]=cases
        graphs={};keepalive=[];checks=[];errors=[]
        for (m,pattern),cases in samples.items():
            ref=[]
            with override_config(base):
                for x,weights,ids in cases[:2]:ref.append(fused_experts(x,w1,w2,weights,ids,inplace=False).clone())
            for name,config in configs.items():
                key=f'm{m}_{pattern}_{name}'
                try:
                    with override_config(config):
                        for j,(x,weights,ids) in enumerate(cases[:2]):
                            val=fused_experts(x,w1,w2,weights,ids,inplace=False)
                            error=float((val.float()-ref[j].float()).abs().max())
                            torch.testing.assert_close(val,ref[j],rtol=.02,atol=2e-5)
                            checks.append(dict(key=key,case=j,max_abs=error,bitwise=torch.equal(val,ref[j])))
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for x,weights,ids in cases:fused_experts(x,w1,w2,weights,ids,inplace=False)
                        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            keepalive.append([fused_experts(x,w1,w2,weights,ids,inplace=False) for x,weights,ids in cases])
                        graphs[key]=graph
                except Exception as e:
                    errors.append(dict(key=key,error=repr(e)));print('CONFIG_FAILED',key,repr(e),flush=True)
        for graph in graphs.values():
            for _ in range(5):graph.replay()
        torch.cuda.synchronize();timings={k:[] for k in graphs};rng=random.Random(42)
        for repeat in range(12):
            keys=list(graphs);rng.shuffle(keys)
            for key in keys:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):graphs[key].replay()
                end.record();end.synchronize();timings[key].append(begin.elapsed_time(end)/(10*8))
        medians={k:statistics.median(v) for k,v in timings.items()}
        recommendations={}
        for m in args.batch_sizes:
            scores={name:statistics.mean(medians[f'm{m}_{p}_{name}']/medians[f'm{m}_{p}_default'] for p in ('shared','disjoint','random')) for name in configs if all(f'm{m}_{p}_{name}' in medians for p in ('shared','disjoint','random'))}
            best=min(scores,key=scores.get)
            # Candidate must not regress any routing pattern by more than 2%.
            if any(medians[f'm{m}_{p}_{best}']>1.02*medians[f'm{m}_{p}_default'] for p in ('shared','disjoint','random')):best='default'
            recommendations[m]=dict(name=best,config=configs[best],mean_normalized_latency=scores[best],patterns={p:dict(default_ms=medians[f'm{m}_{p}_default'],candidate_ms=medians[f'm{m}_{p}_{best}']) for p in ('shared','disjoint','random')})
            print('RECOMMEND',m,json.dumps(recommendations[m]),flush=True)
        # Preserve stock defaults for every other M up to the actual 4096-token engine budget.
        table={str(m):(recommendations[m]['config'] if m in recommendations else try_get_optimal_moe_config((E,2*N,H),(E,H,N),topk,None,m)) for m in range(1,4097)}
        tuned=root/'tuned_configs';tuned.mkdir();filename=get_config_file_name(E,N,None)
        (tuned/filename).write_text(json.dumps(table,indent=2))
        result={'status':'complete','synthetic_microbenchmark':True,'not_model_speedup':True,'shape':dict(experts=E,hidden=H,local_intermediate=N,topk=topk,dtype='bf16'),
                'configs':configs,'parity_checks':checks,'failed_configs':errors,'median_ms':medians,'recommendations':recommendations,'tuned_file':str(tuned/filename),'timings':timings}
        (root/'result.json').write_text(json.dumps(result,indent=2))
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))

if __name__=='__main__':main()
