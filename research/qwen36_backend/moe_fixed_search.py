"""Fixed-query-width backend tuning against the already optimized runtime, using real expert weights."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--out',required=True);p.add_argument('--baseline-folder',required=True);args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    from vlm_dflash.eval.native_resources import GPULease
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['VLLM_TUNED_CONFIG_FOLDER']=args.baseline_folder
        import torch
        from safetensors import safe_open
        from vllm.model_executor.layers.fused_moe import override_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts,try_get_optimal_moe_config,get_config_file_name
        model=Path('/dev/shm/qwen36_tp1_target');idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];weights=[]
        for suffix in ('gate_up_proj','down_proj'):
            key=f'model.language_model.layers.0.mlp.experts.{suffix}'
            with safe_open(str(model/idx[key]),framework='pt') as f:weights.append(f.get_tensor(key).to('cuda').contiguous())
        w1,w2=weights;assert tuple(w1.shape)==(256,1024,2048) and tuple(w2.shape)==(256,2048,512)
        m=3;E,H,N,topk=256,2048,512,8;torch.manual_seed(20261101)
        baseline=try_get_optimal_moe_config(tuple(w1.shape),tuple(w2.shape),topk,None,m);configs={'current':baseline}
        for bn,bk,st,warps in itertools.product((32,64,128,256),(64,128,256),(2,3),(4,8)):
            if warps==8 and (bn<128 or st!=2):continue
            if (16+bn)*bk*2*st>160*1024:continue
            name=f'n{bn}_k{bk}_s{st}_w{warps}';cfg=dict(baseline,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=bk,num_stages=st,num_warps=warps)
            if cfg not in configs.values():configs[name]=cfg
        cases={}
        for pattern in ('shared','disjoint','random'):
            vals=[]
            for i in range(8):
                x=torch.randn(m,H,device='cuda',dtype=torch.bfloat16)
                if pattern=='shared':ids=torch.arange(topk,device='cuda',dtype=torch.int32)[None].expand(m,-1).contiguous()
                elif pattern=='disjoint':ids=torch.arange(m*topk,device='cuda',dtype=torch.int32).view(m,topk)
                else:ids=torch.stack([torch.randperm(E,device='cuda')[:topk].int() for _ in range(m)])
                scores=torch.softmax(torch.randn(m,topk,device='cuda'),-1);vals.append((x,scores,ids))
            cases[pattern]=vals
        graphs={};alive=[];checks=[];failed=[]
        for pattern,vals in cases.items():
            with override_config(baseline):refs=[fused_experts(x,w1,w2,s,i).clone() for x,s,i in vals[:2]]
            for name,cfg in configs.items():
                key=pattern+'_'+name
                try:
                    with override_config(cfg):
                        for j,(x,s,i) in enumerate(vals[:2]):
                            v=fused_experts(x,w1,w2,s,i);same=torch.equal(v,refs[j]);error=float((v.float()-refs[j].float()).abs().max())
                            checks.append(dict(key=key,bitwise=same,max_abs=error))
                            if not same:raise ValueError('Reject candidate with non-bitwise output in real-weight probe')
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for x,s,i in vals:fused_experts(x,w1,w2,s,i)
                        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):alive.append([fused_experts(x,w1,w2,s,i) for x,s,i in vals])
                        graphs[key]=g
                except Exception as e:failed.append(dict(key=key,error=repr(e)));print('SKIP',key,repr(e),flush=True)
        for g in graphs.values():
            for _ in range(5):g.replay()
        torch.cuda.synchronize();rng=random.Random(123);times={k:[] for k in graphs}
        for repeat in range(16):
            order=list(graphs);rng.shuffle(order)
            for key in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(10):graphs[key].replay()
                b.record();b.synchronize();times[key].append(a.elapsed_time(b)/(10*8))
        med={k:statistics.median(v) for k,v in times.items()};scored={}
        for name in configs:
            if not all(p+'_'+name in med for p in cases):continue
            ratios={p:med[p+'_'+name]/med[p+'_current'] for p in cases}
            if max(ratios.values())<=1.02:scored[name]=statistics.mean(ratios.values())
        best=min(scored,key=scored.get);table_path=next(Path(args.baseline_folder).glob('E=256,N=512,device_name=*.json'));table=json.loads(table_path.read_text());table['3']=configs[best]
        dest=root/'candidate_config';dest.mkdir();(dest/table_path.name).write_text(json.dumps(table,indent=2))
        result={'status':'complete','fixed_m':3,'fixed_k':2,'real_expert_layer':0,'synthetic_hidden_and_routes':True,'baseline_folder':args.baseline_folder,'configs':configs,'checks':checks,'failed':failed,'median_ms':med,'scores':scored,'best':best,'candidate_config':configs[best],'not_model_speedup':True}
        (root/'result.json').write_text(json.dumps(result,indent=2));print('BEST',best,scored[best],configs[best],flush=True)
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))

if __name__=='__main__':main()
