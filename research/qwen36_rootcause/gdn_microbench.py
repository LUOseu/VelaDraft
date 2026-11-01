"""Synthetic-shape diagnostic of unchanged upstream GDN kernels, not model E2E."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    from vlm_dflash.eval.native_resources import GPULease
    with GPULease(args.gpu,out/'resources.jsonl') as lease:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        import torch
        import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as generic
        from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as packed
        from launch_override import LaunchOverride
        torch.manual_seed(20261101)
        dev='cuda';dtype=torch.bfloat16;H,HV,D=8,16,128;layers=30
        original=generic.fused_sigmoid_gating_delta_rule_update_kernel
        def case(q):
            return dict(q=torch.randn(1,q,H,D,device=dev,dtype=dtype),k=torch.randn(1,q,H,D,device=dev,dtype=dtype),v=torch.randn(1,q,HV,D,device=dev,dtype=dtype),
                        a=torch.randn(q,HV,device=dev,dtype=dtype)*.1,b=torch.randn(q,HV,device=dev,dtype=dtype)*.1,
                        A_log=torch.zeros(HV,device=dev),dt_bias=torch.zeros(HV,device=dev),
                        initial_state=torch.randn(q+1,HV,D,D,device=dev)*.01,
                        cu_seqlens=torch.tensor([0,q],device=dev,dtype=torch.int32),ssm_state_indices=torch.arange(1,q+1,device=dev,dtype=torch.int32)[None],
                        num_accepted_tokens=torch.ones(1,device=dev,dtype=torch.int32),use_qk_l2norm_in_kernel=True)
        parity=[]
        for q in (1,2,3,4):
            c=case(q);initial=c['initial_state'].clone()
            for accepted in sorted({1,max(1,q//2),q}):
                c['num_accepted_tokens'].fill_(accepted)
                c['initial_state'].copy_(initial);generic.fused_sigmoid_gating_delta_rule_update_kernel=original
                ref_o,_=generic.fused_sigmoid_gating_delta_rule_update(**c);ref_o=ref_o.clone();ref_h=c['initial_state'].clone()
                for warps in (1,2,4):
                    c['initial_state'].copy_(initial);generic.fused_sigmoid_gating_delta_rule_update_kernel=LaunchOverride(original,warps)
                    val,_=generic.fused_sigmoid_gating_delta_rule_update(**c)
                    state_error=float((ref_h-c['initial_state']).abs().max());output_error=float((ref_o.float()-val.float()).abs().max())
                    torch.testing.assert_close(val,ref_o,rtol=.02,atol=2e-4)
                    torch.testing.assert_close(c['initial_state'],ref_h,rtol=1e-4,atol=1e-5)
                    parity.append(dict(q=q,accepted=accepted,warps=warps,state_max_abs=state_error,output_max_abs=output_error,output_bitwise=torch.equal(val,ref_o),state_bitwise=torch.equal(c['initial_state'],ref_h)))
        generic.fused_sigmoid_gating_delta_rule_update_kernel=original
        (out/'parity.json').write_text(json.dumps({'status':'passed','checks':parity},indent=2))
        cases={q:[case(q) for _ in range(layers)] for q in (1,2,3,4)}
        packed_cases=[]
        for c in cases[1]:
            packed_cases.append(dict(mixed_qkv=torch.cat([c['q'][0].flatten(1),c['k'][0].flatten(1),c['v'][0].flatten(1)],dim=-1).contiguous(),
                a=c['a'],b=c['b'],A_log=c['A_log'],dt_bias=c['dt_bias'],scale=D**-.5,initial_state=c['initial_state'],
                out=torch.empty(1,1,HV,D,device=dev,dtype=dtype),ssm_state_indices=c['ssm_state_indices'][:,0].contiguous(),use_qk_l2norm_in_kernel=True))
        graphs={};keepalive=[]
        def capture(name,fn):
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):fn()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):keepalive.append(fn())
            graphs[name]=graph
        capture('packed_ar_q1',lambda:[packed(**c) for c in packed_cases])
        for q in (1,2,3,4):
            for warps in (4,1,2):
                generic.fused_sigmoid_gating_delta_rule_update_kernel=LaunchOverride(original,warps)
                capture(f'spec_q{q}_w{warps}',lambda q=q:[generic.fused_sigmoid_gating_delta_rule_update(**c) for c in cases[q]])
        generic.fused_sigmoid_gating_delta_rule_update_kernel=original
        for graph in graphs.values():
            for _ in range(20):graph.replay()
        torch.cuda.synchronize()
        timings={name:[] for name in graphs};rng=random.Random(42)
        for repeat in range(20):
            order=list(graphs);rng.shuffle(order)
            for name in order:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):graphs[name].replay()
                end.record();end.synchronize();timings[name].append(begin.elapsed_time(end)/(20*layers))
        report={'status':'complete','synthetic_microbenchmark':True,'not_model_speedup':True,'rank_local_shape':dict(key_heads=H,value_heads=HV,key_dim=D,value_dim=D,ssm_dtype='float32',projection_dtype='bfloat16'),'independent_state_buffers_per_graph':layers,
                'source_sha256':hashlib.sha256(Path(generic.__file__).read_bytes()).hexdigest(),'parity_checks':len(parity),
                'measurements':{name:dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),min_ms=min(v),samples=v) for name,v in timings.items()}}
        (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({name:round(statistics.median(v)*1000,3) for name,v in timings.items()}),flush=True)
        # Confirm CUPTI can see kernels inside replayed CUDA graphs before a costly whole-model trace.
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                graphs['packed_ar_q1'].replay();graphs['spec_q2_w4'].replay();torch.cuda.synchronize()
            prof.export_chrome_trace(str(out/'probe_trace.json'))
            events=json.loads((out/'probe_trace.json').read_text()).get('traceEvents',[])
            kernels=[e for e in events if e.get('cat')=='kernel'];(out/'trace_probe.json').write_text(json.dumps({'kernel_events':len(kernels),'names':sorted({e.get('name') for e in kernels})},indent=2))
        except Exception as e:(out/'trace_probe.json').write_text(json.dumps({'error':repr(e)}))
    (out/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))

if __name__=='__main__':main()
