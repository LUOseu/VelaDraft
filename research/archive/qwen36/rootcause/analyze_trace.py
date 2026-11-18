"""Join CUDA kernels to CPU launch scopes by CUPTI correlation, preserving graph replay nodes."""
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

def category(name):
    low=name.lower()
    if 'fused_moe_kernel' in low:return 'moe_expert_gemm'
    if 'gated_delta_rule' in low or 'sigmoid_gating_delta' in low:return 'gdn_state_update'
    if 'causal_conv' in low:return 'gdn_convolution'
    if any(s in low for s in ('all_reduce','allreduce','cross_device_reduce','nccl')):return 'tp_collective'
    if 'flash' in low or 'attention' in low or 'fmha' in low:return 'dense_attention'
    if 'topk' in low or 'top_k' in low:return 'routing_topk'
    if 'gemm' in low or 'gemv' in low:return 'other_gemm'
    if 'rms_norm' in low or 'layer_norm' in low:return 'normalization'
    return 'other'

def analyze(events,spans,component,phase,q):
    annotations=sorted([e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='q36::'+component and e.get('ph')=='X'],key=lambda e:e['ts'])
    component_spans=[s for s in spans if s['component']==component]
    if len(annotations)!=len(component_spans):raise ValueError(f'Cannot align {component}: {len(annotations)} scopes vs {len(component_spans)} spans')
    selected={i for i,s in enumerate(component_spans) if s.get('phase')==phase and s.get('scheduled_tokens')==q}
    if not selected:return {'calls':0}
    starts=[e['ts'] for e in annotations];correlations={}
    for event in events:
        if event.get('cat') not in ('cuda_runtime','cuda_driver'):continue
        corr=event.get('args',{}).get('correlation')
        if corr is None:continue
        i=bisect_right(starts,event['ts'])-1
        if i<0 or i not in selected:continue
        scope=annotations[i]
        if event.get('tid')==scope.get('tid') and event['ts']<=scope['ts']+scope['dur']:correlations[corr]=i
    groups=defaultdict(float);names=defaultdict(lambda:[0,0.]);steps=defaultdict(float);matched=0
    for event in events:
        if event.get('cat')!='kernel':continue
        corr=event.get('args',{}).get('correlation')
        if corr not in correlations:continue
        name=event['name'];duration=event['dur'];matched+=1
        groups[category(name)]+=duration;names[name][0]+=1;names[name][1]+=duration;steps[correlations[corr]]+=duration
    count=len(selected)
    return {'calls':count,'component':component,'phase':phase,'scheduled_tokens':q,'matched_kernel_count':matched,
            'steps_with_kernels':len(steps),'event_stream_mean_ms':statistics.mean(component_spans[i]['stream_elapsed_ms'] for i in selected),
            'kernel_activity_us_per_step':{k:v/count for k,v in sorted(groups.items(),key=lambda x:-x[1])},
            'kernel_names':[dict(name=n,calls=v[0],sum_us=v[1],us_per_step=v[1]/count) for n,v in sorted(names.items(),key=lambda x:-x[1][1])],
            'note':'Kernel activity durations can overlap; their sum is not critical-path wall time. Prefill excluded via paired scheduler metadata.'}

def main():
    root=Path(sys.argv[1]);results={}
    for arm in ('ar','sd'):
        p=root/arm/'profile.json'
        if not p.exists():continue
        report=json.loads(p.read_text())
        if report['status']!='complete':continue
        arm_rows=[]
        for row in report['rows']:
            for rank in row['ranks']:
                trace=json.loads(Path(rank['trace_file']).read_text())
                for comp in ('target_forward','target_execute_inclusive','proposal_total_inclusive'):
                    if arm=='ar' and comp.startswith('proposal'):continue
                    result=analyze(trace['traceEvents'],rank['spans'],comp,'decode' if arm=='ar' else 'verify',1 if arm=='ar' else report['k']+1)
                    result.update(rank=rank['rank'],id=row['id'],turn=row['turn']);arm_rows.append(result)
        results[arm]=arm_rows
    (root/'kernel_analysis.json').write_text(json.dumps(results,indent=2))
    for arm,items in results.items():
        for item in items:
            print(arm,item['rank'],item['component'],item['calls'],item.get('kernel_activity_us_per_step'))

if __name__=='__main__':main()
