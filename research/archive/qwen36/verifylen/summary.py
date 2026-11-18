import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

def quantile(values, q):
    values=sorted(values);i=(len(values)-1)*q;lo=int(i)
    return values[lo]*(1-(i-lo))+values[min(lo+1,len(values)-1)]*(i-lo)

def summarize_stage(data):
    arm=data['arm'];k=data['k'];phase='decode' if arm=='ar' else 'verify';expected=1 if arm=='ar' else k+1
    groups=defaultdict(list)
    for row in data['rows']:
        ranks=row['ranks']
        if len(ranks)!=2 or len({r['rank'] for r in ranks})!=2:raise ValueError('Missing or duplicate TP rank')
        steps=defaultdict(dict)
        for rank in ranks:
            for span in rank['spans']:
                if span.get('phase')!=phase or span.get('scheduled_tokens')!=expected:continue
                if span['component'] not in ('target_forward','target_execute_inclusive','target_sampling_rejection','proposal_total_inclusive','proposal_positions_inputs','proposal_attention_metadata','proposal_build_inputs_inclusive','proposal_context_kv','proposal_feature_fusion','proposal_backbone','proposal_sampling_inclusive','proposal_lm_head'):continue
                key=(span['step'],span['component'])
                if rank['rank'] in steps[key]:raise ValueError('Duplicate per-rank span in a step')
                steps[key][rank['rank']]=span
        for step in {key[0] for key in steps}:
            a=steps.get((step,'target_execute_inclusive'),{});b=steps.get((step,'target_sampling_rejection'),{})
            if len(a)==2 and set(a)==set(b):
                steps[step,'verification_execute_plus_sampling']={rank:{'stream_elapsed_ms':a[rank]['stream_elapsed_ms']+b[rank]['stream_elapsed_ms'],'host_ms':a[rank]['host_ms']+b[rank]['host_ms']} for rank in a}
        for (_,component),pair in steps.items():
            if len(pair)!=2:
                if component in ('target_forward','target_execute_inclusive'):raise ValueError('Unpaired TP target span')
                continue  # Rank-local proposal spans stay in raw, not a fabricated paired aggregate.
            groups[row['id'],row['turn'],component].append((max(s['stream_elapsed_ms'] for s in pair.values()),max(s['host_ms'] for s in pair.values())))
    out=[]
    for (sid,turn,component),values in groups.items():
        gpu=[v[0] for v in values];host=[v[1] for v in values]
        out.append(dict(arm=arm,k=0 if arm=='ar' else k,session_id=sid,turn=turn,component=component,scheduled_positions=expected,calls=len(values),
                        mean_stream_ms=statistics.mean(gpu),p50_stream_ms=statistics.median(gpu),p90_stream_ms=quantile(gpu,.9),mean_host_ms=statistics.mean(host)))
    return out

def collect(root):
    root=Path(root);rows=[];states={}
    for name in ['ar']+[f'sd_k{k}' for k in range(1,8)]:
        p=root/'stages'/name/'profile.json'
        if not p.exists():states[name]='pending';continue
        data=json.loads(p.read_text());states[name]=data['status']
        if data['status'] in ('complete','recovered_complete'):rows.extend(summarize_stage(data))
    ar={(r['session_id'],r['turn']):r['mean_stream_ms'] for r in rows if r['arm']=='ar' and r['component']=='target_forward'}
    ar_total={(r['session_id'],r['turn']):r['mean_stream_ms'] for r in rows if r['arm']=='ar' and r['component']=='verification_execute_plus_sampling'}
    for row in rows:
        baseline=ar.get((row['session_id'],row['turn']))
        row['relative_to_ar_forward']=row['mean_stream_ms']/baseline if baseline and row['component']=='target_forward' else None
        baseline_total=ar_total.get((row['session_id'],row['turn']))
        row['relative_to_ar_execute_sampling']=row['mean_stream_ms']/baseline_total if baseline_total and row['component']=='verification_execute_plus_sampling' else None
    if rows:
        with (root/'component_latency.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        curve=[r for r in rows if r['component'] in ('target_forward','verification_execute_plus_sampling')]
        with (root/'verification_latency.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
    report={'stage_status':states,'rows':rows,'semantics':'Slowest TP-rank stream span per aligned forward step. Not kernel-only time, not end-to-end speedup. Initial contexts are matched; later decoding prefixes can diverge. Full-width verification steps only; raw profiles retain prefill/fallback/tail steps. Rank-local proposal components remain in raw and are not aggregated as paired TP components.'}
    (root/'SUMMARY.json').write_text(json.dumps(report,indent=2))
    lines=['# Draft length 1–7: Qwen3.6 target verification latency','','Per-context means of slowest TP-rank CUDA-stream spans. Forward and execute+sampling are separate metrics; the latter excludes draft and gaps between calls. Observer-affected diagnostic, not E2E speedup.','','| Draft K | Actual scheduled positions | Status | Target-forward ms | Relative to AR forward | Execute+sampling ms |','|---:|---:|---|---:|---:|---:|']
    baseline=[r['mean_stream_ms'] for r in rows if r['arm']=='ar' and r['component']=='target_forward']
    base=statistics.mean(baseline) if baseline else None
    for k in range(8):
        name='ar' if k==0 else f'sd_k{k}'
        vals=[r['mean_stream_ms'] for r in rows if r['k']==k and r['component']=='target_forward']
        mean=statistics.mean(vals) if vals else None
        totals=[r['mean_stream_ms'] for r in rows if r['k']==k and r['component']=='verification_execute_plus_sampling']
        total=statistics.mean(totals) if totals else None
        lines.append(f"| {'AR' if k==0 else k} | {k+1} | {states[name]} | {f'{mean:.3f}' if mean is not None else '—'} | {f'{mean/base:.3f}' if mean is not None and base else '—'} | {f'{total:.3f}' if total is not None else '—'} |")
    lines+=['','Per-context medians/P90 and counts: verification_latency.csv. Proposal and nested components: component_latency.csv. Do not sum inclusive spans and their children.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':collect(sys.argv[1])
