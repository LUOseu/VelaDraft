"""Native vLLM E2E protocol. Imports and dry runs never initialize CUDA.

Prepared exports and method adapters are inputs, not automatic conversions.
HF checkpoint diagnostics remain in eval.run and are not this timer domain.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

FORMAT = 'veladraft-native-vllm-e2e-v1'
METHODS = ('dflash','dspark','domino','jetspec','dflash2','dflare','dfly')
NUMERIC_OVERRIDES = ('FAIR_TORCH_PRECISION_PROFILE','TORCH_BLAS_PREFER_CUBLASLT',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG',
    'VLLM_ATTENTION_BACKEND','VLLM_FLOAT32_MATMUL_PRECISION','CUDA_LAUNCH_BLOCKING')


def digest(path):
    cache=os.environ.get('QWEN8B_DIGEST_CACHE')
    if cache:
        from .digest_cache import cached_sha256
        return cached_sha256(path,cache)
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def validate_environment(environment):
    for key in NUMERIC_OVERRIDES:
        if key in environment:raise ValueError('Non-native numerical/profiling override: '+key)
    if environment.get('VLLM_BATCH_INVARIANT','0')!='0':
        raise ValueError('Non-native numerical override: VLLM_BATCH_INVARIANT')


def stage_order(repeats):
    if type(repeats) is not int or repeats<1:raise ValueError('repeats must be positive')
    return [f'{arm}_r{repeat}' for repeat in range(repeats)
            for arm in (('ar','sd') if repeat%2==0 else ('sd','ar'))]


def engine_args(plan,arm):
    if arm not in ('ar','sd'):raise ValueError('Unknown native arm')
    args=dict(model=plan['target'],dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=plan['max_model_len'],max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=plan['gpu_memory_utilization'],enable_prefix_caching=False,
        mm_processor_cache_gb=0,limit_mm_per_prompt={'image':1},enforce_eager=False,
        generation_config='vllm',disable_log_stats=True,
        mm_processor_kwargs=dict(min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels']),
        worker_extension_cls='vlm_dflash.eval.native_runtime.WorkerMetadata')
    if arm=='sd':
        size=plan['speculative_tokens']+1
        args.update(block_size=16,attention_config={'backend':'FLASH_ATTN'},
            compilation_config=dict(mode=3,cudagraph_mode='PIECEWISE',cudagraph_capture_sizes=[size],max_cudagraph_capture_size=size),
            speculative_config=dict(model=plan['draft'],method='dflash',num_speculative_tokens=plan['speculative_tokens'],
                draft_tensor_parallel_size=1,attention_backend='FLASH_ATTN'))
        if plan.get('method','dflash') in ('dspark','domino','dflash2','dflare','dfly'):
            args['speculative_config']['use_local_argmax_reduction']=False
    return args


def load_sessions(source,images_dir,count):
    rows=[json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    if count<1 or len(rows)<count:raise ValueError('Requested prompt count exceeds available sessions')
    sessions=[];seen=set()
    for row in rows[:count]:
        sid=str(row['id'])
        if sid in seen:raise ValueError('Duplicate session ID: '+sid)
        seen.add(sid)
        if len(row['images'])!=1:raise ValueError('Native protocol requires one image per session')
        conversation=row['conversations'];questions=[]
        for i,turn in enumerate(conversation):
            expected='human' if i%2==0 else 'gpt'
            if turn.get('from')!=expected:raise ValueError('Expected alternating human/gpt conversations')
            if i%2==0:
                question=turn['value'].replace('<image>','').strip()
                if not question:raise ValueError('Empty human question')
                questions.append(question)
        if not questions:raise ValueError('No human question')
        image=(Path(images_dir)/row['images'][0]).resolve()
        sessions.append(dict(id=sid,questions=questions,image=str(image),image_sha256=digest(image)))
    return sessions


def bootstrap_ci(ar,sd,*,samples=2000,seed=42):
    if set(ar)!=set(sd) or not ar:raise ValueError('Incomplete session clusters')
    keys=sorted(ar);a=[sum(ar[k]) for k in keys];b=[sum(sd[k]) for k in keys]
    rng=random.Random(seed);values=[]
    for _ in range(samples):
        selected=[rng.randrange(len(keys)) for _ in keys]
        values.append(sum(a[i] for i in selected)/sum(b[i] for i in selected))
    values.sort()
    def quantile(q):
        index=(len(values)-1)*q;low=int(index);fraction=index-low
        return values[low]*(1-fraction)+values[min(low+1,len(values)-1)]*fraction
    return [quantile(.025),quantile(.975)]


def summarize(plan,reference,reports):
    """Statistics over trusted rows; coordinator separately validates GPU/source/resources."""
    result=dict(format=FORMAT,status='failed',paper_ready=False,quality_evaluation_required=True)
    try:
        expected=[(s['id'],t) for s in plan['sessions'] for t in range(len(s['questions']))]
        key=lambda r:(r['id'],r['turn'])
        if [key(r) for r in reference]!=expected:raise ValueError('Incomplete canonical input scope')
        refs={key(r):r for r in reference};order=stage_order(plan['repeats'])
        if [r['stage'] for r in reports]!=order:raise ValueError('Incomplete/reordered process sequence')
        by_stage={};uuids=[]
        for report in reports:
            if report['status']!='complete':raise ValueError('Failed stage')
            rows=report['rows']
            if [(r['phase'],*key(r)) for r in rows]!=[(phase,*item) for phase in ('warmup','timed') for item in expected]:
                raise ValueError('Missing/duplicate physical requests')
            for row in rows:
                if row['status']!='complete':raise ValueError('Failed physical request')
                if row['prompt_token_ids']!=refs[key(row)]['prompt_token_ids']:raise ValueError('AR/SD input tokens differ')
                elapsed=(row['ended_ns']-row['began_ns'])/1e9
                if not math.isfinite(row['wall_seconds']) or elapsed<=0 or not math.isclose(row['wall_seconds'],elapsed,rel_tol=1e-10,abs_tol=1e-12):
                    raise ValueError('Invalid raw duration')
                uuids.append(row['image_uuid'])
            by_stage[report['stage']]=[r for r in rows if r['phase']=='timed']
        if len(uuids)!=len(set(uuids)):raise ValueError('Physical image UUID reused')
        ar={s['id']:[] for s in plan['sessions']};sd={s['id']:[] for s in plan['sessions']}
        totals={'ar':0.,'sd':0.};tokens={'ar':0,'sd':0};exact=0;terminations={'ar':{},'sd':{}};paired=0
        for repeat in range(plan['repeats']):
            left,right=by_stage[f'ar_r{repeat}'],by_stage[f'sd_r{repeat}']
            for a,b in zip(left,right):
                paired+=1;exact+=a['tokens']==b['tokens']
                for arm,row,groups in (('ar',a,ar),('sd',b,sd)):
                    groups[row['id']].append(row['wall_seconds']);totals[arm]+=row['wall_seconds'];tokens[arm]+=len(row['tokens'])
                    end=str(row['finish_reason']);terminations[arm][end]=terminations[arm].get(end,0)+1
        result.update(status='complete',speedup_sum_ar_over_sd=totals['ar']/totals['sd'],
            ar_seconds=totals['ar'],sd_seconds=totals['sd'],ar_output_tokens=tokens['ar'],sd_output_tokens=tokens['sd'],
            timed_pairs=paired,exact_output_pairs=exact,all_measured_output_tokens_exact=exact==paired,terminations=terminations,
            session_cluster_CI95=bootstrap_ci(ar,sd),bootstrap_samples=2000,bootstrap_seed=42,
            confidence_interval_scope='session clusters, fixed process sequence; not run/card/seed variance',
            repeats=plan['repeats'],sessions=len(ar),smoke=plan['repeats']<3 or len(ar)<2)
    except (KeyError,ValueError,TypeError,ZeroDivisionError) as error:result['error']=str(error)
    return result


def verify_plan(plan):
    for name,expected in plan['files'].items():
        if digest(name)!=expected:raise ValueError('Pinned artifact changed: '+name)
    validate_environment(os.environ)


def verify_preservation(plan):
    adapter=plan.get('adapter')
    if not adapter:return {'scope':'prepared stock DFlash export; no checkpoint preservation assertion'}
    import torch
    from safetensors.torch import load_file
    checkpoint=torch.load(adapter['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    if checkpoint.get('algorithm','dflash')!=plan['method']:raise ValueError('Checkpoint method mismatch')
    original=checkpoint['model'];exported=load_file(str(Path(plan['draft'])/'model.safetensors'))
    if set(original)!=set(exported):raise ValueError('Export dropped or substituted trained modules')
    for name,value in original.items():
        other=exported[name]
        if value.shape!=other.shape or value.dtype!=other.dtype or not torch.equal(value.contiguous().reshape(-1).view(torch.uint8),other.contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError('Trained tensor differs: '+name)
    return dict(scope='complete CPU checkpoint vs export; not GPU canonical parity',tensor_count=len(original),all_values_bitwise_equal=True)


def make_plan(args):
    validate_environment(os.environ)
    if args.method not in METHODS:raise ValueError('Unregistered method')
    if args.speculative_tokens<1 or args.max_new_tokens<1 or args.repeats<1:raise ValueError('Budgets/repeats must be positive')
    if args.max_model_len<args.max_new_tokens or not 0<args.gpu_memory_utilization<1:raise ValueError('Invalid context/memory budget')
    if not 0<args.min_pixels<=args.max_pixels:raise ValueError('Invalid image pixel bounds')
    plan={k:getattr(args,k) for k in ('speculative_tokens','max_new_tokens','max_model_len','gpu_memory_utilization','min_pixels','max_pixels','repeats','method','gpu')}
    plan.update(format=FORMAT,seed=43,stop_token_ids=[151645,151643],target=str(Path(args.target).resolve()),draft=str(Path(args.draft).resolve()),adapter=None,files={})
    plan['source_integrity_policy']='SHA256 on first observation or file metadata change; inode/size/mtime/ctime keyed cache' if os.environ.get('QWEN8B_DIGEST_CACHE') else 'full SHA256 on every observation'
    for directory in (plan['target'],plan['draft']):
        if not Path(directory).is_dir():raise ValueError('Prepared local model directory required: '+directory)
    draft=Path(plan['draft'])
    if set(draft.glob('*.safetensors'))!={draft/'model.safetensors'} or list(draft.glob('*.bin')) or list(draft.glob('*.index.json')):
        raise ValueError('Prepared draft requires one complete model.safetensors, no alternate weight sources')
    for directory in (Path(plan['target']),draft):
        for path in sorted(directory.iterdir()):
            if path.is_file():plan['files'][str(path.resolve())]=digest(path)
    source=Path(args.source).resolve();plan['files'][str(source)]=digest(source)
    plan['sessions']=load_sessions(source,args.images_dir or source.parent/'images',args.num_prompts)
    for session in plan['sessions']:plan['files'][session['image']]=session['image_sha256']
    if args.runtime_manifest:
        path=Path(args.runtime_manifest).resolve();adapter=json.loads(path.read_text())
        if adapter.get('method')!=args.method or not adapter.get('scope'):raise ValueError('Adapter must declare method and implementation scope')
        for field in ('bootstrap','checkpoint'):adapter[field]=str((path.parent/adapter[field]).resolve())
        pins={str((path.parent/k).resolve()):v for k,v in adapter['files'].items()}
        if any(name not in pins for name in (adapter['bootstrap'],adapter['checkpoint'])):raise ValueError('Adapter bootstrap and full checkpoint need explicit SHA pins')
        if any(k in adapter.get('environment',{}) for k in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','MKL_NUM_THREADS')):
            raise ValueError('Adapter cannot change process/device/thread controls')
        validate_environment(adapter.get('environment',{}));plan['files'].update(pins)
        plan['files'][str(path)]=digest(path);plan['adapter']=adapter
    elif args.method!='dflash':raise ValueError('This method requires an explicit preserved full export and runtime manifest')
    own=Path(__file__).parent
    for name in ('native.py','native_runtime.py','native_resources.py','native_bootstrap/sitecustomize.py','digest_cache.py'):
        path=own/name;plan['files'][str(path.resolve())]=digest(path)
    try:
        dist=importlib.metadata.distribution('vllm')
        if dist.version!='0.22.1':raise ValueError('Native protocol pins vLLM0.22.1')
        for name in ('vllm/engine/arg_utils.py','vllm/model_executor/models/qwen3_dflash.py','vllm/v1/spec_decode/dflash.py','vllm/v1/sample/rejection_sampler.py'):
            path=Path(dist.locate_file(name));plan['files'][str(path.resolve())]=digest(path)
    except importlib.metadata.PackageNotFoundError:
        if not args.dry_run:raise ValueError('Install vLLM0.22.1 in a separate CUDA runtime environment') from None
    verify_plan(plan)
    return plan


def add_arguments(parser):
    for name in ('target','draft','source','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--images-dir');parser.add_argument('--runtime-manifest')
    parser.add_argument('--method',choices=METHODS,default='dflash')
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--num-prompts',type=int,default=32)
    parser.add_argument('--max-new-tokens',type=int,default=1024);parser.add_argument('--speculative-tokens',type=int,default=7)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--max-model-len',type=int,default=8192)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.45)
    parser.add_argument('--min-pixels',type=int,default=200704);parser.add_argument('--max-pixels',type=int,default=1003520)
    parser.add_argument('--dry-run',action='store_true',help='validate prepared artifacts and emit plan; no LLM/CUDA')


def run(args):
    out=Path(args.out).resolve();evidence=out.with_suffix('.evidence')
    if out.exists() or evidence.exists():raise FileExistsError('Existing native evidence preserved; choose a new --out')
    if args.dry_run:
        plan=make_plan(args)
        out.parent.mkdir(parents=True,exist_ok=True);evidence.mkdir()
        save(evidence/'plan.json',plan)
        save(out,dict(format=FORMAT,status='planned',GPU_executed=False,paper_ready=False,plan=plan));return 0
    out.parent.mkdir(parents=True,exist_ok=True);evidence.mkdir()
    from .native_resources import GPULease
    result=dict(format=FORMAT,status='failed',paper_ready=False,quality_evaluation_required=True,evidence=str(evidence))
    try:
        # Reserve before expensive hashing so coordinated jobs see the lease immediately.
        with GPULease(args.gpu,evidence/'resources.jsonl') as lease:
            plan=make_plan(args)
            verify_preservation(plan)
            plan['gpu_uuid']=lease.uuid;save(evidence/'plan.json',plan)
            plan_path=evidence/'plan.json';plan_sha=digest(plan_path)
            for stage in ('reference',*stage_order(args.repeats)):
                env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(args.gpu),
                    VELADRAFT_NATIVE_PLAN=str(plan_path),VELADRAFT_NATIVE_PLAN_SHA256=plan_sha,
                    VELADRAFT_NATIVE_ARM='sd' if stage.startswith('sd') else 'ar',
                    VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_BATCH_INVARIANT='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    VLLM_CACHE_ROOT=str(evidence/stage/'cache'),TORCHINDUCTOR_CACHE_DIR=str(evidence/stage/'inductor'),
                    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
                repo=Path(__file__).resolve().parents[2]
                env['PYTHONPATH']=os.pathsep.join((str(Path(__file__).parent/'native_bootstrap'),str(repo)))
                command=[sys.executable,'-m','vlm_dflash.eval.native_runtime','--plan',str(plan_path),'--stage',stage,'--directory',str(evidence)]
                with (evidence/(stage+'.log')).open('x') as log:
                    completed=subprocess.run(command,env=env,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:raise ValueError('Native stage failed: '+stage+'; preserved raw/logs')
        resources=lease.validate();verify_plan(plan)
        reference=json.loads((evidence/'reference.json').read_text());reports=[json.loads((evidence/(s+'.json')).read_text()) for s in stage_order(args.repeats)]
        if reference['status']!='complete':raise ValueError('Canonical creator failed')
        for report in (reference,*reports):
            for row in report['rows']:
                if not resources['began_wall_ns']<=row['began_wall_ns']<=row['ended_wall_ns']<=resources['ended_wall_ns']:
                    raise ValueError('Resource samples do not bracket physical requests')
            arm='sd' if report['stage'].startswith('sd') else 'ar'
            if report['engine_args']!=engine_args(plan,arm):raise ValueError('Actual engine constructor differs from frozen plan')
            for field in ('numeric_parent','worker_before','worker_after'):
                actual=report[field]
                if field=='numeric_parent':expected=reference[field]
                else:expected=reference['worker_before']
                if field=='numeric_parent':
                    if actual!=expected:raise ValueError('Numerical parent configuration differs from native AR')
                elif actual['numeric']!=expected['numeric']:raise ValueError('Worker numerical flags differ from native AR')
            if report['gpu_uuid']!=lease.uuid:raise ValueError('Actual CUDA GPU differs from resource lease')
        result=summarize(plan,reference['rows'],reports);result.update(evidence=str(evidence),resource_evidence=resources,
            scope='steady-state batch1 image/RGB/template/prefill/decode/output E2E; startup and full warmup excluded',
            method=args.method,implementation_scope=(plan.get('adapter') or {}).get('scope','prepared stock DFlash'),
            source_plan_sha256=plan_sha,source_preservation=verify_preservation(plan),resources_verified=True)
    except Exception as error:
        result=dict(format=FORMAT,status='failed',paper_ready=False,quality_evaluation_required=True,
                    evidence=str(evidence),error=str(error))
    save(out,result);print(json.dumps(result,indent=2));return 0 if result['status']=='complete' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
