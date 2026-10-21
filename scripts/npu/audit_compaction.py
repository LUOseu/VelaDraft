#!/usr/bin/env python3
"""Fixed natural-data DFly stage2 compaction contrast; no shape/performance search.

Use --phase parity before --phase benchmark. Raw per-update measurements include
forward/backward/update synchronization, projection rows and effective work.
"""
from __future__ import annotations
import argparse, copy, gc, hashlib, json, math, sys, time, traceback
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.npu.audit_common import load_bundle,build_module,make_batch,create_optimizer,optimizer_step,digest_tensors
from vlm_dflash import runtime
from vlm_dflash.objectives import TeacherLogitsStream


def select_batches(records):
    if len(records)<64:raise ValueError('Fixed protocol requires 64 source records')
    order=sorted(range(len(records)),key=lambda i:(float(records[i]['loss_mask'].sum()),i))
    windows=[order[i:i+8] for i in range(len(order)-7)]
    def padding(group):
        counts=[float(records[i]['loss_mask'].sum()) for i in group]
        return 1-sum(counts)/(8*max(counts))
    low=min(windows,key=lambda group:(padding(group),group))
    high=[item for pair in zip(order[:4],reversed(order[-4:])) for item in pair]
    return [low,high]+[order[i:i+8] for i in range(0,48,8)]


def tree_digest(value):
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda k:str(k)):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()



def source_bundle_digest(path):
    """Bind actual teacher/head/features, not merely descriptive metadata.

    File bundles retain their original full-file SHA256. Directory stores bind
    the contract and every file under features (manifest, index, head, shards).
    Hash files incrementally to avoid allocating the whole store in host RAM.
    """
    path=Path(path)
    if path.is_file():
        return file_digest(path)
    features=path/'features'
    if not features.is_dir():
        raise FileNotFoundError(f'Feature store missing: {features}')
    files=[path/'contract.json']+sorted(p for p in features.rglob('*') if p.is_file())
    return tree_digest({str(p.relative_to(path)):file_digest(p) for p in files})


def require_parity(payload,binding):
    if payload.get('status')!='PASS':raise ValueError('parity gate required')
    if payload.get('binding')!=binding:raise ValueError('stale parity binding; rerun parity for these inputs and source')


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=str)+'\n')


def configuration(bundle,device):
    cfg=yaml.safe_load((ROOT/'configs/dfly/qwen2_5_vl_3b_2k_recipe_matched.yaml').read_text())
    cfg['target']['model']=bundle['contract']['target']['model_path']
    cfg['target']['device']=device
    cfg['drafter'].update(attention_backend='sdpa_chunked',attention_chunk_size=128,
        capture_layer_ids=bundle['manifest']['capture_layer_ids'],recipe_compile_tv_kl=False)
    cfg['train'].update(device=device,draft_dtype='bfloat16',micro_batch_size=8,
        grad_accumulation_steps=1,feature_mode='offline',optimizer_foreach=False)
    return cfg


def plans_for(module,bundle,groups):
    result=[]
    for i,group in enumerate(groups):
        batch=make_batch(bundle,group,'cpu',torch.bfloat16)
        torch.manual_seed(20261021+i)
        anchors,keep=module._sample_anchor_positions(batch['input_ids'].shape[1],batch['loss_mask'],torch.device('cpu'))
        n=anchors.shape[1]
        result.append({'indices':group,'anchors':anchors,'keep':keep,'padding_fraction':1-float(keep.sum())/keep.numel(),
                       'context':batch['input_ids'].shape[1],'requested_anchors':256,'actual_anchors':n,
                       'query_rows_per_sample':n*8,'allocated_rows':len(group)*n*8,
                       'kept_rows':int(keep.sum())*8})
    if not any(p['padding_fraction']>0 for p in result) or not any(p['padding_fraction']<=.1 for p in result):
        raise ValueError('Selection must include nonzero padding and <=10% padding; do not alter masks')
    return result


def get_batch(bundle,plan,device):
    batch=make_batch(bundle,plan['indices'],device,torch.bfloat16)
    batch['anchor_table']=(plan['anchors'].to(device),plan['keep'].to(device))
    return batch


class Observer:
    def __init__(self,module,mapping=False):self.module=module;self.mapping=mapping;self.rows=[];self.maps=[]
    def __enter__(self):
        def head(module,args):self.rows.append({'rows':args[0].numel()//args[0].shape[-1],'student':bool(args[0].requires_grad)})
        self.handle=self.module.lm_head.register_forward_pre_hook(head)
        self.original=TeacherLogitsStream.__init__
        self.masks=[];self.weights=[]
        if self.mapping:
            from vlm_dflash import objectives
            from vlm_dflash.dflash import trainer_core
            self.original_recipe=trainer_core.recipe_loss
            self.original_weights=objectives.dpace_position_weights
            def recipe(stage,**kwargs):
                self.masks.append(kwargs['block_mask'].detach().cpu().clone())
                return self.original_recipe(stage,**kwargs)
            def weights(*args,**kwargs):
                value=self.original_weights(*args,**kwargs)
                self.weights.append(value.detach().cpu().clone())
                return value
            trainer_core.recipe_loss=recipe;objectives.dpace_position_weights=weights
        if self.mapping:
            obs=self
            def init(stream,*args,**kwargs):
                obs.original(stream,*args,**kwargs)
                obs.maps.append(torch.stack((stream._batch,stream._positions),dim=-1).cpu())
            TeacherLogitsStream.__init__=init
        return self
    def __exit__(self,*args):
        self.handle.remove()
        if self.mapping:
            from vlm_dflash import objectives
            from vlm_dflash.dflash import trainer_core
            TeacherLogitsStream.__init__=self.original
            trainer_core.recipe_loss=self.original_recipe
            objectives.dpace_position_weights=self.original_weights


def forward_backward(module,batch,compacted,mapping=False):
    module.recipe_row_compaction=compacted
    module.set_training_progress(75,100)
    module.zero_grad(set_to_none=True)
    with Observer(module,mapping) as obs:
        metrics=module(**batch)
        forward_rows=copy.deepcopy(obs.rows)
        metrics.loss.backward()
    grads={n:p.grad.detach().cpu().clone() for n,p in module.draft_model.named_parameters() if p.grad is not None}
    result={'loss':float(metrics.loss.detach()),'valid_tokens':int(metrics.valid_tokens),'loss_weight':float(metrics.loss_weight),
            'forward_head_rows':forward_rows,'all_head_rows':obs.rows,
            'extra_metrics':{k:float(v) for k,v in (metrics.extra_metrics or {}).items()}}
    assert result['valid_tokens']>0 and result['loss_weight']>0 and math.isfinite(result['loss'])
    assert result['extra_metrics'].get('recipe_stage')==1
    assert 'e2e_tv_loss' in result['extra_metrics'] and 'ce_loss' in result['extra_metrics']
    if mapping:
        assert len(obs.masks)==len(obs.weights)==1
        obs.maps.extend([obs.masks[0],obs.masks[0]*obs.weights[0]])
    return result,grads,obs.maps


def run_parity(module,bundle,plans,out,binding):
    results=[]
    for i,plan in enumerate(plans):
        batch=get_batch(bundle,plan,str(next(module.parameters()).device))
        rng=runtime.capture_rng_state()
        off,g0,m0=forward_backward(module,batch,False,True)
        runtime.restore_rng_state(rng)
        on,g1,m1=forward_backward(module,batch,True,True)
        assert g0.keys()==g1.keys() and len(m0)==len(m1)==3
        expected=m0[0].reshape(-1,7,2)[plan['keep'].reshape(-1)].reshape(-1,2)
        mapping_equal=torch.equal(expected,m1[0])
        expected_mask=m0[1].reshape(-1,7)[plan['keep'].reshape(-1)].reshape(-1)
        expected_weight=m0[2].reshape(-1,7)[plan['keep'].reshape(-1)].reshape(-1)
        mask_equal=torch.equal(expected_mask,m1[1].reshape(-1))
        weight_delta=(expected_weight-m1[2].reshape(-1)).abs()
        weights_close=bool(torch.all(weight_delta<=.02+.02*expected_weight.abs()))
        errors={}
        for name in g0:
            a,b=g0[name].float(),g1[name].float();delta=(a-b).abs()
            errors[name]={'max_abs':float(delta.max()),'relative_l2':float(delta.norm()/a.norm().clamp_min(1e-30)),
                'bitwise':torch.equal(g0[name],g1[name]),'bf16_envelope':bool(torch.all(delta<=.02+.02*a.abs()))}
        strict_loss=abs(off['loss']-on['loss'])<=1e-12+1e-9*abs(off['loss'])
        bf16_loss=abs(off['loss']-on['loss'])<=.02+.02*abs(off['loss'])
        case={'batch':i,'off':off,'on':on,'teacher_mapping_equal':mapping_equal,'mask_equal':mask_equal,'position_weights_max_abs_error':float(weight_delta.max()),'position_weights_bf16_envelope':weights_close,
              'gradient_errors':errors,'strict_bitwise_contract_pass':strict_loss and all(x['bitwise'] for x in errors.values()),
              'bf16_envelope_pass':bf16_loss and all(x['bf16_envelope'] for x in errors.values()),
              'off_gradient_digest':digest_tensors(g0),'on_gradient_digest':digest_tensors(g1)}
        case['passed']=case['bf16_envelope_pass'] and mapping_equal and mask_equal and weights_close and off['valid_tokens']==on['valid_tokens'] and off['loss_weight']==on['loss_weight']
        results.append(case);write(out/'parity.json',{'status':'PASS' if all(x['passed'] for x in results) and len(results)==8 else 'INCOMPLETE_OR_FAILED','binding':binding,'cases':results,'tolerance':{'bf16_rtol':.02,'bf16_atol':.02,'source':'pre-registered existing H1 BF16 envelope; strict CPU checks reported separately'}})
        del batch,g0,g1;gc.collect()
    if not all(x['passed'] for x in results):raise RuntimeError('Fixed numerical gate failed; no performance claim permitted')


def run_leg(bundle,cfg,checkpoint,plans,device,seed,compacted,updates,warmup):
    runtime.seed_all(seed,device)
    module,geometry=build_module(bundle,cfg,device,seed=seed,checkpoint=checkpoint)
    module.recipe_row_compaction=compacted
    masters,opt,sched=create_optimizer(module,cfg,100)
    initial=digest_tensors(module.draft_model.state_dict())
    initial_states={k:tree_digest(v) for k,v in {'model':module.draft_model.state_dict(),'master':masters,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':runtime.capture_rng_state()}.items()}
    runtime.reset_peak_memory_stats(device)
    rows=[]
    for step in range(updates):
        if step==warmup:
            module.reset_attention_runtime_stats();runtime.reset_peak_memory_stats(device)
        plan=plans[step%len(plans)];batch=get_batch(bundle,plan,device)
        module.set_training_progress(75+step,100);module.zero_grad(set_to_none=True)
        with Observer(module) as obs:
            runtime.synchronize(device);t0=time.perf_counter()
            metrics=module(**batch)
            runtime.synchronize(device);t1=time.perf_counter()
            forward_rows=copy.deepcopy(obs.rows)
            metrics.loss.backward()
            runtime.synchronize(device);t2=time.perf_counter()
            norm=optimizer_step(module,masters,opt,sched,1.0)
            runtime.synchronize(device);t3=time.perf_counter()
        assert torch.isfinite(metrics.loss) and math.isfinite(norm)
        rows.append({'step':step,'measured':step>=warmup,'batch':step%len(plans),'loss':float(metrics.loss.detach()),
          'valid_tokens':int(metrics.valid_tokens),'loss_weight':float(metrics.loss_weight),'grad_norm':norm,
          'forward_seconds':t1-t0,'backward_seconds':t2-t1,'update_seconds':t3-t2,'wall_seconds':t3-t0,
          'student_projected_rows_forward':sum(r['rows'] for r in forward_rows if r['student']),
          'teacher_projected_rows_forward':sum(r['rows'] for r in forward_rows if not r['student']),
          'projected_rows_including_backward_recomputation':sum(r['rows'] for r in obs.rows),
          'allocated_rows':plan['allocated_rows'],'kept_rows':plan['kept_rows'],'padding_fraction':plan['padding_fraction']})
        del batch,metrics
    result={'seed':seed,'compaction':compacted,'initial_sha256':initial,'initial_state_hashes':initial_states,'optimizer':'production FP32-master AdamW',
       'optimizer_horizon':100,'optimizer_start_step':0,'objective_progress_start':75,
       'steps':rows,'peak_hbm_bytes':runtime.max_memory_allocated(device),'attention':module.attention_runtime_report(),
       'model_geometry':{k:getattr(geometry,k) for k in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim')}}
    del module,masters,opt,sched;gc.collect()
    getattr(torch,torch.device(device).type).empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='npu:0')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['parity','benchmark','all'],default='all')
    p.add_argument('--target',default=None);p.add_argument('--max-seconds',type=float,default=7200)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);started=time.monotonic();status={'status':'FAILED'}
    try:
        device=str(runtime.ensure_runtime_device(a.device));torch.use_deterministic_algorithms(True)
        bundle=load_bundle(a.bundle)
        if a.target:bundle['contract']['target']['model_path']=a.target
        cfg=configuration(bundle,device);module,geometry=build_module(bundle,cfg,device,checkpoint=a.checkpoint)
        assert (geometry.hidden_size,geometry.intermediate_size,geometry.num_hidden_layers)==(2048,11008,3)
        groups=select_batches(bundle['records']);plans=plans_for(module,bundle,groups)
        torch.save(plans,a.out/'anchor_plan.pt');write(a.out/'config.json',cfg)
        write(a.out/'batches.json',[{k:v for k,v in r.items() if k not in ('anchors','keep')} for r in plans])
        from scripts.npu.audit_recipes import source_provenance
        binding={'checkpoint':file_digest(a.checkpoint),'bundle':source_bundle_digest(a.bundle),
                 'config':tree_digest(cfg),'plan':tree_digest(plans),'source':tree_digest(source_provenance(ROOT)['source_hashes'])}
        write(a.out/'binding.json',binding)
        if a.phase in ('parity','all'):run_parity(module,bundle,plans,a.out,binding)
        del module;gc.collect();getattr(torch,torch.device(device).type).empty_cache()
        if a.phase in ('benchmark','all'):
            require_parity(json.loads((a.out/'parity.json').read_text()),binding)
            pilot=run_leg(bundle,cfg,a.checkpoint,plans,device,42,False,5,0)
            estimate=sum(r['wall_seconds'] for r in pilot['steps'])/5*90
            write(a.out/'pilot.json',{'leg':pilot,'estimated_90_update_seconds':estimate})
            if time.monotonic()-started+estimate>a.max_seconds:raise TimeoutError('pilot predicts budget exceeded; do not shrink workload')
            for pair,seed in enumerate((42,43,44)):
                pair_initial=None
                for compacted in ((False,True) if pair%2==0 else (True,False)):
                    if time.monotonic()-started>a.max_seconds:raise TimeoutError('budget exhausted')
                    result=run_leg(bundle,cfg,a.checkpoint,plans,device,seed,compacted,15,5)
                    result['binding']=binding
                    if pair_initial is None:pair_initial=result['initial_state_hashes']
                    else:assert pair_initial==result['initial_state_hashes'],'paired initial state/RNG mismatch'
                    write(a.out/f'pair{pair}_{"on" if compacted else "off"}.json',result)
        status={'status':'PASS','phase':a.phase,'device':device,'config':cfg}
    except Exception as e:status.update(error=str(e),traceback=traceback.format_exc())
    status['elapsed_seconds']=time.monotonic()-started;write(a.out/'result.json',status)
    return 0 if status['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
