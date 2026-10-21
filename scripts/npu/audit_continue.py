#!/usr/bin/env python3
"""NPU weight-only continuation of an existing R3 DFly asset, then export.

Fresh FP32-master AdamW, fixed stage2; this is not an optimizer-state resume or
from-scratch training. Creates ordinary production checkpoint/export formats.
"""
from __future__ import annotations
import argparse,copy,json,math,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.npu.audit_common import load_bundle,build_module,make_batch,create_optimizer,optimizer_step,digest_tensors
from scripts.npu.audit_compaction import configuration,file_digest,write
from scripts.npu.audit_recipes import RecipeObserver,source_provenance
from vlm_dflash import runtime
from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.export.hf import export_run
from vlm_dflash.provenance import identity_from_contract


def continue_and_export(bundle,cfg,checkpoint,out,device,steps=10):
    out=Path(out);(out/'train').mkdir(parents=True,exist_ok=True);(out/'data').mkdir(exist_ok=True)
    cfg=copy.deepcopy(cfg);cfg['run']={'dir':str(out),'name':'npu_weight_only_continuation'}
    cfg['train'].update(lr=1e-5,micro_batch_size=1,grad_accumulation_steps=1)
    cfg['algorithm']['train_recipe']={'name':'upstream_two_stage','cold_start_fraction':.5}
    module,geometry=build_module(bundle,cfg,device,checkpoint=checkpoint)
    if module.train_recipe.name!='upstream_two_stage':raise ValueError('formal recipe required')
    module.set_training_progress(75,100)
    masters,opt,sched=create_optimizer(module,cfg,100)
    initial=digest_tensors(module.draft_model.state_dict());history=[]
    runtime.reset_peak_memory_stats(device)
    started=time.monotonic()
    with RecipeObserver(module,'dfly') as observer:
        for step in range(steps):
            module.set_training_progress(75+step,100)
            batch=make_batch(bundle,[step%len(bundle['records'])],device,next(module.draft_model.parameters()).dtype)
            module.zero_grad(set_to_none=True)
            metrics=module(**batch);metrics.loss.backward()
            norm=optimizer_step(module,masters,opt,sched,1.0)
            if not math.isfinite(norm):
                raise FloatingPointError(f'Non-finite gradient norm at continuation step {step+1}: {norm}')
            history.append({'step':step+1,'loss':float(metrics.loss.detach()),'valid_tokens':int(metrics.valid_tokens),'grad_norm':norm})
            write(out/'train/train_summary.json',{'status':'RUNNING','steps':len(history),'scope':'weight_only_continuation'})
        observer.validate({1})
    runtime.synchronize(device)
    for name,value in module.draft_model.state_dict().items():
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Continuation final state has non-finite parameter or buffer: {name}')
    final=digest_tensors(module.draft_model.state_dict())
    assert initial!=final,'no parameter updates'
    summary={'status':'PASS','steps':steps,'final_loss':history[-1]['loss'],'nan_seen':False,'device':str(device),
      'device_type':torch.device(device).type,'backend':torch.device(device).type,'feature_mode':'offline',
      'wall_seconds':time.monotonic()-started,'peak_accelerator_mem_gb':None if runtime.max_memory_allocated(device) is None else runtime.max_memory_allocated(device)/1e9,
      'scope':'pretrained R3 weight-only continuation, fresh FP32 masters/optimizer, objective stage2; no cross-backend bitwise claim',
      'train_recipe':module.train_recipe.as_dict(),'objective_progress_start':75,'horizon':100,
      'all_gradient_norms_finite':True,'final_state_finite':True,
      'initial_weight_sha256':initial,'final_weight_sha256':final,'source_checkpoint_sha256':file_digest(checkpoint),
      'observation':observer.report(),**source_provenance(ROOT)}
    contract=bundle['contract'];algorithm=get_algorithm('dfly')
    payload={'format':'vlm-dflash-ckpt-v1','algorithm':'dfly','identity':identity_from_contract(contract),'step':steps,
             'feature_mode':'offline','model':{k:v.detach().cpu() for k,v in module.draft_model.state_dict().items()},
             'attention_backend':'sdpa_chunked',**algorithm.checkpoint_metadata(contract,cfg,geometry)}
    torch.save(payload,out/'train/ckpt_final.pt')
    write(out/'config.json',cfg);write(out/'contract.json',contract)
    write(out/'data/manifest.json',{'source':'fixed_natural_training_subset','num_train':len(bundle['records']),'num_val':64,'note':'existing natural corpus; teacher features reused'})
    write(out/'train/train_summary.json',summary)
    (out/'train/train_log.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history))
    export_run(out)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='npu:0')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        device=str(runtime.ensure_runtime_device(a.device));runtime.seed_all(42,device);torch.use_deterministic_algorithms(True)
        bundle=load_bundle(a.bundle);cfg=configuration(bundle,device)
        result=continue_and_export(bundle,cfg,a.checkpoint,a.out,device)
        write(a.out/'result.json',result);return 0
    except Exception as e:
        write(a.out/'result.json',{'status':'FAILED','error':str(e),'traceback':traceback.format_exc()});return 1

if __name__=='__main__':raise SystemExit(main())
