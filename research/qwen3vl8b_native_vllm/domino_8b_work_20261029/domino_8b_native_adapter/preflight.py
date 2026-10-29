"""Logic-only CPU receipt now; explicit deferred full-byte gate for later review."""
from __future__ import annotations
import argparse,contextlib,hashlib,io,json,os,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
if 'contracts' in sys.modules and Path(sys.modules['contracts'].__file__).resolve()!=ROOT/'contracts.py':raise RuntimeError('Foreign contracts module')
from contracts import OWN_FILES,CHECKPOINT_SHA,CHECKPOINT_PATH,TARGET_ROOT,own_source_hashes,verify_dependencies,validate_config,validate_state,validate_export_layout,validate_target_layout,validate_full_receipt,runtime_contract,_stamp

def run(output,*,full=False,allow_large_io=False,export_path=None):
    if full and not allow_large_io:raise RuntimeError('Full preflight large I/O requires later explicit root authorization')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('OMP_NUM_THREADS')!='1' or os.environ.get('MKL_NUM_THREADS')!='1':raise RuntimeError('CPU hidden CUDA and OMP/MKL1 required')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier receipts; choose a new output')
    before=own_source_hashes();dependencies=verify_dependencies()
    import torch
    import vllm
    import domino_model,domino_proposer
    from install import check_runtime
    from export_domino import sha256,verify_tensor_values
    check_runtime()
    # Import actual classes first, then forbid any model/LLM construction or CUDA init.
    LLM=vllm.LLM
    def forbid(*args,**kwargs):raise AssertionError('CPU preflight forbids model/LLM/CUDA construction')
    log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['test_domino','test_interface','test_runtime_approval'])
    with patch.object(torch.nn.Module,'__init__',forbid),patch.object(LLM,'__init__',forbid),patch.object(torch.cuda,'_lazy_init',forbid):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not result.wasSuccessful():
        output.with_suffix('.failed.log').write_text(log.getvalue())
        raise RuntimeError(log.getvalue())
    if result.skipped:raise RuntimeError('Full pinned CPU suite must execute without skipped checks')
    if torch.cuda.is_initialized():raise AssertionError('CUDA initialized')
    receipt={'schema':'domino-logic-receipt-v2','classification':'logic_only','gpu_eligible':False,'requires_independent_review':True,'created_unix':time.time(),'cpu_only':True,'cuda_initialized':False,'model_constructed':False,'LLM_constructed':False,'tests_run':result.testsRun,'tests_passed':result.testsRun,'test_output':log.getvalue(),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','actual_public_config_and_class_interface_checks':True,'constructor_proof':'explicit non-promotable CPU logic interface and official proposer constructor shim only; actual model/backbone creation blocked before Module initialization','math_proof':'tiny synthetic FP32/BF16 tensors and exact canonical AST-extracted reference; no actual checkpoint or draft forward','real_checkpoint_contents_read':False,'target_weight_bodies_read':False,'dependency_binary_bodies_rehashed':False,'export_created':False,'forward_executed':False,'target_embedding_head_alias_evidence':'inherited pinned share-method/source path only; actual model object aliases not validated','recorded_inventory_sha256':json.loads((ROOT/'source_bindings.json').read_text())['reference_inventory_recorded_sha256'],'production_CPU_escape_present':False,'dependency_sources_verified':dependencies,'own_sources':before,'unvalidated':['real40-value export/reload','full current checkpoint/target/binary bytes','actual model construction/loading','actual draft forward/graph/cache/mask/GPU proposals','target token compatibility','performance']}
    receipt.update(tests_skipped=0,source_root=str(ROOT),receipt_path=str(output),runtime_contract=runtime_contract())
    if full:
        if export_path is None:raise ValueError('Full preflight requires future actual export path')
        binds=json.loads((ROOT/'source_bindings.json').read_text());artifacts=binds['deferred_full_artifacts']
        validate_target_layout(TARGET_ROOT,artifacts)
        from vllm.config import LoadConfig
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        target_loader_selection={}
        expected_target_shards=sorted(str(Path(TARGET_ROOT)/name) for name in ('model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors'))
        for mode in ('auto','safetensors'):
            loader=DefaultModelLoader(LoadConfig(load_format=mode))
            folder,files,safe=loader._prepare_weights(TARGET_ROOT,None,None,True,None)
            if str(Path(folder).resolve())!=TARGET_ROOT or safe is not True or sorted(files)!=expected_target_shards:raise ValueError('Actual target loader selected unapproved weight files')
            target_loader_selection[mode]=sorted(files)
        initial_stamps={name:_stamp(name) for name in artifacts}
        for name,expected in artifacts.items():
            if sha256(name)!=expected:raise ValueError('Full bytes changed: '+name)
        export_path=Path(export_path).resolve();validate_export_layout(export_path)
        initial_stamps.update({str(export_path/name):_stamp(export_path/name) for name in ('config.json','model.safetensors','export_receipt.json')})
        config=json.loads((export_path/'config.json').read_text());validate_config(config)
        export_receipt=json.loads((export_path/'export_receipt.json').read_text())
        if export_receipt.get('schema')!='domino-export-v1' or export_receipt.get('source_sha256')!=CHECKPOINT_SHA or export_receipt.get('all40_values_bitwise_equal') is not True or export_receipt.get('tensor_count')!=40 or export_receipt.get('cuda_initialized') is not False or export_receipt.get('model_constructed') is not False:raise ValueError('Unbound export receipt')
        for name,field in (('config.json','config_sha256'),('model.safetensors','weights_sha256')):
            if sha256(export_path/name)!=export_receipt.get(field):raise ValueError('Export bytes changed')
        source_path=Path(CHECKPOINT_PATH)
        payload=torch.load(source_path,map_location='cpu',mmap=True,weights_only=False)
        if payload.get('algorithm')!='domino' or payload.get('step')!=75000:raise ValueError('Wrong source checkpoint metadata')
        validate_config(payload['draft_config'],source=True)
        from safetensors.torch import load_file
        restored=load_file(str(export_path/'model.safetensors'),device='cpu')
        validate_state(payload['model']);validate_state(restored)
        tensors=verify_tensor_values(payload['model'],restored)
        if tensors!=export_receipt.get('tensors'):raise ValueError('Tensor receipts differ')
        full_artifacts=dict(artifacts)
        full_artifacts.update({str(export_path/name):sha256(export_path/name) for name in ('config.json','model.safetensors','export_receipt.json')})
        if any(_stamp(name)!=stamp for name,stamp in initial_stamps.items()):raise RuntimeError('Full bytes changed during CPU proof')
        validate_target_layout(TARGET_ROOT,artifacts)
        receipt.update(target_model_path=TARGET_ROOT,draft_model_path=str(export_path),schema='domino-flash-full-preflight-v1',classification='full_export_and_bytes_verified',all40_values_bitwise_equal=True,real_checkpoint_contents_read=True,target_weight_bodies_read=True,dependency_binary_bodies_rehashed=True,full_artifacts=full_artifacts,source_root=str(ROOT),receipt_path=str(output),runtime_contract=runtime_contract(),checkpoint_path=CHECKPOINT_PATH,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_step=75000,tensor_count=40,tensors=tensors)
        receipt.update(tests_skipped=0,target_weight_layout_validated=True,target_loader_selection=target_loader_selection,math_proof='Tiny canonical FP32/BF16 math tests; all40 actual parent/export tensor bytes separately compared; no actual model forward',unvalidated=['actual model construction/loading','actual draft forward/graph/cache/mask/GPU proposals','target token compatibility','performance'])
        validate_full_receipt(receipt)
        # Still false: independent review and an explicit root-approved receipt are required.
    if before!=own_source_hashes():raise RuntimeError('Tested sources changed during preflight')
    if verify_dependencies()!=dependencies:raise RuntimeError('Pinned dependencies changed during preflight')
    if torch.cuda.is_initialized():raise AssertionError('CUDA initialized')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'CPU_LOGIC_RECEIPT.json');p.add_argument('--full',action='store_true');p.add_argument('--allow-large-io',action='store_true');p.add_argument('--export',type=Path);a=p.parse_args()
    r=run(a.output,full=a.full,allow_large_io=a.allow_large_io,export_path=a.export)
    print(json.dumps({k:r[k] for k in ('schema','classification','gpu_eligible','tests_run','cuda_initialized','model_constructed','real_checkpoint_contents_read','target_weight_bodies_read')}))
