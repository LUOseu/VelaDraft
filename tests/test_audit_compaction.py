import torch

def test_batch_selection_frozen_from_supervision_not_timing():
    from scripts.npu.audit_compaction import select_batches
    records=[]
    for i in range(64):
        count=12+i*3
        records.append({'input_ids':torch.zeros(300,dtype=torch.long),'loss_mask':torch.cat([torch.zeros(300-count),torch.ones(count)])})
    plans=select_batches(records)
    assert len(plans)==8 and all(len(p)==8 for p in plans)
    assert plans==select_batches(records)
    assert len(set(plans[0]))==8
    assert max(int(records[i]['loss_mask'].sum()) for i in plans[1])-min(int(records[i]['loss_mask'].sum()) for i in plans[1])>100

def test_parity_observer_tracks_teacher_mapping_and_gradients():
    from scripts.npu.audit_compaction import forward_backward
    from tests.test_recipe_row_compaction import _tiny_module, _batch
    module=_tiny_module(stage=1)
    ids,mask,hidden,teacher=_batch()
    torch.manual_seed(42)
    anchors,keep=module._sample_anchor_positions(ids.shape[1],mask,torch.device('cpu'))
    batch={'input_ids':ids,'loss_mask':mask,'hidden_states':hidden,'teacher_hidden':teacher,'anchor_table':(anchors,keep)}
    off,g0,m0=forward_backward(module,batch,False,True)
    on,g1,m1=forward_backward(module,batch,True,True)
    assert off['loss']==on['loss']
    assert all(torch.equal(g0[k],g1[k]) for k in g0)
    assert torch.equal(m0[0].reshape(-1,7,2)[keep.reshape(-1)].reshape(-1,2),m1[0])
    assert on['forward_head_rows'][0]['rows']<off['forward_head_rows'][0]['rows']

def test_stale_parity_binding_is_rejected():
    import pytest
    from scripts.npu.audit_compaction import require_parity
    good={'status':'PASS','binding':{'checkpoint':'a','bundle':'b','plan':'c'}}
    require_parity(good,good['binding'])
    with pytest.raises(ValueError,match='binding'):
        require_parity(good,{'checkpoint':'changed','bundle':'b','plan':'c'})

def test_tree_digest_covers_rng_and_optimizer_fields():
    from scripts.npu.audit_compaction import tree_digest
    original={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.ones(2)}},'rng':torch.tensor([1,2],dtype=torch.uint8)}
    import copy
    other=copy.deepcopy(original)
    assert tree_digest(original)==tree_digest(other)
    other['rng'][0]=2
    assert tree_digest(original)!=tree_digest(other)


def test_directory_binding_covers_feature_and_head_bytes(tmp_path):
    from scripts.npu.audit_compaction import source_bundle_digest
    features = tmp_path / 'features'
    features.mkdir()
    (tmp_path / 'contract.json').write_text('{}')
    (features / 'manifest.json').write_text('{"fixed": true}')
    shard = features / 'shard_000.pt'
    head = features / 'target_head.pt'
    shard.write_bytes(b'original feature values')
    head.write_bytes(b'original teacher head')
    initial = source_bundle_digest(tmp_path)
    shard.write_bytes(b'changed feature values')
    changed_features = source_bundle_digest(tmp_path)
    assert changed_features != initial
    head.write_bytes(b'changed teacher head')
    assert source_bundle_digest(tmp_path) != changed_features
    assert (features / 'manifest.json').read_text() == '{"fixed": true}'


def test_file_bundle_binding_keeps_original_full_file_hash(tmp_path):
    from scripts.npu.audit_compaction import source_bundle_digest, file_digest
    bundle = tmp_path / 'bundle.pt'
    bundle.write_bytes(b'unchanged full bundle bytes')
    assert source_bundle_digest(bundle) == file_digest(bundle)
