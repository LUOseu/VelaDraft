"""Dispatch-only mapping for the trained anchor-slot DSpark variant."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def native_config(source):
    result=copy.deepcopy(source);draft=result['dflash_config']
    expected=dict(markov_rank=256,markov_head_type='vanilla',
                  enable_confidence_head=True,confidence_head_with_markov=True,
                  sample_from_anchor=True)
    assert all(draft.get(k)==v for k,v in expected.items())
    assert draft['spatial_rope_enabled'] is True
    assert result['hidden_size']==2048 and result['num_hidden_layers']==3
    for k,v in expected.items():
        if k!='sample_from_anchor':
            assert k not in result or result[k]==v
            result[k]=v
    # Native DFlash safely allocates K sample slots for K+1 query slots.
    # The bridge changes only the hidden rows supplied to the native Markov walk.
    del draft['sample_from_anchor']
    result['veladraft_anchor_slot_predictions']=True
    result['architectures']=['DFlashVelaDSparkModel']
    return result


def prepare(source,destination):
    source,destination=Path(source),Path(destination)
    old=json.loads((source/'config.json').read_text());config=native_config(old)
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'config.json').write_text(json.dumps(config,indent=2))
    os.link(source/'model.safetensors',destination/'model.safetensors')
    assert os.path.samefile(source/'model.safetensors',destination/'model.safetensors')
    receipt=dict(method='dspark',source_export=str(source),same_weight_inode=True,
        source_config=old,effective_config=config,
        config_sha256=hashlib.sha256((destination/'config.json').read_bytes()).hexdigest(),
        source_export_receipt=json.loads((source/'export_receipt.json').read_text()),
        scope='VelaDraft K+1 queries, anchor predicts first token; public Qwen3 DSpark backbone and Markov walk, original BF16 bias-add precision.')
    (destination/'native_receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    prepare(sys.argv[1],sys.argv[2]);print('DSPARK_NATIVE_CONFIG_PREPARED',flush=True)
