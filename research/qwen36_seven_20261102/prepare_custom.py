"""Architecture dispatch only for the remaining trained VelaDraft modules."""
import json
import hashlib
import os
from pathlib import Path
import sys


def prepare(method):
    root=Path('/root/autodl-tmp/qwen36_seven_20261102')
    src=root/'exports_raw'/method;dst=root/'exports'/method
    config=json.loads((src/'config.json').read_text())
    before=json.loads(json.dumps(config))
    names=dict(domino='DFlashVelaDominoModel',dflare='DFlashVelaDFlareModel',dfly='DFlashVelaDFlyModel')
    assert method in names and config['dflash_config']['spatial_rope_enabled'] is True
    config['architectures']=[names[method]]
    config['veladraft_custom_method']=method
    if method=='domino':config['veladraft_domino_reference']=True
    dst.mkdir(parents=True,exist_ok=False)
    (dst/'config.json').write_text(json.dumps(config,indent=2))
    os.link(src/'model.safetensors',dst/'model.safetensors')
    assert os.path.samefile(src/'model.safetensors',dst/'model.safetensors')
    receipt=dict(method=method,same_weight_inode=True,source_config=before,effective_config=config,
                 source_export_receipt=json.loads((src/'export_receipt.json').read_text()),
                 config_sha256=hashlib.sha256((dst/'config.json').read_bytes()).hexdigest())
    (dst/'native_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('CUSTOM_CONFIG_PREPARED',method,flush=True)


if __name__=='__main__':prepare(sys.argv[1])
