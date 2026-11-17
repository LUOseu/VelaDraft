"""Architecture dispatch only for the remaining trained VelaDraft modules."""
import json
import hashlib
import os
from pathlib import Path
import sys


def prepare(method,src=None,dst=None):
    if src is None or dst is None:
        raise ValueError("Provide explicit source and destination export directories")
    src,dst=Path(src),Path(dst)
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


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=('domino','dflare','dfly'))
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    prepare(args.method,args.source,args.destination)
