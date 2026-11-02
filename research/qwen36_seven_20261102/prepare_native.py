"""Map dispatch metadata to public vLLM 0.32 models; share immutable weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def native_config(source, method):
    result = copy.deepcopy(source)
    draft = result['dflash_config']
    assert draft['spatial_rope_enabled'] is True
    assert result['hidden_size'] == 2048 and result['num_hidden_layers'] == 3
    if method == 'dflash2':
        expected = dict(conv_kernel_size=2, conv_group_size=16,
                        selector_rank=256, selector_top_k=16)
        assert draft['dflash2'] == expected
        for key, value in expected.items():
            assert key not in draft or draft[key] == value
            draft[key] = value
        result['architectures'] = ['DFlash2DraftModel']
    elif method == 'jetspec':
        assert draft['inference_in_block_causal'] is True
        assert not draft.get('sample_from_anchor')
        assert result.get('is_causal', True) is True
        result['architectures'] = ['DFlashDraftModel']
        # Native per-layer policy: context remains visible; query block is causal.
        result['is_causal'] = True
    else:
        raise ValueError(method)
    return result


def prepare(method, source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    old = json.loads((source/'config.json').read_text())
    config = native_config(old, method)
    (destination/'config.json').write_text(json.dumps(config, indent=2))
    os.link(source/'model.safetensors', destination/'model.safetensors')
    assert os.path.samefile(source/'model.safetensors', destination/'model.safetensors')
    receipt = dict(method=method, source_export=str(source),
                   same_weight_inode=True, source_config=old,
                   effective_config=config,
                   config_sha256=hashlib.sha256((destination/'config.json').read_bytes()).hexdigest(),
                   source_export_receipt=json.loads((source/'export_receipt.json').read_text()),
                   note='Only native dispatch/config-key mapping; all weight bytes unchanged.')
    (destination/'native_receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('dflash2', 'jetspec'), required=True)
    parser.add_argument('source'); parser.add_argument('destination')
    args = parser.parse_args()
    prepare(args.method, args.source, args.destination)
    print('NATIVE_CONFIG_PREPARED', args.method, args.destination)
