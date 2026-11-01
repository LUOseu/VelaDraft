"""Prepare a diagnostic-only K sweep without changing final checkpoint tensors."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TASK = Path('/home/lyh/qwen36_k_sweep_20261101')
REPO = TASK / 'repo'
ADAPTER = REPO / 'research/qwen36_native_vllm'
sys.path.insert(0, str(ADAPTER))
from build_mmspec_source import sharegpt, TOPICS

CHECKPOINTS = {
    'dflash2': ('3c4442f21fd57fd815c6ae07085367af8d81c41a', '2bd9a036589981ef60760b2e3b64fb392ca01e3e1d8986d8645a66109fb44c4d'),
    'dfly': ('566ec8b7f50cc38b14fe6bc597c78e39649b3cb5', '2aac4b851bd576b7b5f4df78156c0845e6d9a56139ba6575424c1564e6b6330c'),
}

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def main():
    base = Path('/home/lyh/a6000_3b_v2_L300/shared/benchmarks/mmspec')
    assert digest(base / 'mmspec.jsonl') == '440028aaa95325919659675b1f7239a55fe348492b8ca4b691882bef56895c35'
    raw = [json.loads(x) for x in (base / 'mmspec.jsonl').read_text().splitlines()]
    chosen = []
    for topic in TOPICS:
        candidates = [(i, r) for i, r in enumerate(raw) if r['topic'] == topic and (topic != 'multi-turn conversation' or len(r['turns']) >= 2)]
        chosen.extend(candidates[:2])
    assert len(chosen) == 12
    data = [sharegpt(r, i) for i, r in chosen]
    source = TASK / 'diagnostic12.jsonl'
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data)
    if source.exists(): assert source.read_text() == text
    else: source.write_text(text)
    (TASK / 'dataset.json').write_text(json.dumps({'scope': 'diagnostic test subset; not held-out development; not paper-main data', 'selection': 'first two full-test sessions per topic; multi-turn requires >=2 turns; chosen before K results', 'ids': [r['id'] for r in data], 'sessions': 12, 'turns': sum(len(r['turns']) for r in data), 'source_sha256': digest(source)}, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', PYTHONPATH=str(REPO))
    for method, (revision, expected) in CHECKPOINTS.items():
        ckpt = TASK / 'checkpoints' / f'{method}_final.pt'
        assert digest(ckpt) == expected, method
        export = TASK / 'exports' / method
        if not (export / 'export_receipt.json').exists():
            subprocess.run([sys.executable, str(ADAPTER / 'export_checkpoint.py'), '--method', method, str(ckpt), str(export)], env=env, check=True)
        receipt = json.loads((export / 'export_receipt.json').read_text())
        assert receipt['source_sha256'] == expected and receipt['step'] == 75000 and receipt['all_tensors_bitwise_equal']
        for k in (7, 1, 3, 5):
            manifest = TASK / 'exports' / f'{method}_k{k}_runtime.json'
            if not manifest.exists():
                subprocess.run([sys.executable, str(ADAPTER / 'make_runtime_manifest.py'), '--method', method, '--checkpoint', str(ckpt), '--sweep-k', str(k), '--out', str(manifest)], env=env, check=True)
    (TASK / 'PREPARED.json').write_text(json.dumps({'status': 'prepared', 'methods': list(CHECKPOINTS), 'k_order': [1, 3, 5, 7], 'sessions': 12, 'turns': sum(len(r['turns']) for r in data), 'repeats': 1, 'target_tp': 2, 'max_new_tokens': 1024, 'diagnostic_only': True}, indent=2))
    print('PREPARED', flush=True)

if __name__ == '__main__': main()
