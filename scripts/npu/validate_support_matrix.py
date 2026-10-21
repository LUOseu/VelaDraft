#!/usr/bin/env python3
"""Fail-closed validation of context-rich audit claims in the existing matrix.

Artifact paths are relative to the matrix directory. A PASS binds to exactly one
profile_records entry in the SHA256-pinned JSON artifact; matching a global
summary alone is insufficient. This checks provenance consistency, not whether
an experiment's measurements are scientifically sufficient for a larger claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

IDENTITY = ('id','method','objective_profile','stage','backend','mode','status')
CONTEXT = ('dtype','model_geometry','shape','data_kind','samples','updates','verification_mode','source_revision')
GEOMETRY = ('num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','block_size')
SHAPE = ('context','anchors','query_rows','chunks','valid_supervision_rows','padding_fraction')
METHODS = {'dflash','dspark','domino','jetspec','dflash2','dflare','dfly'}


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def validate_matrix(matrix_path, expected_revision=None):
    path=Path(matrix_path)
    errors=[]
    try:
        matrix=json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        return [f'matrix unreadable: {exc}']
    rows=matrix.get('audit_profiles') if isinstance(matrix,dict) else None
    if not isinstance(rows,list) or not rows:
        return ['audit_profiles must be a nonempty list']
    seen=set()
    for index,row in enumerate(rows):
        label=f'profile[{index}]'
        def fail(message): errors.append(f'{label}: {message}')
        if not isinstance(row,dict):
            fail('profile must be an object');continue
        label=str(row.get('id',label))
        for key in IDENTITY:
            if not isinstance(row.get(key),str) or not row[key].strip(): fail(f'missing {key}')
        if label in seen: fail('duplicate profile id')
        seen.add(label)
        if row.get('method') not in METHODS: fail('unknown method')
        if row.get('backend') not in {'cpu','cuda','npu'}: fail('unknown backend')
        if row.get('status') not in {'PASS','PENDING','FAILED'}:
            fail('unknown status');continue
        if row['status'] != 'PASS':
            if not isinstance(row.get('reason'),str) or not row['reason'].strip(): fail('non-PASS needs reason')
            continue
        for key in CONTEXT:
            if key not in row: fail(f'PASS missing {key}')
        for key in ('dtype','data_kind','verification_mode'):
            if not isinstance(row.get(key),str) or not row[key].strip(): fail(f'invalid {key}')
        geometry=row.get('model_geometry')
        if not isinstance(geometry,dict): fail('model_geometry must be an object')
        else:
            for key in GEOMETRY:
                if not _integer(geometry.get(key)): fail(f'invalid model_geometry.{key}')
        shape=row.get('shape')
        if not isinstance(shape,dict): fail('shape must be an object')
        else:
            for key in SHAPE[:-1]:
                if not _integer(shape.get(key)): fail(f'invalid shape.{key}')
            padding=shape.get('padding_fraction')
            if type(padding) not in (int,float) or not math.isfinite(padding) or not 0<=padding<1:
                fail('invalid shape.padding_fraction')
        if not _integer(row.get('samples')): fail('samples must be positive')
        if not _integer(row.get('updates'),0): fail('updates must be nonnegative')
        if row.get('mode') in {'offline','online'} and not _integer(row.get('updates')):
            fail('training PASS needs positive updates')
        revision=row.get('source_revision')
        if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision): fail('invalid source_revision')
        if expected_revision is not None and revision != expected_revision: fail('stale source_revision')
        artifact=row.get('artifact')
        if not isinstance(artifact,dict): fail('PASS needs artifact');continue
        digest=artifact.get('sha256')
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            fail('invalid artifact sha256');continue
        reference=artifact.get('path')
        if not isinstance(reference,str) or not reference:
            fail('missing artifact path');continue
        relative=Path(reference)
        if relative.is_absolute() or '..' in relative.parts:
            fail('artifact path must stay relative to matrix directory');continue
        artifact_path=path.parent/relative
        try:
            raw=artifact_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:
                fail('artifact digest mismatch');continue
            evidence=json.loads(raw)
        except (OSError,ValueError) as exc:
            fail(f'artifact unreadable: {exc}');continue
        records=evidence.get('profile_records') if isinstance(evidence,dict) else None
        if not isinstance(records,list): fail('artifact missing profile_records');continue
        matches=[record for record in records if isinstance(record,dict) and record.get('id')==artifact.get('record_id')]
        if len(matches)!=1: fail('artifact must contain exactly one matching profile record');continue
        record=matches[0]
        # Optional context claims (for example a dirty-tree digest) must also
        # agree; adding a field must not bypass artifact binding.
        compare_keys=set(IDENTITY+CONTEXT) | (set(row)-{'artifact','reason'})
        for key in sorted(compare_keys):
            if key not in record or record[key]!=row.get(key): fail(f'artifact profile mismatch: {key}')
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,default=Path(__file__).resolve().parents[2]/'docs/npu/support_matrix.json')
    parser.add_argument('--expected-revision')
    args=parser.parse_args()
    errors=validate_matrix(args.matrix,args.expected_revision)
    print(json.dumps({'status':'FAILED' if errors else 'VALID','errors':errors},indent=2))
    return int(bool(errors))

if __name__=='__main__':
    raise SystemExit(main())
