"""Fail-closed generation from immutable small audit artifacts."""
import copy
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_completed_artifacts_generate_scoped_profiles():
    from scripts.npu.summarize_audit import generate
    evidence, report = generate(ROOT / 'docs/npu/audit-2026-10-21', ROOT / 'docs/npu/evidence/2026-10-21/cuda_regression')
    ids = {r['id'] for r in evidence['profile_records']}
    assert 'dfly-compaction-npu' in ids
    assert 'dfly-stage-boundary-npu' in ids
    assert 'dfly-npu-export-cuda' not in ids
    assert 'slower' in report and 'PENDING' in report
    assert all(r['source_revision_kind'] == 'base_commit_plus_dirty_source_hashes' for r in evidence['profile_records'])


def test_failed_parity_cannot_be_promoted():
    from scripts.npu.summarize_audit import require_pass
    with pytest.raises(ValueError, match='parity'):
        require_pass({'status': 'FAILED'}, 'parity')
