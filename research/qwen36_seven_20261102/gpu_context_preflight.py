"""Require the live GPU graph-replay regression test before context-model timing."""
import json
from pathlib import Path
import sys
import unittest
import torch
from research.qwen36_seven_20261102 import test_context_contract

assert torch.cuda.is_available(), 'A real GPU is required; a skipped test is not a pass'
suite=unittest.defaultTestLoader.loadTestsFromModule(test_context_contract)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            skipped=len(result.skipped),device=torch.cuda.get_device_name(),
            status='complete' if result.wasSuccessful() and not result.skipped else 'failed')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
assert report['status']=='complete',report
