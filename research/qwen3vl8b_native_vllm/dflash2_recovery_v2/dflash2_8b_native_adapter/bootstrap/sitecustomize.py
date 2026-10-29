"""Opt-in worker bootstrap; failed installation aborts instead of falling back."""
import os
from pathlib import Path
import sys

if os.environ.get('DFLASH2_NATIVE_ENABLE')=='1':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    try:
        from dflash2_8b_native_adapter.contracts import runtime_receipt
        runtime_receipt()
        from dflash2_8b_native_adapter.install import install
        install()
    except Exception as error:
        raise SystemExit('DFlash2 native adapter installation failed: '+repr(error)) from error
