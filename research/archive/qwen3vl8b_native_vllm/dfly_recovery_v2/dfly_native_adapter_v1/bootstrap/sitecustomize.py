"""Opt-in bootstrap requires real frozen CPU/export receipt; fails closed."""
import os
from pathlib import Path
import sys
if os.environ.get('DFLY_NATIVE_ENABLE')=='1':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    try:
        from dfly_native_adapter_v1.install import install
        install()
    except Exception as error:
        raise SystemExit('DFly native installation failed: '+repr(error)) from error
