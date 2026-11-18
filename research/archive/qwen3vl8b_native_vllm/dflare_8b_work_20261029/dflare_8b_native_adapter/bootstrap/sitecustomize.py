"""Opt-in bootstrap requires real frozen CPU/export receipt; fails closed."""
import os
from pathlib import Path
import sys
if os.environ.get('DFLARE_NATIVE_ENABLE')=='1':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    try:
        from dflare_8b_native_adapter.install import install
        install()
    except Exception as error:
        raise SystemExit('DFlare native installation failed: '+repr(error)) from error
