"""Inert by default; opt in to the local shared-block JetSpec mount explicitly."""
import os

if os.environ.get('VLLM_OFFICIAL_JETSPEC_LOCAL_BLOCK_ADAPTER')=='1':
    import sys
    from pathlib import Path
    try:
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
        from official_jetspec_adapter import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(86)
