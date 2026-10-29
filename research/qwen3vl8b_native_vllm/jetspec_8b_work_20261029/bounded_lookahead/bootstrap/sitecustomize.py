"""Separate bounded lookahead bootstrap; inert unless explicitly enabled."""
import os

if os.environ.get("VLLM_OFFICIAL_SPATIAL_BOUNDED_ADAPTER") == "1":
    import sys
    from pathlib import Path
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from official_spatial_bounded_adapter import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(86)
