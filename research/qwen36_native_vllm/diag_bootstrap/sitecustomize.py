"""Diagnostics only: run the adapter bootstrap named by Q36_DIAG_BOOTSTRAP in every process."""
import os

if os.environ.get("Q36_DIAG_BOOTSTRAP"):
    import runpy
    runpy.run_path(os.environ["Q36_DIAG_BOOTSTRAP"], run_name="__q36_diag_adapter__")
