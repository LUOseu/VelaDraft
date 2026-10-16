"""Test package marker.

Without this file the repo-local ``tests`` directory is a namespace package and
loses to any real ``tests`` package earlier/later on ``sys.path`` (the active
conda env ships one), which breaks ``from tests.test_online_mode import ...``.
"""
