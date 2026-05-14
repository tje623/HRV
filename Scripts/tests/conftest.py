"""Pytest configuration: make Scripts/ importable from test files.

Tests live under Scripts/tests/ and import modules like `data_pipeline`
directly. This conftest prepends Scripts/ to sys.path so those imports
resolve without a packaging/install step.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
