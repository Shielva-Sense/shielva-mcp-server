"""Pytest configuration.

The mcp-server package layout puts source under ``src/`` without a
real package root above it (FastAPI services in this repo are run
via ``uvicorn src.main:app`` with ``cwd=mcp-server``). For pytest
to resolve ``import src.domain...`` we need the project root on
``sys.path``. Hook it here once.
"""

from __future__ import annotations

import sys
from pathlib import Path

# /…/mcp-server/tests/conftest.py → /…/mcp-server
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
