"""Pytest configuration.

The mcp-server package layout puts source under ``src/`` without a
real package root above it (FastAPI services in this repo are run
via ``uvicorn src.main:app`` with ``cwd=mcp-server``). For pytest
to resolve ``import src.domain...`` we need the project root on
``sys.path``. Hook it here once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# /…/mcp-server/tests/conftest.py → /…/mcp-server
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 🚨 Before any `src` import, which is why it lives HERE and not in a test file.
#
# This service builds its settings at IMPORT time and refuses to start with a
# sealed field it cannot open. Under pytest there is no envelope and no vault, so
# collection died on the first module that imported anything from `src` — the
# whole suite, on master, for want of three placeholder values. A per-file fix
# would work for that file and leave the next one to rediscover it.
#
# Placeholders under the permissive flag, exactly as a dev shell does. Nothing
# here is a real credential and nothing connects: `setdefault`, so a developer
# or CI that exports the real thing still wins.
os.environ.setdefault("SHIELVA_SEALED_PERMISSIVE", "1")
os.environ.setdefault("JWT_SECRET_KEY", "x" * 32)
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("AUDIT_HMAC_SECRET", "y" * 32)
os.environ.setdefault("VAULT_AUDIT_HMAC_SECRET", "y" * 32)
