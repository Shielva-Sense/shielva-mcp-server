"""JSON-RPC 2.0 + MCP-defined error codes.

JSON-RPC reserves -32768 to -32000 for protocol errors; MCP uses
-32002, -32004 for resource/tool not-found per the spec. Everything
≥ -31999 is application-defined; we use -32700+ for parse errors as
per JSON-RPC, and 4xx-style codes mapped down for our domain errors.
"""
from __future__ import annotations

from typing import Any, Optional


class JsonRpcException(Exception):
    """Raised by handlers; caught by the dispatcher and converted to
    a JSON-RPC error response with ``code`` and ``message``."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# ── JSON-RPC 2.0 standard codes ───────────────────────────────────────
PARSE_ERROR      = -32700  # malformed JSON
INVALID_REQUEST  = -32600  # not a valid JSON-RPC request object
METHOD_NOT_FOUND = -32601
INVALID_PARAMS   = -32602
INTERNAL_ERROR   = -32603

# ── MCP-defined codes ─────────────────────────────────────────────────
RESOURCE_NOT_FOUND = -32002
TOOL_NOT_FOUND     = -32004

# ── Shielva extensions (≥ -31999, application-defined) ────────────────
UNAUTHENTICATED  = -31999  # missing/invalid X-Tenant-ID
FORBIDDEN        = -31998  # tenant has no access to this resource
RATE_LIMITED     = -31997
