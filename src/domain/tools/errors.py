"""Tools-context domain errors."""

from __future__ import annotations

from ..shared.errors import DomainError, NotFoundError, UnauthorizedError


class ToolNotFoundError(NotFoundError):
    """``tools/call`` for a name not in the catalogue.

    Interface adapter maps to JSON-RPC ``-32004 Tool not found`` per
    the MCP spec's reserved error codes.
    """


class ToolPermissionDeniedError(UnauthorizedError):
    """Tenant lacks one of the tool's required permissions.

    Interface adapter maps to JSON-RPC ``-31998 Forbidden`` (the
    Shielva extension code for permission denial — distinct from
    auth-not-presented which the transport rejects upstream).
    """


class ToolExecutionError(DomainError):
    """The tool ran but raised. The adapter catches this and turns
    it into a ``ToolResult.failure`` rather than a JSON-RPC error —
    per spec, tool failures are visible to the LLM, not the host."""
