"""Chat-context domain errors."""
from __future__ import annotations

from ..shared.errors import ConflictError, NotFoundError


class SessionNotFoundError(NotFoundError):
    """Mcp-Session-Id refers to a session this server doesn't know.

    Interface layer maps this to HTTP 404 per the Streamable HTTP
    transport spec ("after a server terminates a session it MUST
    respond to requests containing that session ID with HTTP 404").
    """


class SessionStateError(ConflictError):
    """An operation was attempted in a state that doesn't allow it
    (e.g., tools/list before notifications/initialized). Interface
    maps to JSON-RPC ``-32600 Invalid Request``."""
