"""Chat bounded context.

Owns the lifecycle of an MCP **session** — the stateful conversation
established by ``initialize`` and torn down by ``DELETE /mcp``. The
session is the aggregate root of this context.

Public surface:
    Entities      : Session
    Value objects : SessionId, SessionState, ProtocolVersion, ClientInfo
    Ports         : ChatSessionRepository
    Errors        : SessionNotFoundError, SessionStateError
"""

from .entities import Session
from .errors import SessionNotFoundError, SessionStateError
from .repositories import ChatSessionRepository
from .value_objects import (
    ClientInfo,
    ProtocolVersion,
    SessionId,
    SessionState,
)

__all__ = [
    "ChatSessionRepository",
    "ClientInfo",
    "ProtocolVersion",
    "Session",
    "SessionId",
    "SessionNotFoundError",
    "SessionState",
    "SessionStateError",
]
