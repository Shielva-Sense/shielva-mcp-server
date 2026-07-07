"""Value objects for the chat context.

All immutable, equal-by-value. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

# A session id is a cryptographically-secure unique string. The
# generator lives in infrastructure (uuid.uuid4) so domain stays free
# of stdlib import-side-effects. NewType keeps it nominal — accidental
# str → SessionId is a type error in mypy.
SessionId = NewType("SessionId", str)


class SessionState(str, Enum):
    """MCP session lifecycle states.

    State transitions (enforced by Session aggregate):
        INITIALIZING → READY    on notifications/initialized
        READY        → CLOSING  on DELETE / explicit shutdown
        CLOSING      → (gone)   after cleanup
    """

    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"


@dataclass(frozen=True, slots=True)
class ProtocolVersion:
    """MCP spec version negotiated at initialize.

    Compared lexicographically (spec uses date-formatted versions
    like ``2024-11-05`` / ``2025-03-26``). Server returns its
    supported version; if the client doesn't support it, the client
    disconnects (server has no choice — there's no version range).
    """

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ClientInfo:
    """Self-reported identity of the connecting MCP client."""

    name: str
    version: str
