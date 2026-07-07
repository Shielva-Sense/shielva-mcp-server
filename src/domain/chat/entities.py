"""Session aggregate root.

The Session is the only aggregate this context owns. Its identity is
:class:`SessionId`. Its lifecycle is encoded as a state machine on
:class:`SessionState` and enforced by methods — callers cannot mutate
``state`` directly.

What lives ON the aggregate vs. on the application service:
    On the aggregate:
        * state transitions (initialize → ready → closing)
        * tenant identity (immutable after construction)
        * client metadata captured at initialize
        * log-level setting (per-session preference)
    Outside the aggregate (application services):
        * persistence (the ChatSessionRepository port)
        * notifications/progress fan-out (delivered via the SSE
          adapter, not the aggregate — the aggregate has no I/O)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..shared.tenant import TenantContext
from .errors import SessionStateError
from .value_objects import ClientInfo, ProtocolVersion, SessionId, SessionState


@dataclass(slots=True)
class Session:
    """MCP session aggregate.

    Mutable — but mutation goes through methods so invariants are
    preserved. The repository (port) is responsible for round-
    tripping the aggregate's fields without leaking storage detail.
    """

    id: SessionId
    tenant: TenantContext
    protocol_version: ProtocolVersion
    state: SessionState = SessionState.INITIALIZING
    client_info: ClientInfo | None = None
    log_level: str = "info"
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)

    # ── state transitions ──────────────────────────────────────────

    def mark_ready(self) -> None:
        """Apply ``notifications/initialized``. Idempotent if
        already ready (spec is lenient on re-sending the
        notification — we ignore rather than fail)."""
        if self.state is SessionState.INITIALIZING:
            self.state = SessionState.READY
        elif self.state is SessionState.READY:
            return
        else:
            raise SessionStateError(f"Cannot mark ready from state {self.state.value!r}")

    def mark_closing(self) -> None:
        """Begin teardown. The repository GCs the session afterward."""
        self.state = SessionState.CLOSING

    # ── per-session preferences ────────────────────────────────────

    def set_log_level(self, level: str) -> None:
        # Validation of allowed levels happens at the interface layer
        # against the spec enum; the aggregate just stores what it's
        # told.
        self.log_level = level

    # ── activity tracking (for idle-GC) ────────────────────────────

    def touch(self, now: float | None = None) -> None:
        self.last_activity_at = now if now is not None else time.time()

    # ── invariants helpers (used by application services) ─────────

    def allows_method(self, method: str) -> bool:
        """Spec rule: only ``initialize`` and ``ping`` are allowed
        before the client sends notifications/initialized."""
        if self.state is SessionState.READY:
            return True
        if self.state is SessionState.INITIALIZING:
            return method in ("initialize", "ping")
        return False
