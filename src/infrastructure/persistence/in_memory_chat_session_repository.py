"""In-process implementation of :class:`ChatSessionRepository`.

Single-pod deployments are happy with this. HA deployments will swap
in a Redis-backed adapter that keeps the session map cluster-wide.
The application layer doesn't change — that's the point of the port.

Concurrency:
    asyncio is single-threaded inside one event loop, but multiple
    concurrent coroutines can interleave on the same dict. We guard
    every mutation with an asyncio.Lock so iteration during gc_idle
    never races with a concurrent save() or delete().
"""

from __future__ import annotations

import asyncio
import time

import structlog

from src.domain.chat.entities import Session
from src.domain.chat.repositories import ChatSessionRepository
from src.domain.chat.value_objects import SessionId

logger = structlog.get_logger(__name__)


class InMemoryChatSessionRepository(ChatSessionRepository):
    """Process-local session store. Bound to the uvicorn worker
    lifetime — restart loses sessions, which matches the spec's
    expectation that clients must re-initialize after disconnect."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: SessionId) -> Session | None:
        # No lock needed for a single dict-read; lock is for the
        # composite mutations below.
        return self._sessions.get(str(session_id))

    async def save(self, session: Session) -> None:
        async with self._lock:
            self._sessions[str(session.id)] = session

    async def delete(self, session_id: SessionId) -> bool:
        async with self._lock:
            return self._sessions.pop(str(session_id), None) is not None

    async def gc_idle(self, idle_seconds: int) -> int:
        cutoff = time.time() - max(1, int(idle_seconds))
        async with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_activity_at < cutoff]
            for sid in stale:
                self._sessions.pop(sid, None)
        if stale:
            logger.info("mcp.sessions_gc", reaped=len(stale))
        return len(stale)

    # ── test seam (not part of the port) ──────────────────────────
    # Tests sometimes want to count sessions without leaking that
    # capability through the port (which is intentionally
    # minimal). This helper is fine because tests construct the
    # adapter directly.

    def _size(self) -> int:
        return len(self._sessions)
