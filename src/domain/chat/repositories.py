"""Chat-context ports.

A *port* is an interface the domain defines on its own terms —
adapters under ``infrastructure/`` implement it. The domain never
imports adapters; the composition root wires them.

We use abstract base classes (``Protocol`` would also work, but ABC
gives clearer error messages on missing methods and supports
``isinstance`` checks in tests).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .entities import Session
from .value_objects import SessionId


class ChatSessionRepository(ABC):
    """Persistence + lifecycle for :class:`Session` aggregates.

    Why a port (and not an in-memory dict directly):
        * Different deployments swap the implementation: in-process
          dict for single-pod, Redis for multi-pod HA, Mongo for
          long-lived sessions, etc.
        * Tests use a fake — no need to spin up Redis to unit-test
          the application layer.
        * The spec's idle-GC + DELETE semantics are repository
          concerns, not domain concerns; concentrating them behind
          a port keeps the application service tiny.
    """

    @abstractmethod
    async def get(self, session_id: SessionId) -> Optional[Session]:
        """Return the session by id, or None if unknown.

        Adapters are responsible for refusing cross-tenant access:
        if the lookup hits a session whose tenant doesn't match the
        caller, return None (do NOT raise — the caller can't
        distinguish "stale id" from "id stolen across tenants",
        and either way the right answer is "unknown")."""

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Upsert. Persists state transitions + last_activity_at."""

    @abstractmethod
    async def delete(self, session_id: SessionId) -> bool:
        """Remove the session. Returns True if the row existed."""

    @abstractmethod
    async def gc_idle(self, idle_seconds: int) -> int:
        """Drop sessions whose last_activity_at is older than now -
        idle_seconds. Returns count reaped. Called from a background
        sweeper (composition wires the schedule)."""
