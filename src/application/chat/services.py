"""Chat application service — orchestrates Session lifecycle.

What lives here:
    * Initialize a session for a tenant (mints a SessionId, persists
      a fresh Session aggregate via the port).
    * Mark ready (transitions state on the aggregate, persists).
    * Look up a session, enforcing the spec's stale-id rules.
    * Set log level.
    * Close (DELETE) a session.

What does NOT live here:
    * Tools/resources/prompts dispatch — those are separate
      application services in their own contexts (slice 2+).
    * HTTP framing, JSON-RPC envelopes — interface adapter.
    * In-memory storage choice — that's infrastructure.

The service is constructed with the port + a session-id factory
injected. The factory is a callable (``Callable[[], str]``) so tests
can plug in deterministic ids without touching uuid.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Optional

import structlog

from src.domain.chat.entities import Session
from src.domain.chat.errors import SessionNotFoundError, SessionStateError
from src.domain.chat.repositories import ChatSessionRepository
from src.domain.chat.value_objects import (
    ClientInfo, ProtocolVersion, SessionId, SessionState,
)
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


SessionIdFactory = Callable[[], str]


class ChatApplicationService:
    """Orchestrates session lifecycle for the MCP JSON-RPC layer."""

    def __init__(
        self,
        *,
        repository: ChatSessionRepository,
        session_id_factory: SessionIdFactory,
        server_protocol_version: str,
    ) -> None:
        self._repo = repository
        self._make_id = session_id_factory
        self._server_protocol_version = server_protocol_version

    # ── lifecycle ──────────────────────────────────────────────────

    async def initialize(
        self,
        *,
        tenant: TenantContext,
        client_info: ClientInfo,
        client_protocol_version: str,
    ) -> Session:
        """Mint a fresh Session aggregate and persist it.

        Per spec, the server returns *its* supported protocol version
        regardless of what the client asked for. The client must
        either accept or disconnect.
        """
        session = Session(
            id               = SessionId(self._make_id()),
            tenant           = tenant,
            protocol_version = ProtocolVersion(self._server_protocol_version),
            client_info      = client_info,
            state            = SessionState.INITIALIZING,
        )
        await self._repo.save(session)
        logger.info(
            "mcp.session_created",
            session_id=str(session.id),
            tenant_id=tenant.tenant_id,
            client=f"{client_info.name}/{client_info.version}",
            requested_version=client_protocol_version,
            served_version=self._server_protocol_version,
        )
        return session

    async def mark_initialized(self, *, session_id: SessionId,
                               tenant: TenantContext) -> None:
        """Apply ``notifications/initialized`` — moves INITIALIZING → READY."""
        session = await self._load_owned(session_id, tenant)
        session.mark_ready()
        session.touch()
        await self._repo.save(session)

    async def ping(self, *, session_id: Optional[SessionId],
                   tenant: TenantContext) -> None:
        """Liveness. Allowed pre-session (no session_id supplied).
        When session_id is supplied it must exist + belong to the tenant."""
        if session_id is None:
            return
        await self._load_owned(session_id, tenant)

    async def set_log_level(self, *, session_id: SessionId,
                            tenant: TenantContext, level: str) -> None:
        session = await self._load_owned(session_id, tenant)
        session.set_log_level(level)
        session.touch()
        await self._repo.save(session)

    async def close(self, *, session_id: SessionId,
                    tenant: TenantContext) -> bool:
        """DELETE /mcp. Returns True if a session was removed."""
        # We load first so the tenant-ownership check fires; then we
        # remove. Not-found is silently fine (idempotent close).
        try:
            await self._load_owned(session_id, tenant)
        except SessionNotFoundError:
            return False
        return await self._repo.delete(session_id)

    # ── helpers used by interface adapter ─────────────────────────

    async def get_for_tenant(self, *, session_id: SessionId,
                             tenant: TenantContext) -> Session:
        """Load + ownership check + touch. The interface adapter calls
        this on every request that carries a session id; if the
        session is unknown we raise :class:`SessionNotFoundError`
        which the adapter maps to HTTP 404 (per the Streamable HTTP
        spec — see ``interface/mcp_jsonrpc``)."""
        session = await self._load_owned(session_id, tenant)
        session.touch()
        await self._repo.save(session)
        return session

    async def assert_method_allowed(self, *, session_id: SessionId,
                                    tenant: TenantContext, method: str) -> Session:
        """Combines load + state-check. Raises SessionStateError if
        the method isn't allowed in the session's current state."""
        session = await self._load_owned(session_id, tenant)
        if not session.allows_method(method):
            raise SessionStateError(
                f"Method {method} requires a completed initialize handshake",
            )
        session.touch()
        await self._repo.save(session)
        return session

    # ── internal ──────────────────────────────────────────────────

    async def _load_owned(self, session_id: SessionId,
                          tenant: TenantContext) -> Session:
        session = await self._repo.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")
        if session.tenant.tenant_id != tenant.tenant_id:
            # The spec lets us return 404 here — surfacing
            # "different tenant" would leak that a session id exists.
            raise SessionNotFoundError(f"Unknown session: {session_id}")
        return session
