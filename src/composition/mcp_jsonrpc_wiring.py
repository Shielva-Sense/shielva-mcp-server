"""Wiring for the MCP JSON-RPC inbound adapter.

Composition rules
-----------------
* The infrastructure adapter (``InMemoryChatSessionRepository``) is
  instantiated **once** here and held for the process lifetime.
* The application service (``ChatApplicationService``) is constructed
  with the adapter + a session-id factory + the server's spec
  version. It is also a process singleton.
* The interface dispatcher receives the application service and
  exposes the JSON-RPC method surface.
* :func:`build_mcp_jsonrpc_router` returns the FastAPI router so
  ``main.py`` can ``app.include_router(...)`` it.

Why a wiring module (not a DI framework):
    The graph is small enough to wire by hand. Adding a framework
    (dependency-injector, lagom, …) would obscure what we're
    composing without adding meaningful safety.
"""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter

from src.application.chat import ChatApplicationService
from src.domain.chat.repositories import ChatSessionRepository
from src.infrastructure.persistence import (
    InMemoryChatSessionRepository,
    RedisChatSessionRepository,
)
from src.interface.mcp_jsonrpc.dispatcher import (
    PROTOCOL_VERSION,
    MCPDispatcher,
)
from src.interface.mcp_jsonrpc.transport import build_router

# Process-wide singletons. Tests construct their own instances.
_chat_repository: ChatSessionRepository | None = None
_chat_service: ChatApplicationService | None = None
_dispatcher: MCPDispatcher | None = None


def _make_session_repository() -> ChatSessionRepository:
    """Redis whenever it is configured; in-process only for local dev.

    🚨 The in-memory store is per PROCESS. The container runs
    ``uvicorn --workers 4``, so with it ``initialize`` created a session in one
    worker and the next request hit another that had never seen it — about
    three MCP calls in four failed with "requires a completed initialize
    handshake" / "Session terminated". A shared store is not an HA nicety here;
    it is what makes a single multi-worker pod work at all.

    Deliberately no silent fallback when REDIS_URL IS set but unreachable:
    degrading to per-worker memory would quietly reinstate exactly that bug.
    """
    from config.settings import get_settings

    settings = get_settings()
    url = settings.redis_url.get_secret_value() if settings.redis_url else ""
    if not url:
        return InMemoryChatSessionRepository()

    import redis.asyncio as redis_asyncio

    client = redis_asyncio.from_url(url, db=settings.redis_db, decode_responses=True)
    ttl = int(os.getenv("MCP_SESSION_TTL_SECONDS", "3600"))
    return RedisChatSessionRepository(client, ttl_seconds=ttl)


def _make_session_id() -> str:
    """Cryptographically-secure session id factory. UUID4 fits the
    spec's character-set requirement (0x21-0x7E) and uniqueness."""
    return str(uuid.uuid4())


def _server_version() -> str:
    """Server's own version string surfaced in InitializeResult.
    Override via env so build pipelines can stamp the release."""
    return os.getenv("MCP_SERVER_VERSION", "1.0.0")


def build_mcp_jsonrpc_router() -> APIRouter:
    """Return the configured router. Idempotent — repeated calls
    return the same router (singleton wiring)."""
    global _chat_repository, _chat_service, _dispatcher

    if _dispatcher is None:
        _chat_repository = _make_session_repository()
        _chat_service = ChatApplicationService(
            repository=_chat_repository,
            session_id_factory=_make_session_id,
            server_protocol_version=PROTOCOL_VERSION,
        )
        _dispatcher = MCPDispatcher(
            chat_service=_chat_service,
            server_version=_server_version(),
        )
    return build_router(_dispatcher)
