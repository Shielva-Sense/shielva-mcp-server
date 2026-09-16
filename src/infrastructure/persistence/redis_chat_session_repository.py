"""Redis implementation of :class:`ChatSessionRepository`.

🚨 WHY THIS EXISTS — the in-memory store broke every MCP client in production.

The container runs ``uvicorn --workers 4``. ``InMemoryChatSessionRepository``
is a per-PROCESS dict, so ``initialize`` created a session in worker A and the
very next request — ``notifications/initialized`` or ``tools/call`` — landed on
worker B, C or D, which had never heard of it. Roughly three calls in four
failed with one of:

    mcp.initialized_unknown_session
    Method tools/call requires a completed initialize handshake
    Session terminated

and clients re-initialised in a loop. It looked intermittent because a request
that happened to land on the creating worker worked fine. One pod was enough to
hit it: this is not a multi-replica problem, it is a multi-process one, and it
gets strictly worse once the deployment scales horizontally.

Sessions now live in Redis, so every worker and every pod sees the same map.

TTL IS THE IDLE GC
------------------
``ChatApplicationService`` touches and saves the session on every request, and
each save resets the key's TTL. A session therefore expires exactly
``ttl_seconds`` after its last activity — which is what ``gc_idle`` exists to
enforce. No sweeper is wired for the in-memory store either, so it leaked
forever; here Redis does the sweeping and ``gc_idle`` has nothing left to do.

WHAT IS STORED
--------------
The aggregate's own fields. ``TenantContext`` carries ids, role and permissions
and deliberately holds NO credential (see ``src/tools/_caller.py`` — the
caller's token lives in a request-scoped ContextVar precisely so it never lands
anywhere like this). The cross-tenant check stays where it was, in
``ChatApplicationService._load_owned``, which compares the stored tenant id with
the caller's; this adapter only has to round-trip that id faithfully.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.domain.chat.entities import Session
from src.domain.chat.repositories import ChatSessionRepository
from src.domain.chat.value_objects import ClientInfo, ProtocolVersion, SessionId, SessionState
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)

#: Namespaced so a session key can never collide with the RAG query cache or
#: anything else sharing this Redis database.
KEY_PREFIX = "mcp:session:"


def _key(session_id: SessionId | str) -> str:
    return f"{KEY_PREFIX}{session_id}"


def _to_record(session: Session) -> dict[str, Any]:
    t = session.tenant
    return {
        "id": str(session.id),
        "tenant": {
            "tenant_id": t.tenant_id,
            "user_id": t.user_id,
            "user_email": t.user_email,
            "role": t.role,
            "permissions": list(t.permissions),
        },
        "protocol_version": str(session.protocol_version),
        "state": session.state.value,
        "client_info": (
            {"name": session.client_info.name, "version": session.client_info.version}
            if session.client_info is not None
            else None
        ),
        "log_level": session.log_level,
        "created_at": session.created_at,
        "last_activity_at": session.last_activity_at,
    }


def _from_record(rec: dict[str, Any]) -> Session:
    t = rec["tenant"]
    ci = rec.get("client_info")
    return Session(
        id=SessionId(rec["id"]),
        tenant=TenantContext(
            tenant_id=t["tenant_id"],
            user_id=t["user_id"],
            user_email=t["user_email"],
            role=t.get("role", "Customer_Basic"),
            permissions=tuple(t.get("permissions") or ()),
        ),
        protocol_version=ProtocolVersion(value=rec["protocol_version"]),
        state=SessionState(rec["state"]),
        client_info=ClientInfo(name=ci["name"], version=ci["version"]) if ci else None,
        log_level=rec.get("log_level", "info"),
        created_at=float(rec["created_at"]),
        last_activity_at=float(rec["last_activity_at"]),
    )


class RedisChatSessionRepository(ChatSessionRepository):
    """Cluster-wide session store. Shared by every worker and every pod."""

    def __init__(self, client: Any, ttl_seconds: int = 3600) -> None:
        # ``client`` is a ``redis.asyncio.Redis``. Injected rather than built
        # here so tests pass a fake and the composition root owns the URL.
        self._redis = client
        self._ttl = max(60, int(ttl_seconds))

    async def get(self, session_id: SessionId) -> Session | None:
        raw = await self._redis.get(_key(session_id))
        if raw is None:
            return None
        try:
            return _from_record(json.loads(raw))
        except (ValueError, KeyError, TypeError):
            # A record we cannot read is an unknown session, not a crash: the
            # client re-initialises, which is the spec's recovery path anyway.
            logger.warning("mcp.session_record_unreadable", session_id=str(session_id))
            return None

    async def save(self, session: Session) -> None:
        await self._redis.set(_key(session.id), json.dumps(_to_record(session)), ex=self._ttl)

    async def delete(self, session_id: SessionId) -> bool:
        return bool(await self._redis.delete(_key(session_id)))

    async def gc_idle(self, idle_seconds: int) -> int:
        # Expiry is enforced by the TTL reset on every save(); see module docstring.
        return 0
