"""ChatApplicationService — end-to-end with the in-memory repo.

The repo is real (we want to exercise the lock + dict mutation
behaviour); the only injected fakes are the id factory + protocol
version, so tests are deterministic.

Proves:
    * initialize creates a session in state=INITIALIZING.
    * mark_initialized transitions to READY.
    * Tenant ownership is enforced — load with a foreign tenant
      raises SessionNotFoundError (the spec MUST: never leak
      cross-tenant session ids).
    * close is idempotent.
    * assert_method_allowed honours the spec's pre-initialize gate.
"""

from __future__ import annotations

import pytest

from src.application.chat import ChatApplicationService
from src.domain.chat.errors import SessionNotFoundError, SessionStateError
from src.domain.chat.value_objects import ClientInfo, SessionId, SessionState
from src.domain.shared.tenant import TenantContext
from src.infrastructure.persistence import InMemoryChatSessionRepository

_PROTOCOL = "2024-11-05"


def _ids():
    """Deterministic id factory — each call returns the next "sid-N"."""
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"sid-{counter['n']}"

    return factory


def _service() -> ChatApplicationService:
    return ChatApplicationService(
        repository=InMemoryChatSessionRepository(),
        session_id_factory=_ids(),
        server_protocol_version=_PROTOCOL,
    )


def _client() -> ClientInfo:
    return ClientInfo(name="test", version="0.1")


def _tenant(tid: str = "t1") -> TenantContext:
    return TenantContext(tenant_id=tid, user_id="u", user_email="u@x")


@pytest.mark.asyncio
async def test_initialize_creates_session_in_initializing_state():
    svc = _service()
    s = await svc.initialize(
        tenant=_tenant(),
        client_info=_client(),
        client_protocol_version=_PROTOCOL,
    )
    assert s.state is SessionState.INITIALIZING
    assert str(s.id) == "sid-1"


@pytest.mark.asyncio
async def test_mark_initialized_transitions_to_ready():
    svc = _service()
    s = await svc.initialize(
        tenant=_tenant(),
        client_info=_client(),
        client_protocol_version=_PROTOCOL,
    )
    await svc.mark_initialized(session_id=s.id, tenant=_tenant())
    reloaded = await svc.get_for_tenant(session_id=s.id, tenant=_tenant())
    assert reloaded.state is SessionState.READY


@pytest.mark.asyncio
async def test_foreign_tenant_cannot_load_session():
    svc = _service()
    s = await svc.initialize(
        tenant=_tenant("alice"),
        client_info=_client(),
        client_protocol_version=_PROTOCOL,
    )
    with pytest.raises(SessionNotFoundError):
        await svc.get_for_tenant(session_id=s.id, tenant=_tenant("bob"))


@pytest.mark.asyncio
async def test_close_is_idempotent():
    svc = _service()
    s = await svc.initialize(
        tenant=_tenant(),
        client_info=_client(),
        client_protocol_version=_PROTOCOL,
    )
    assert await svc.close(session_id=s.id, tenant=_tenant()) is True
    # Calling close again on an unknown session must not raise.
    assert await svc.close(session_id=s.id, tenant=_tenant()) is False


@pytest.mark.asyncio
async def test_assert_method_allowed_blocks_pre_initialize():
    svc = _service()
    s = await svc.initialize(
        tenant=_tenant(),
        client_info=_client(),
        client_protocol_version=_PROTOCOL,
    )
    with pytest.raises(SessionStateError):
        await svc.assert_method_allowed(
            session_id=s.id,
            tenant=_tenant(),
            method="tools/list",
        )


@pytest.mark.asyncio
async def test_assert_method_allowed_after_ready():
    svc = _service()
    s = await svc.initialize(
        tenant=_tenant(),
        client_info=_client(),
        client_protocol_version=_PROTOCOL,
    )
    await svc.mark_initialized(session_id=s.id, tenant=_tenant())
    # Must not raise — session is READY, all methods unblocked.
    await svc.assert_method_allowed(
        session_id=s.id,
        tenant=_tenant(),
        method="tools/list",
    )


@pytest.mark.asyncio
async def test_unknown_session_raises_session_not_found():
    svc = _service()
    with pytest.raises(SessionNotFoundError):
        await svc.get_for_tenant(
            session_id=SessionId("missing"),
            tenant=_tenant(),
        )


@pytest.mark.asyncio
async def test_ping_without_session_is_no_op():
    """Ping must work pre-session for liveness probing."""
    svc = _service()
    # Should not raise even though no session exists.
    await svc.ping(session_id=None, tenant=_tenant())


@pytest.mark.asyncio
async def test_ping_with_unknown_session_id_raises():
    """When an id IS supplied it must be valid — distinct from
    no-id-at-all liveness."""
    svc = _service()
    with pytest.raises(SessionNotFoundError):
        await svc.ping(session_id=SessionId("nope"), tenant=_tenant())
