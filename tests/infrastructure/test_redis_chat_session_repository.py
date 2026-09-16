"""RedisChatSessionRepository — the store every uvicorn worker shares.

🚨 THE BUG THESE TESTS PIN: the container runs ``uvicorn --workers 4`` and the
session store was a per-process dict. ``initialize`` created a session in one
worker; ``notifications/initialized`` and ``tools/call`` landed on another that
had never seen it. About three MCP calls in four failed with "requires a
completed initialize handshake" or "Session terminated".

Two service instances over one shared store is exactly two workers over one
Redis, so that is how the regression is reproduced here — and the in-memory
store is run through the same scenario to prove it really does fail it.
"""

from __future__ import annotations

import json

import pytest

from src.application.chat import ChatApplicationService
from src.domain.chat.errors import SessionNotFoundError
from src.domain.chat.value_objects import ClientInfo, SessionId, SessionState
from src.domain.shared.tenant import TenantContext
from src.infrastructure.persistence import (
    InMemoryChatSessionRepository,
    RedisChatSessionRepository,
)
from src.infrastructure.persistence.redis_chat_session_repository import KEY_PREFIX

_PROTOCOL = "2024-11-05"


class FakeRedis:
    """The four async calls the adapter makes, over one shared dict."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.data[key] = value
        if ex is not None:
            self.ttl[key] = ex
        return True

    async def delete(self, key: str) -> int:
        existed = key in self.data
        self.data.pop(key, None)
        self.ttl.pop(key, None)
        return int(existed)


def _ids():
    n = {"i": 0}

    def factory() -> str:
        n["i"] += 1
        return f"sid-{n['i']}"

    return factory


def _tenant(tid: str = "Tenant-a") -> TenantContext:
    return TenantContext(tenant_id=tid, user_id="u1", user_email="u@x", role="platform_owner", permissions=("p1", "p2"))


def _worker(repo) -> ChatApplicationService:
    return ChatApplicationService(repository=repo, session_id_factory=_ids(), server_protocol_version=_PROTOCOL)


@pytest.mark.asyncio
async def test_a_session_created_by_one_worker_is_ready_on_another():
    """The production failure, reproduced: initialize on worker A, then the
    handshake and a tools/call on worker B."""
    shared = FakeRedis()
    worker_a = _worker(RedisChatSessionRepository(shared))
    worker_b = _worker(RedisChatSessionRepository(shared))

    s = await worker_a.initialize(
        tenant=_tenant(), client_info=ClientInfo(name="claude", version="1"), client_protocol_version=_PROTOCOL
    )
    await worker_b.mark_initialized(session_id=s.id, tenant=_tenant())
    # Before the fix this raised "requires a completed initialize handshake".
    await worker_b.assert_method_allowed(session_id=s.id, tenant=_tenant(), method="tools/call")


@pytest.mark.asyncio
async def test_the_in_memory_store_really_does_fail_across_workers():
    """Proof the scenario above reproduces the bug rather than passing
    vacuously: per-process stores cannot see each other's sessions."""
    worker_a = _worker(InMemoryChatSessionRepository())
    worker_b = _worker(InMemoryChatSessionRepository())
    s = await worker_a.initialize(
        tenant=_tenant(), client_info=ClientInfo(name="claude", version="1"), client_protocol_version=_PROTOCOL
    )
    with pytest.raises(SessionNotFoundError):
        await worker_b.mark_initialized(session_id=s.id, tenant=_tenant())


@pytest.mark.asyncio
async def test_a_foreign_tenant_still_cannot_load_the_session():
    """Moving storage must not weaken tenant isolation: the stored tenant id is
    what ChatApplicationService._load_owned compares against."""
    shared = FakeRedis()
    worker_a = _worker(RedisChatSessionRepository(shared))
    worker_b = _worker(RedisChatSessionRepository(shared))
    s = await worker_a.initialize(
        tenant=_tenant("Tenant-a"),
        client_info=ClientInfo(name="claude", version="1"),
        client_protocol_version=_PROTOCOL,
    )
    with pytest.raises(SessionNotFoundError):
        await worker_b.get_for_tenant(session_id=s.id, tenant=_tenant("Tenant-b"))


@pytest.mark.asyncio
async def test_every_field_round_trips():
    shared = FakeRedis()
    repo = RedisChatSessionRepository(shared)
    svc = _worker(repo)
    s = await svc.initialize(
        tenant=_tenant(), client_info=ClientInfo(name="claude", version="9.9"), client_protocol_version=_PROTOCOL
    )
    await svc.mark_initialized(session_id=s.id, tenant=_tenant())

    back = await RedisChatSessionRepository(shared).get(SessionId(str(s.id)))
    assert back is not None
    assert back.state is SessionState.READY
    assert back.tenant == _tenant()
    assert back.client_info == ClientInfo(name="claude", version="9.9")
    assert str(back.protocol_version) == _PROTOCOL


@pytest.mark.asyncio
async def test_every_save_resets_the_idle_ttl_and_no_credential_is_stored():
    shared = FakeRedis()
    repo = RedisChatSessionRepository(shared, ttl_seconds=900)
    s = await _worker(repo).initialize(
        tenant=_tenant(), client_info=ClientInfo(name="claude", version="1"), client_protocol_version=_PROTOCOL
    )

    key = f"{KEY_PREFIX}{s.id}"
    assert shared.ttl[key] == 900
    stored = json.loads(shared.data[key])
    flat = json.dumps(stored).lower()
    for forbidden in ("authorization", "bearer", "api_key", "x-api-key", "token"):
        assert forbidden not in flat, f"session record must never carry a credential ({forbidden})"


@pytest.mark.asyncio
async def test_unreadable_record_is_an_unknown_session_not_a_crash():
    shared = FakeRedis()
    shared.data[f"{KEY_PREFIX}bad"] = "{not json"
    assert await RedisChatSessionRepository(shared).get(SessionId("bad")) is None


@pytest.mark.asyncio
async def test_delete_reports_whether_the_session_existed():
    shared = FakeRedis()
    repo = RedisChatSessionRepository(shared)
    s = await _worker(repo).initialize(
        tenant=_tenant(), client_info=ClientInfo(name="claude", version="1"), client_protocol_version=_PROTOCOL
    )
    assert await repo.delete(s.id) is True
    assert await repo.delete(s.id) is False
