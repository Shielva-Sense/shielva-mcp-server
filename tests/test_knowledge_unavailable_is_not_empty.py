"""A knowledge base that cannot be READ is not a knowledge base with nothing in it.

What a customer saw when the two were the same: a clinic's pooled database
connection had gone stale, every search failed and returned an empty list, the
answer step said "no relevant knowledge", and the model quoted a Botox price of
its own (2,000 EGP; the clinic's is 3,500).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.exc import OperationalError, ProgrammingError

from src.context import assembler as asm
from src.context.assembler import KNOWLEDGE_UNAVAILABLE, NO_RELEVANT_KNOWLEDGE, ContextAssembler, KnowledgeUnavailable
from src.protocol.models import SessionContext, TenantContext
from src.rag_engine.vectorstore import supabase_store as store_mod
from src.rag_engine.vectorstore.supabase_store import PgVectorStore, VectorStoreUnavailable

TENANT = "Tenant-x"


def _tenant() -> TenantContext:
    return TenantContext(tenant_id=TENANT, user_id="u1", user_email="u@example.com")


# ── the pool ─────────────────────────────────────────────────────────────────


def test_the_pool_checks_a_connection_before_handing_it_out(monkeypatch):
    made: dict[str, Any] = {}

    def _engine(url, **kwargs):
        made.update(kwargs, url=url)
        return object()

    class _Old:
        disposed = False

        def dispose(self):
            _Old.disposed = True

    class _Client:
        engine = _Old()
        Session = None

    monkeypatch.setattr(sqlalchemy, "create_engine", _engine)
    client = _Client()
    store_mod._with_live_pool(client, "postgresql://u:p@db/vecs")
    assert made["pool_pre_ping"] is True, "a dropped connection is replaced, not used"
    assert made["pool_recycle"] > 0
    assert made["connect_args"]["keepalives"] == 1
    assert _Old.disposed, "the bare pool vecs built is closed"
    assert client.Session is not None


# ── the store ────────────────────────────────────────────────────────────────


class _FailingCollection:
    def query(self, **kwargs):
        raise OperationalError("select", {}, Exception("could not send data to server: Connection timed out"))


class _Client:
    def __init__(self, session_error: Exception | None = None):
        self._session_error = session_error

    def get_collection(self, name):
        return _FailingCollection()

    def Session(self):
        error = self._session_error

        class _Sess:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **kw):
                raise error

        return _Sess()


def _store(client: Any) -> PgVectorStore:
    s = PgVectorStore(db_url="postgresql://u:p@db/vecs")
    s._client = client
    return s


def test_a_failed_vector_search_raises_instead_of_finding_nothing():
    with pytest.raises(VectorStoreUnavailable):
        asyncio.run(_store(_Client()).search(tenant_id=TENANT, kb_ids=["kb1"], query_embedding=[0.0] * 3))


def test_a_dead_connection_in_keyword_search_raises():
    dead = OperationalError("select", {}, Exception("server closed the connection unexpectedly"))
    with pytest.raises(VectorStoreUnavailable):
        asyncio.run(_store(_Client(dead)).keyword_search(tenant_id=TENANT, kb_ids=["kb1"], query="botox"))


def test_a_kb_without_a_keyword_index_still_finds_nothing_quietly():
    missing = ProgrammingError("select", {}, Exception('relation "vecs.x" does not exist'))
    assert asyncio.run(_store(_Client(missing)).keyword_search(tenant_id=TENANT, kb_ids=["kb1"], query="x")) == []


# ── the prompt ───────────────────────────────────────────────────────────────


class _BrokenRag:
    async def retrieve(self, **kwargs):
        raise VectorStoreUnavailable("vector search failed")


class _EmptyRag:
    async def retrieve(self, **kwargs):
        return []


class _Registry:
    async def get_bot(self, bot_id, tenant_id):
        return {"kb_ids": ["kb1"], "kbs": [{"id": "kb1", "tenant_id": TENANT}], "prompt_config": {}}


def _assembler(rag: Any) -> ContextAssembler:
    return ContextAssembler(rag_client=rag, bot_registry=_Registry(), session_store=None, prompt_engine=None)


def _knowledge_in_prompt(rag: Any, monkeypatch) -> str:
    seen: dict[str, str] = {}

    async def _messages(self, *, system_prompt, knowledge_context, query, session):
        seen["knowledge"] = knowledge_context
        return []

    monkeypatch.setattr(asm.ContextAssembler, "_build_messages", _messages)
    session = SessionContext(session_id="s1", tenant_context=_tenant(), bot_id="b1")
    asyncio.run(
        _assembler(rag).assemble(query="البوتوكس بكام؟", session=session, tenant_context=_tenant(), bot_id="b1")
    )
    return seen["knowledge"]


def test_an_unreadable_knowledge_base_is_said_to_the_model(monkeypatch):
    knowledge = _knowledge_in_prompt(_BrokenRag(), monkeypatch)
    assert knowledge == KNOWLEDGE_UNAVAILABLE
    assert "Do NOT state any price" in knowledge


def test_nothing_found_still_forbids_inventing_a_price(monkeypatch):
    knowledge = _knowledge_in_prompt(_EmptyRag(), monkeypatch)
    assert knowledge == NO_RELEVANT_KNOWLEDGE
    assert "Do not state prices" in knowledge


def test_retrieval_errors_surface_from_the_retriever():
    bot = {"kb_ids": ["kb1"], "kbs": [{"id": "kb1", "tenant_id": TENANT}]}
    with pytest.raises(KnowledgeUnavailable):
        asyncio.run(_assembler(_BrokenRag())._retrieve_knowledge(query="q", bot_config=bot, tenant_context=_tenant()))
