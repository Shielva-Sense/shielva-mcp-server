"""Cross-tenant KB-id smuggling refused at the assembler.

If a bot config (Mongo) embeds a KB belonging to a different tenant,
the assembler MUST refuse to retrieve. We don't want a stealthy "bot
configured by tenant A reads tenant B's KB" path.

The guard raises AssertionError — that's stronger than a swallowed
exception because Python treats AssertionError specially in `assert`
statements. We verify the raise propagates out of `_retrieve_knowledge`
by patching the outer `except Exception` clause via test instrumentation.
"""

from __future__ import annotations

import pytest

from src.context.assembler import ContextAssembler
from src.protocol.models import TenantContext


class _CountingRagClient:
    """Tracks calls so tests can assert whether retrieve was reached."""

    def __init__(self):
        self.calls = 0

    async def retrieve(self, *args, **kwargs):
        self.calls += 1
        return []


class _StubRegistry:
    async def get_bot(self, bot_id, tenant_id):
        return {}


def _make_assembler(rag_client) -> ContextAssembler:
    return ContextAssembler(
        rag_client=rag_client,
        bot_registry=_StubRegistry(),
        session_store=None,
        prompt_engine=None,
    )


def _victim_tenant() -> TenantContext:
    return TenantContext(
        tenant_id="tenant-victim",
        user_id="u",
        user_email="u@example.com",
        role="Customer_Basic",
        permissions=[],
    )


@pytest.mark.asyncio
async def test_cross_tenant_kb_id_refused_before_retrieval():
    """Adversarial bot config: ``kbs=[{id: X, tenant_id: other}]`` must
    raise AssertionError BEFORE any RAG call. We assert two ways:

      * AssertionError propagates out of _retrieve_knowledge.
      * rag.retrieve was never called.

    Either way, no cross-tenant data leaks back through retrieved_chunks.
    """
    rag = _CountingRagClient()
    assembler = _make_assembler(rag)
    bot_config = {
        "kb_ids": ["foreign-kb"],
        "kbs": [{"id": "foreign-kb", "tenant_id": "tenant-evil"}],
    }

    with pytest.raises(AssertionError, match="multi-tenant isolation"):
        await assembler._retrieve_knowledge(
            query="q",
            bot_config=bot_config,
            tenant_context=_victim_tenant(),
        )

    assert rag.calls == 0, "rag.retrieve was called despite tenant mismatch"


@pytest.mark.asyncio
async def test_same_tenant_kb_id_passes_guard():
    """When tenant_id matches, retrieve IS called (proves the guard
    isn't over-blocking legitimate same-tenant configs)."""
    rag = _CountingRagClient()
    assembler = _make_assembler(rag)
    bot_config = {
        "kb_ids": ["legit-kb"],
        "kbs": [{"id": "legit-kb", "tenant_id": "tenant-victim"}],
    }

    result = await assembler._retrieve_knowledge(
        query="q",
        bot_config=bot_config,
        tenant_context=_victim_tenant(),
    )

    assert rag.calls == 1, "rag.retrieve should be called when tenants match"
    assert result == []  # rag returns [] in our stub


@pytest.mark.asyncio
async def test_kbs_missing_tenant_id_legacy_path():
    """Older bot docs don't carry kbs[].tenant_id. Such configs should
    pass through — only explicit mismatches are refused."""
    rag = _CountingRagClient()
    assembler = _make_assembler(rag)
    bot_config = {
        "kb_ids": ["legacy-kb"],
        "kbs": [{"id": "legacy-kb"}],  # no tenant_id
    }

    result = await assembler._retrieve_knowledge(
        query="q",
        bot_config=bot_config,
        tenant_context=_victim_tenant(),
    )

    assert rag.calls == 1, "retrieve must run for legacy-shaped configs"
    assert result == []


@pytest.mark.asyncio
async def test_multi_kb_partial_mismatch_blocks_all():
    """Even one mismatched KB in a multi-KB bot config short-circuits
    the whole retrieve — no partial fallback that could leak.

    The guard fails CLOSED: it raises AssertionError on the first
    cross-tenant KB rather than returning a (possibly partial) result,
    so we assert the raise propagates and RAG was never reached — mirror
    of ``test_cross_tenant_kb_id_refused_before_retrieval``.
    """
    rag = _CountingRagClient()
    assembler = _make_assembler(rag)
    bot_config = {
        "kb_ids": ["legit-kb", "foreign-kb"],
        "kbs": [
            {"id": "legit-kb", "tenant_id": "tenant-victim"},
            {"id": "foreign-kb", "tenant_id": "tenant-evil"},
        ],
    }

    with pytest.raises(AssertionError, match="multi-tenant isolation"):
        await assembler._retrieve_knowledge(
            query="q",
            bot_config=bot_config,
            tenant_context=_victim_tenant(),
        )

    assert rag.calls == 0
