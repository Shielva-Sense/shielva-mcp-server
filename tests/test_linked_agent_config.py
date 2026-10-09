"""An agent LINKED to a workspace answers with its owner's config and knowledge — for that workspace only.

core-api assigns an agent to a customer's workspace by a single link
(``agent_workspace_links``: ``{_id: bot_id, owner_tenant_id, tenant_id}``);
the agent is never copied. A turn arrives with the CUSTOMER's tenant:

* the linked workspace gets the owner's bot config, and its KBs are searched in
  the owner's collections (where they are);
* any other workspace gets nothing of it (the mock bot, no knowledge);
* a reassign takes effect on the next turn, cache or not.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.context.assembler import ContextAssembler
from src.protocol.models import TenantContext
from src.registry.bot_registry import AGENT_LINKS, WORKSPACE_SCOPE, BotRegistry

OWNER = "Tenant-platform"
CLINIC_A = "Tenant-hiwaga"
CLINIC_B = "Tenant-other"
AGENT = "agent-booking"


def _match(doc: dict[str, Any], flt: dict[str, Any]) -> bool:
    for key, want in flt.items():
        if key == "bots.id":
            if not any(b.get("id") == want for b in doc.get("bots") or []):
                return False
        elif doc.get(key) != want:
            return False
    return True


class _Coll:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    async def find_one(self, flt: dict[str, Any], _projection: Any = None) -> dict[str, Any] | None:
        return next((d for d in self.docs if _match(d, flt)), None)


class _DB:
    def __init__(self, customers: _Coll, links: _Coll) -> None:
        self.customerService = customers
        self._links = links

    def __getitem__(self, name: str) -> _Coll:
        assert name == AGENT_LINKS
        return self._links


class _Client:
    def __init__(self, db: _DB) -> None:
        self._db = db

    def __getitem__(self, _name: str) -> _DB:
        return self._db


@pytest.fixture
def links() -> _Coll:
    return _Coll([{"_id": AGENT, "owner_tenant_id": OWNER, "tenant_id": CLINIC_A}])


@pytest.fixture
def registry(links: _Coll) -> BotRegistry:
    customers = _Coll(
        [
            {
                "tenant_id": OWNER,
                "scope": WORKSPACE_SCOPE,
                "bots": [{"id": AGENT, "name": "Booking agent", "kbs": ["kb-faq"]}],
            },
            {"tenant_id": CLINIC_A, "scope": WORKSPACE_SCOPE, "bots": []},
            {"tenant_id": CLINIC_B, "scope": WORKSPACE_SCOPE, "bots": []},
        ]
    )
    return BotRegistry(mongodb_client=_Client(_DB(customers, links)))


@pytest.mark.asyncio
async def test_the_linked_workspace_gets_the_owners_config(registry: BotRegistry) -> None:
    bot = await registry.get_bot(AGENT, CLINIC_A)
    assert bot["name"] == "Booking agent"
    assert bot["kb_ids"] == ["kb-faq"]
    assert bot["config_tenant_id"] == OWNER


@pytest.mark.asyncio
async def test_a_workspace_it_is_not_linked_to_gets_none_of_it(registry: BotRegistry) -> None:
    bot = await registry.get_bot(AGENT, CLINIC_B)
    assert bot["name"] != "Booking agent"
    assert "kb-faq" not in (bot.get("kb_ids") or [])
    assert bot.get("config_tenant_id") in (None, CLINIC_B)


@pytest.mark.asyncio
async def test_a_reassign_takes_effect_on_the_next_turn_even_with_a_warm_cache(
    registry: BotRegistry, links: _Coll
) -> None:
    assert registry._cache_ttl > 0
    assert (await registry.get_bot(AGENT, CLINIC_A))["name"] == "Booking agent"
    links.docs[0]["tenant_id"] = CLINIC_B
    assert (await registry.get_bot(AGENT, CLINIC_A))["name"] != "Booking agent"
    assert (await registry.get_bot(AGENT, CLINIC_B))["name"] == "Booking agent"


@pytest.mark.asyncio
async def test_the_owners_own_turns_are_unchanged(registry: BotRegistry) -> None:
    bot = await registry.get_bot(AGENT, OWNER)
    assert (bot["name"], bot["config_tenant_id"]) == ("Booking agent", OWNER)


class _Rag:
    def __init__(self) -> None:
        self.tenants: list[str] = []

    async def retrieve(self, *, tenant_id: str, **_kwargs: Any) -> list[Any]:
        self.tenants.append(tenant_id)
        return []


def _tenant(tenant_id: str) -> TenantContext:
    return TenantContext(tenant_id=tenant_id, user_id="u", user_email="u@wa.invalid", permissions=[])


@pytest.mark.asyncio
async def test_a_linked_agents_knowledge_is_searched_where_it_lives(registry: BotRegistry) -> None:
    rag = _Rag()
    assembler = ContextAssembler(rag_client=rag, bot_registry=registry, session_store=None, prompt_engine=None)
    bot = await registry.get_bot(AGENT, CLINIC_A)
    await assembler._retrieve_knowledge("opening hours", bot, _tenant(CLINIC_A))
    assert rag.tenants == [OWNER]


@pytest.mark.asyncio
async def test_without_a_link_knowledge_is_the_request_tenants_own(registry: BotRegistry) -> None:
    rag = _Rag()
    assembler = ContextAssembler(rag_client=rag, bot_registry=registry, session_store=None, prompt_engine=None)
    await assembler._retrieve_knowledge("opening hours", {"kb_ids": ["kb-own"], "kbs": ["kb-own"]}, _tenant(CLINIC_B))
    assert rag.tenants == [CLINIC_B]
