"""BotApplicationService — get/list/render_prompt with a fake repo."""

from __future__ import annotations

import pytest

from src.application.bots import BotApplicationService
from src.domain.bots.entities import Bot
from src.domain.bots.errors import BotNotFoundError
from src.domain.bots.repositories import BotRepository
from src.domain.bots.value_objects import (
    BotId,
    BotName,
    BotStatus,
    PromptTemplate,
)
from src.domain.shared.tenant import TenantContext


class _FakeRepo(BotRepository):
    def __init__(self, bots: list[Bot]) -> None:
        self._bots: dict[str, Bot] = {str(b.id): b for b in bots}

    async def get(self, bot_id: BotId, *, tenant: TenantContext) -> Bot | None:
        b = self._bots.get(str(bot_id))
        if b is None or b.tenant_id != tenant.tenant_id:
            return None
        return b

    async def list_for_tenant(self, *, tenant: TenantContext) -> list[Bot]:
        return [b for b in self._bots.values() if b.tenant_id == tenant.tenant_id]

    async def save(self, bot: Bot) -> None:
        self._bots[str(bot.id)] = bot


def _bot(bot_id: str, tenant_id: str, prompt: str) -> Bot:
    return Bot(
        id=BotId(bot_id),
        tenant_id=tenant_id,
        name=BotName(f"Bot {bot_id}"),
        description=f"desc {bot_id}",
        status=BotStatus.ACTIVE,
        prompt_template=PromptTemplate(text=prompt),
    )


def _tenant(tid: str = "t1") -> TenantContext:
    return TenantContext(tenant_id=tid, user_id="u", user_email="u@x")


@pytest.mark.asyncio
async def test_get_bot_happy_path():
    svc = BotApplicationService(repository=_FakeRepo([_bot("b1", "t1", "hi")]))
    bot = await svc.get_bot(bot_id="b1", tenant=_tenant())
    assert str(bot.id) == "b1"


@pytest.mark.asyncio
async def test_get_bot_unknown_raises_not_found():
    svc = BotApplicationService(repository=_FakeRepo([]))
    with pytest.raises(BotNotFoundError):
        await svc.get_bot(bot_id="missing", tenant=_tenant())


@pytest.mark.asyncio
async def test_get_bot_foreign_tenant_raises_not_found():
    """Cross-tenant access leaks must surface as NOT_FOUND, not
    FORBIDDEN — otherwise the existence of the bot id is leaked."""
    svc = BotApplicationService(
        repository=_FakeRepo([_bot("b1", "alice", "hi")]),
    )
    with pytest.raises(BotNotFoundError):
        await svc.get_bot(bot_id="b1", tenant=_tenant("bob"))


@pytest.mark.asyncio
async def test_list_bots_returns_tenant_bots_only():
    svc = BotApplicationService(
        repository=_FakeRepo(
            [
                _bot("b1", "alice", ""),
                _bot("b2", "alice", ""),
                _bot("b3", "bob", ""),
            ]
        )
    )
    alice_bots = await svc.list_bots(tenant=_tenant("alice"))
    assert sorted(str(b.id) for b in alice_bots) == ["b1", "b2"]


@pytest.mark.asyncio
async def test_render_prompt_substitutes_arguments():
    svc = BotApplicationService(
        repository=_FakeRepo([_bot("b1", "t1", "Hi {{name}}!")]),
    )
    rendered = await svc.render_prompt(
        bot_id="b1",
        tenant=_tenant(),
        arguments={"name": "Vivek"},
    )
    assert rendered == "Hi Vivek!"


@pytest.mark.asyncio
async def test_render_prompt_leaves_unknown_vars_intact():
    """MCP spec: client may omit args; missing variables remain
    in literal ``{{name}}`` form for client diagnostics."""
    svc = BotApplicationService(
        repository=_FakeRepo([_bot("b1", "t1", "Hi {{name}}, role {{role}}")]),
    )
    rendered = await svc.render_prompt(
        bot_id="b1",
        tenant=_tenant(),
        arguments={"name": "Vivek"},
    )
    assert rendered == "Hi Vivek, role {{role}}"
