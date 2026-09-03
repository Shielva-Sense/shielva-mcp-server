"""A bot must resolve from the WORKSPACE row, not whichever row Mongo returns.

Regression test. ``customerService`` holds one row per member plus a workspace
row, and core-api moved bots/knowledge/groups onto the workspace row, leaving
member rows with ``bots: []``. The old lookup was ``find_one({"tenant_id"})``,
so it could return a member row, fail to find the bot, and fall back to
``_get_mock_bot`` — answering with mock config instead of the real prompt and
knowledge, silently. Verified against production data before fixing.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.registry.bot_registry import WORKSPACE_SCOPE, BotRegistry

BOT_ID = "bot-real"
TENANT = "Tenant-1"

REAL_BOT = {"id": BOT_ID, "name": "Real bot", "kbs": ["kb-1"]}


class _Coll:
    """Minimal find_one honouring the filters this code actually sends."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    async def find_one(self, flt: dict[str, Any]) -> dict[str, Any] | None:
        for d in self.docs:
            if d.get("tenant_id") != flt.get("tenant_id"):
                continue
            if "scope" in flt and d.get("scope") != flt["scope"]:
                continue
            if "bots.id" in flt:
                if not any(b.get("id") == flt["bots.id"] for b in d.get("bots") or []):
                    continue
            return d
        return None


class _DB:
    def __init__(self, coll: _Coll) -> None:
        self.customerService = coll


class _Client:
    def __init__(self, coll: _Coll) -> None:
        self._db = _DB(coll)

    def __getitem__(self, _name: str) -> _DB:
        return self._db


def _registry(docs: list[dict[str, Any]]) -> BotRegistry:
    reg = BotRegistry(mongodb_client=_Client(_Coll(docs)))
    reg._cache_ttl = 0  # never serve a cached answer inside a test
    return reg


@pytest.mark.asyncio
async def test_resolves_from_workspace_row_when_member_row_is_empty() -> None:
    """The exact production shape: member row first, and empty."""
    reg = _registry(
        [
            {"tenant_id": TENANT, "user_email": "someone@example.com", "bots": []},
            {"tenant_id": TENANT, "scope": WORKSPACE_SCOPE, "bots": [REAL_BOT]},
        ]
    )
    bot = await reg.get_bot(BOT_ID, TENANT)
    assert bot["id"] == BOT_ID
    assert bot["name"] == "Real bot", "fell back to the mock bot"


@pytest.mark.asyncio
async def test_still_resolves_a_tenant_not_yet_migrated() -> None:
    """No workspace row yet: the bot is still on the member row."""
    reg = _registry([{"tenant_id": TENANT, "user_email": "a@b.c", "bots": [REAL_BOT]}])
    bot = await reg.get_bot(BOT_ID, TENANT)
    assert bot["name"] == "Real bot"


@pytest.mark.asyncio
async def test_empty_workspace_row_does_not_hide_an_unmigrated_bot() -> None:
    """core-api creates the workspace row EMPTY on first read, so an empty
    workspace row is not proof the bot is absent."""
    reg = _registry(
        [
            {"tenant_id": TENANT, "scope": WORKSPACE_SCOPE, "bots": []},
            {"tenant_id": TENANT, "user_email": "a@b.c", "bots": [REAL_BOT]},
        ]
    )
    bot = await reg.get_bot(BOT_ID, TENANT)
    assert bot["name"] == "Real bot"


@pytest.mark.asyncio
async def test_groups_come_from_the_workspace_row_while_bots_are_migrating() -> None:
    """Half-migrated: bot on the member row, KB-groups already on the
    workspace row. The bot's group-assigned KB must still resolve."""
    reg = _registry(
        [
            {
                "tenant_id": TENANT,
                "scope": WORKSPACE_SCOPE,
                "bots": [],
                "kb_groups": [{"group_id": "kbg-1", "kb_ids": ["kb-from-group"]}],
            },
            {
                "tenant_id": TENANT,
                "user_email": "a@b.c",
                "bots": [{**REAL_BOT, "kb_group_ids": ["kbg-1"]}],
            },
        ]
    )
    bot = await reg.get_bot(BOT_ID, TENANT)
    assert "kb-from-group" in bot["kb_ids"]
    assert "kb-1" in bot["kb_ids"]
