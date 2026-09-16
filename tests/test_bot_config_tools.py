"""Bot-configuration tools: right service, right route, no way around the gateway."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from src.protocol.models import TenantContext
from src.tools import _caller
from src.tools import bot_config_tools as cfg


@pytest.fixture(autouse=True)
def _no_credential():
    _caller.remember_caller_credential({})
    yield
    _caller.remember_caller_credential({})


def _t():
    return TenantContext(tenant_id="t1", user_id="u1", user_email="u@example.com")


def test_no_tool_builds_its_own_http_client():
    """🚨 An httpx client here would be a second path to customer data that
    skips every decision _gateway.py makes — which is how a service token
    creeps back in and the grant map stops meaning anything."""
    src = inspect.getsource(cfg)
    assert "httpx" not in src
    assert "core-api:" not in src
    assert "cms-core:" not in src


def test_config_tools_refuse_without_the_callers_credential():
    for call in (
        cfg.shielva_list_identities(_t()),
        cfg.shielva_list_intents(_t()),
        cfg.shielva_list_decision_rules(_t()),
    ):
        out = asyncio.run(call)
        assert out["status"] == "failed"
        assert "credential" in out["error"].lower()


def test_bot_persona_is_not_the_identity_service(monkeypatch):
    """🚨 A real name collision. /cms/api/v1/identity is who the BOT is;
    /identity/api/v1/* is the identity SERVICE that manages user accounts and is
    not reachable from a customer's MCP key. Hitting the wrong one would be a
    tool that edits people instead of personas."""
    seen = {}

    async def fake(method, path, tenant, **kw):
        seen["path"] = path
        return []

    monkeypatch.setattr(cfg, "_call", fake)
    asyncio.run(cfg.shielva_list_identities(_t()))
    assert seen["path"] == "/cms/api/v1/identity"
    assert not seen["path"].startswith("/identity/")


def test_decision_rules_go_to_the_signals_rules_endpoint(monkeypatch):
    seen = {}

    async def fake(method, path, tenant, **kw):
        seen["path"] = path
        return []

    monkeypatch.setattr(cfg, "_call", fake)
    asyncio.run(cfg.shielva_list_decision_rules(_t()))
    assert seen["path"] == "/cms/api/v1/signals/rules"


def test_every_write_requests_live_updates(monkeypatch):
    seen = {}

    async def fake(method, path, tenant, **kw):
        seen.setdefault("params", []).append(kw.get("params"))
        return {}

    monkeypatch.setattr(cfg, "_call", fake)
    asyncio.run(cfg.shielva_set_bot_prompt(_t(), "b1", "be helpful"))
    asyncio.run(cfg.shielva_create_intent(_t(), "greet", "b1"))
    assert all(p and p.get("sse") == "true" for p in seen["params"])


def test_reading_config_survives_a_partial_refusal(monkeypatch):
    """🚨 A grant map may permit capabilities and withhold signals. Losing the
    whole response in that case would read as 'this bot has no configuration'."""

    async def fake(method, path, tenant, **kw):
        if "signals" in path:
            raise cfg.ToolCallError("not permitted")
        return {"ok": True}

    monkeypatch.setattr(cfg, "_call", fake)
    out = asyncio.run(cfg.shielva_get_bot_config(_t(), "b1"))
    assert out["status"] == "ok"
    assert out["capabilities"] == {"ok": True}
    assert "unavailable" in out["signals"]


def test_a_wrapped_or_bare_list_both_read(monkeypatch):
    """Upstreams differ; the model must not have to know which."""
    assert cfg._items([{"id": 1}]) == [{"id": 1}]
    assert cfg._items({"items": [{"id": 2}]}) == [{"id": 2}]
    assert cfg._items({"rules": [{"id": 3}]}, key="rules") == [{"id": 3}]
    assert cfg._items({"nope": 1}) == []


def test_config_tools_are_visible_but_not_in_a_live_bots_toolset():
    """Two flags, two surfaces: visible to an MCP client, absent from the
    runtime toolset of a bot that is answering somebody."""
    for definition, _ in cfg.BOT_CONFIG_TOOL_DEFINITIONS:
        assert definition.requires_permissions == []
        assert definition.enabled_by_default is False


def test_required_arguments_are_checked_before_any_call():
    for out in (
        asyncio.run(cfg.shielva_create_identity(_t(), "", "b1")),
        asyncio.run(cfg.shielva_update_intent(_t(), "i1", {})),
        asyncio.run(cfg.shielva_set_bot_capability(_t(), "", "voice", True)),
    ):
        assert out["status"] == "failed"
