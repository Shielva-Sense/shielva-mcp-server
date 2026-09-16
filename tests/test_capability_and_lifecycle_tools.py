"""shielva_set_bot_capability resolves a real provider; lifecycle tools hit the real routes.

🚨 THE CAPABILITY BUG: the tool sent ``{"enabled": true}``. The endpoint stores a
PROVIDER CHOICE from ``connector`` + ``action`` and never reads ``enabled``; an
empty connector means "clear it". So enabling a capability REMOVED it, reported
``updated``, and every sms/mail/calendar/crm node on the bot did nothing.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from src.protocol.models import TenantContext
from src.tools import bot_config_tools as cfg
from src.tools import lifecycle_tools as life


def _t() -> TenantContext:
    return TenantContext(tenant_id="Tenant-x", user_id="u1", user_email="u@example.com")


class _Recorder:
    """Stands in for _gateway.call: records the request, answers like core-api."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, method, path, tenant_context, *, params=None, json=None):
        self.calls.append({"method": method, "path": path, "params": params, "json": json})
        return self.reply(json) if callable(self.reply) else self.reply


class _Catalogue:
    def __init__(self, providers):
        self.providers = providers
        self.asked: list[str] = []

    async def providers_for(self, capability, tenant):
        self.asked.append(capability)
        return self.providers


def _stores_what_it_is_sent(capability):
    def reply(body):
        if "connector" not in (body or {}):
            return {"ok": True}  # the per-bot connector link that follows an enable
        row = {"connector": body["connector"], "action": body["action"]} if body["connector"] else None
        return {"bot_id": "b1", "capabilities": ({capability: row} if row else {})}

    return reply


# ── the capability fix ────────────────────────────────────────────────


def test_enabling_sends_a_provider_not_an_enabled_flag(monkeypatch):
    rec = _Recorder(_stores_what_it_is_sent("crm.create_lead"))
    cat = _Catalogue([("shielva_sales", "create_lead")])
    monkeypatch.setattr(cfg, "_call", rec)
    monkeypatch.setattr(cfg, "_connector_catalogue", lambda: cat)

    out = asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "crm.create_lead", True))

    body = next(c["json"] for c in rec.calls if "/capabilities/" in c["path"])
    assert "enabled" not in body, "the endpoint ignores `enabled` — sending only it cleared the row"
    assert body == {"connector": "shielva_sales", "action": "create_lead"}
    assert out["status"] == "updated"
    assert out["enabled"] is True
    assert out["provider"] == {"connector": "shielva_sales", "action": "create_lead"}


def test_a_named_connector_is_used_without_consulting_the_catalogue(monkeypatch):
    rec = _Recorder(_stores_what_it_is_sent("calendar.create_event"))
    cat = _Catalogue([("outlook_calendar", "create_event")])
    monkeypatch.setattr(cfg, "_call", rec)
    monkeypatch.setattr(cfg, "_connector_catalogue", lambda: cat)

    out = asyncio.run(
        cfg.shielva_set_bot_capability(
            _t(), "b1", "calendar.create_event", True, connector="google_calendar", action="create_event"
        )
    )
    assert next(c["json"] for c in rec.calls if "/capabilities/" in c["path"]) == {
        "connector": "google_calendar",
        "action": "create_event",
    }
    assert cat.asked == []
    assert out["provider"]["connector"] == "google_calendar"


def test_disabling_sends_an_empty_connector(monkeypatch):
    rec = _Recorder(_stores_what_it_is_sent("mail.send"))
    monkeypatch.setattr(cfg, "_call", rec)
    out = asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "mail.send", False))
    assert rec.calls[0]["json"] == {"connector": "", "action": ""}
    assert out["enabled"] is False


def test_no_installed_provider_fails_loudly_and_writes_nothing(monkeypatch):
    rec = _Recorder({})
    monkeypatch.setattr(cfg, "_call", rec)
    monkeypatch.setattr(cfg, "_connector_catalogue", lambda: _Catalogue([]))
    out = asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "sms.send", True))
    assert out["status"] == "failed"
    assert "sms.send" in out["error"]
    assert rec.calls == [], "must not write a cleared row when nothing provides the capability"


def test_a_write_that_did_not_stick_is_reported_not_trusted(monkeypatch):
    """Render the result of the write, not what we hoped the write did."""
    rec = _Recorder({"bot_id": "b1", "capabilities": {}})
    monkeypatch.setattr(cfg, "_call", rec)
    monkeypatch.setattr(cfg, "_connector_catalogue", lambda: _Catalogue([("google_gmail_connector", "send_email")]))
    out = asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "mail.send", True))
    assert out["status"] == "failed"


def test_other_installed_providers_are_surfaced(monkeypatch):
    rec = _Recorder(_stores_what_it_is_sent("mail.send"))
    monkeypatch.setattr(cfg, "_call", rec)
    monkeypatch.setattr(
        cfg,
        "_connector_catalogue",
        lambda: _Catalogue([("google_gmail_connector", "send_email"), ("outlook_mail", "send_mail")]),
    )
    out = asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "mail.send", True))
    assert out["provider"]["connector"] == "google_gmail_connector"
    assert out["other_installed_providers"] == [{"connector": "outlook_mail", "action": "send_mail"}]


def test_providers_for_only_offers_installed_connectors_declaring_the_capability():
    from src.infrastructure.tools.connector_catalogue import ConnectorToolCatalogue

    cat = ConnectorToolCatalogue(base_url="http://x")

    async def installed(_tenant):
        return {"shielva_sales", "google_calendar"}

    async def types():
        return {
            "shielva_sales": {"capability_actions": [{"capability": "crm.create_lead", "action": "create_lead"}]},
            "google_calendar": {
                "capability_actions": [{"capability": "calendar.create_event", "action": "create_event"}]
            },
            # declares CRM but is NOT installed — must never be offered
            "hubspot": {"capability_actions": [{"capability": "crm.create_lead", "action": "create_contact"}]},
        }

    cat._installed = installed  # type: ignore[method-assign]
    cat._types = types  # type: ignore[method-assign]
    assert asyncio.run(cat.providers_for("crm.create_lead", _t())) == [("shielva_sales", "create_lead")]
    assert asyncio.run(cat.providers_for("sms.send", _t())) == []


# ── lifecycle tools ───────────────────────────────────────────────────


def test_lifecycle_tools_never_build_their_own_http_client():
    src = inspect.getsource(life)
    assert "httpx" not in src
    assert "core-api:" not in src


@pytest.mark.parametrize(
    ("call", "method", "path", "reply", "key"),
    [
        (lambda: life.shielva_create_bot(_t(), "London Estates"), "POST", "/bots/create", {"bot_id": "new1"}, "bot_id"),
        (lambda: life.shielva_chat_with_bot(_t(), "b1", "hi"), "POST", "/bots/b1/chat", {"reply": "hello"}, "response"),
        (lambda: life.shielva_deploy_bot(_t(), "b1"), "POST", "/bots/b1/deploy", {"status": "queued"}, "deployment"),
        (
            lambda: life.shielva_create_knowledge_base(_t(), "Listings"),
            "POST",
            "/knowledge/create",
            {"kb_id": "kb1", "id": "kb1"},
            "kb_id",
        ),
        (
            lambda: life.shielva_ingest_url(_t(), "kb1", "https://example.com"),
            "POST",
            "/knowledge/kb1/ingest-url",
            {"ok": True},
            "response",
        ),
        (
            lambda: life.shielva_set_bot_knowledge_groups(_t(), "b1", ["g1"]),
            "PUT",
            "/bots/b1/kb-groups",
            {"ok": True},
            "response",
        ),
    ],
)
def test_each_lifecycle_tool_hits_the_real_route(monkeypatch, call, method, path, reply, key):
    rec = _Recorder(reply)
    monkeypatch.setattr(life, "_call", rec)
    out = asyncio.run(call())
    assert rec.calls[0]["method"] == method
    assert rec.calls[0]["path"] == path
    assert key in out


def test_create_bot_without_an_id_back_is_a_failure(monkeypatch):
    monkeypatch.setattr(life, "_call", _Recorder({"success": True}))
    assert asyncio.run(life.shielva_create_bot(_t(), "X"))["status"] == "failed"


def test_lifecycle_tools_are_registered_and_visible():
    from src.tools.lifecycle_tools import LIFECYCLE_TOOL_DEFINITIONS

    names = {d.name for d, _ in LIFECYCLE_TOOL_DEFINITIONS}
    assert {"shielva_create_bot", "shielva_chat_with_bot", "shielva_deploy_bot"} <= names
    for d, _ in LIFECYCLE_TOOL_DEFINITIONS:
        assert d.requires_permissions == [], f"{d.name} would be hidden from tools/list"
        assert d.enabled_by_default is False, f"{d.name} must never be in a live bot's runtime toolset"


def test_enabling_a_capability_also_enables_its_connector_for_the_bot(monkeypatch):
    """🚨 Without this the bot is refused on every call: the runtime gates on the
    bot's enabled-connector list, which the capability binding does not touch."""
    calls: list[tuple[str, str, Any]] = []

    async def gw(method, path, tenant_context, *, params=None, json=None):
        calls.append((method, path, json))
        if path.endswith("/capabilities/calendar.create_event"):
            return {
                "capabilities": {"calendar.create_event": {"connector": json["connector"], "action": json["action"]}}
            }
        return {"ok": True}

    monkeypatch.setattr(cfg, "_call", gw)
    monkeypatch.setattr(cfg, "_connector_catalogue", lambda: _Catalogue([("google_calendar", "create_event")]))
    out = asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "calendar.create_event", True))

    link = [c for c in calls if c[1] == "/bots/b1/api-connectors"]
    assert link, "the connector was never enabled for the bot"
    assert link[0][2] == {"connector_id": "google_calendar", "enabled": True, "action": "link"}
    assert out["connector_enabled_for_bot"] is True


def test_disabling_a_capability_does_not_unlink_the_connector(monkeypatch):
    calls: list[str] = []

    async def gw(method, path, tenant_context, *, params=None, json=None):
        calls.append(path)
        return {"capabilities": {}}

    monkeypatch.setattr(cfg, "_call", gw)
    asyncio.run(cfg.shielva_set_bot_capability(_t(), "b1", "mail.send", False))
    assert "/bots/b1/api-connectors" not in calls


def test_add_flow_node_can_wire_a_branch_output(monkeypatch):
    from src.tools import flow_tools

    saved: dict[str, Any] = {}

    async def gw(method, path, tenant_context, *, params=None, json=None):
        if method == "GET":
            return {"nodes": [{"id": "route", "data": {"kind": "classify"}}], "edges": []}
        saved.update(json or {})
        return {"ok": True}

    monkeypatch.setattr(flow_tools, "_call", gw)
    asyncio.run(
        flow_tools.shielva_add_flow_node(
            _t(), "b1", {"id": "ask_name", "data": {"kind": "ask"}}, edge_from="route", edge_handle="c_buy"
        )
    )
    edge = saved["edges"][-1]
    assert edge["source"] == "route"
    assert edge["sourceHandle"] == "c_buy", "a classify case routes only down its own handle"
