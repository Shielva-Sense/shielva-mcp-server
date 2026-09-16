"""Flow tools call the platform AS THE CALLER, and never around the grants."""

from __future__ import annotations

import asyncio

import pytest

from src.protocol.models import TenantContext
from src.tools import _caller, flow_tools


@pytest.fixture(autouse=True)
def _clear_credential():
    _caller.remember_caller_credential({})
    yield
    _caller.remember_caller_credential({})


class _Headers(dict):
    def get(self, key, default=None):
        want = key.lower()
        for k, v in self.items():
            if k.lower() == want:
                return v
        return default


def _tenant():
    return TenantContext(tenant_id="t1", user_id="u1", user_email="u@example.com")


# ── the security property ────────────────────────────────────────────


def test_a_tool_refuses_rather_than_calling_without_the_callers_credential():
    """🚨 THE point of this module. A tool that fell back to an unauthenticated
    or service-authenticated call would reach every bot in every workspace, and
    the MCP grant map would be bypassed entirely — still configured, protecting
    nothing."""
    out = asyncio.run(flow_tools.shielva_list_bots(_tenant()))
    assert out["status"] == "failed"
    assert "credential" in out["error"].lower()


def test_the_credential_is_captured_from_either_scheme():
    _caller.remember_caller_credential(_Headers({"X-API-Key": "shv_mcp_abc"}))
    assert _caller.caller_auth_headers() == {"X-API-Key": "shv_mcp_abc"}

    _caller.remember_caller_credential(_Headers({"authorization": "Bearer xyz"}))
    assert _caller.caller_auth_headers() == {"Authorization": "Bearer xyz"}


def test_the_credential_is_never_put_on_the_tenant_context():
    """🚨 TenantContext is logged across this service. A credential on it is one
    logger.info(tenant_context=...) away from printing customer secrets."""
    _caller.remember_caller_credential(_Headers({"X-API-Key": "shv_mcp_secret"}))
    dumped = _tenant().model_dump_json()
    assert "shv_mcp_secret" not in dumped


def test_callers_cannot_mutate_each_others_forwarded_headers():
    _caller.remember_caller_credential(_Headers({"X-API-Key": "k"}))
    first = _caller.caller_auth_headers()
    first["X-Injected"] = "nope"
    assert "X-Injected" not in _caller.caller_auth_headers()


def test_the_transport_captures_the_credential_before_building_context():
    """Pins the wiring: if the capture is removed, every flow tool starts
    refusing and this says why."""
    import inspect

    from src.interface.mcp_jsonrpc import transport

    src = inspect.getsource(transport)
    assert "remember_caller_credential(request.headers)" in src
    assert src.index("remember_caller_credential(request.headers)") < src.index("return TenantContext(")


# ── behaviour ────────────────────────────────────────────────────────


def test_tools_go_through_the_gateway_not_straight_to_the_service():
    """🚨 core-api directly would skip the grants plugin, which is the only
    thing deciding whether this credential may touch this route."""
    import inspect

    src = inspect.getsource(flow_tools)
    assert "_GATEWAY_URL" in src
    assert "core-api:" not in src
    assert "http://bot:" not in src


def test_every_write_asks_for_live_updates():
    """The owner asked to watch nodes appear; a write that does not request
    streaming is a write nobody sees."""
    assert flow_tools._LIVE == {"sse": "true"}


def test_adding_a_node_refuses_a_duplicate_id(monkeypatch):
    async def fake_call(method, path, tenant, **kw):
        return {"flow": {"nodes": [{"id": "n1"}], "edges": []}}

    monkeypatch.setattr(flow_tools, "_call", fake_call)
    out = asyncio.run(flow_tools.shielva_add_flow_node(_tenant(), "b1", {"id": "n1"}))
    assert out["status"] == "failed"
    assert "already exists" in out["error"]


def test_adding_a_node_refuses_when_the_flow_changed_underneath(monkeypatch):
    """🚨 Read-modify-write against a graph a person may be dragging nodes in.
    Without this the last writer silently wins and their node vanishes with no
    error anywhere."""

    async def fake_call(method, path, tenant, **kw):
        return {"flow": {"nodes": [{"id": "a"}, {"id": "b"}], "edges": []}}

    monkeypatch.setattr(flow_tools, "_call", fake_call)
    out = asyncio.run(flow_tools.shielva_add_flow_node(_tenant(), "b1", {"id": "c"}, expected_node_count=1))
    assert out["status"] == "failed"
    assert "changed underneath" in out["error"]


def test_adding_a_node_appends_and_links(monkeypatch):
    seen: dict = {}

    async def fake_call(method, path, tenant, **kw):
        if method == "GET":
            return {"flow": {"nodes": [{"id": "start"}], "edges": []}}
        seen["body"] = kw.get("json")
        seen["params"] = kw.get("params")
        return {"flow": seen["body"]}

    monkeypatch.setattr(flow_tools, "_call", fake_call)
    out = asyncio.run(flow_tools.shielva_add_flow_node(_tenant(), "b1", {"id": "ask"}, edge_from="start"))
    assert out["status"] == "added"
    assert [n["id"] for n in seen["body"]["nodes"]] == ["start", "ask"]
    assert seen["body"]["edges"] == [{"id": "start->ask", "source": "start", "target": "ask"}]
    assert seen["params"] == {"sse": "true"}


def test_an_edge_cannot_point_at_a_node_that_is_not_there(monkeypatch):
    async def fake_call(method, path, tenant, **kw):
        return {"flow": {"nodes": [{"id": "start"}], "edges": []}}

    monkeypatch.setattr(flow_tools, "_call", fake_call)
    out = asyncio.run(flow_tools.shielva_add_flow_node(_tenant(), "b1", {"id": "n"}, edge_from="ghost"))
    assert out["status"] == "failed"


def test_a_refusal_names_what_was_refused(monkeypatch):
    """A bare 403 sends a model into a retry loop; naming the route tells the
    person reading the transcript what to change."""
    import inspect

    src = inspect.getsource(flow_tools._call)
    assert "MCP API Access" in src
