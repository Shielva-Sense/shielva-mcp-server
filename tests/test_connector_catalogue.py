"""A tenant's installed connectors, exposed as MCP tools.

The test that matters most is `test_a_tool_the_listing_never_offered_cannot_be_called`:
`tools/list` is discovery, not a boundary, and nothing obliges a caller to have
read it before calling something.
"""

from __future__ import annotations

import pytest

from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.value_objects import ToolName, ToolSchema
from src.infrastructure.tools.connector_catalogue import (
    ConnectorToolCatalogue,
    _schema_for,
    _split,
    tool_name_for,
)

GMAIL = {
    "type": "google_gmail_connector",
    "display_name": "Gmail",
    "apis": [
        {
            "id": "send_email",
            "description": "Send an email via the Gmail API.",
            "params": [
                {"key": "to", "type": "text", "required": True, "label": "To"},
                {"key": "subject", "type": "text", "required": True, "label": "Subject"},
                {"key": "attachments", "type": "json", "required": False, "help": "List of files"},
            ],
        },
        {"id": "health_check", "description": "ping"},
        {"id": "install", "description": "plumbing"},
    ],
}
HUBSPOT = {"type": "hubspot_connector", "display_name": "HubSpot", "apis": [{"id": "create_contact"}]}


def _tenant(tid: str = "Tenant-A") -> TenantContext:
    return TenantContext(tenant_id=tid, user_id="u1", user_email="u@t.test")


class _Cat(ConnectorToolCatalogue):
    """The catalogue with its two upstream reads stubbed."""

    def __init__(self, installed: set[str], types: dict) -> None:
        super().__init__(base_url="http://runtime.invalid")
        self._fake_installed = installed
        self._fake_types = types
        self.calls: list[tuple[str, str, dict]] = []

    async def _installed(self, tenant):  # type: ignore[override]
        return self._fake_installed

    async def _types(self):  # type: ignore[override]
        return self._fake_types


# ── naming ──────────────────────────────────────────────────────────


def test_the_name_drops_the_suffix_every_connector_carries():
    assert tool_name_for("google_gmail_connector", "send_email") == "google_gmail__send_email"
    assert _split("google_gmail__send_email") == ("google_gmail_connector", "send_email")


def test_a_name_that_is_not_a_connector_tool_is_rejected_not_guessed():
    assert _split("rag_query") == ("", "")
    assert _split("") == ("", "")


# ── schema ──────────────────────────────────────────────────────────


def test_params_become_a_json_schema_an_llm_can_satisfy():
    schema = _schema_for(GMAIL["apis"][0]).json_schema
    assert schema["type"] == "object"
    assert schema["properties"]["to"]["type"] == "string"
    assert schema["properties"]["attachments"]["type"] == "object"
    assert sorted(schema["required"]) == ["subject", "to"]
    # A param with no description is one the model fills by guessing.
    assert schema["properties"]["attachments"]["description"] == "List of files"


# ── listing ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_only_installed_connectors_are_listed():
    cat = _Cat({"google_gmail_connector"}, {"google_gmail_connector": GMAIL, "hubspot_connector": HUBSPOT})
    names = [str(t.name) for t in await cat.list_for(_tenant())]
    assert "google_gmail__send_email" in names
    assert not any(n.startswith("hubspot") for n in names), "an uninstalled connector must not be offered"


@pytest.mark.asyncio
async def test_plumbing_methods_are_not_offered_as_tools():
    """`install` and `authorize` would let a model re-run an OAuth handshake
    mid-conversation; `health_check` is noise."""
    cat = _Cat({"google_gmail_connector"}, {"google_gmail_connector": GMAIL})
    names = [str(t.name) for t in await cat.list_for(_tenant())]
    assert names == ["google_gmail__send_email"]


@pytest.mark.asyncio
async def test_a_tenant_with_nothing_installed_gets_an_empty_list():
    cat = _Cat(set(), {"google_gmail_connector": GMAIL})
    assert await cat.list_for(_tenant()) == []


@pytest.mark.asyncio
async def test_an_installed_connector_missing_from_the_catalogue_is_skipped():
    """Installed, but the published snapshot moved on. Offering it would be a
    tool nothing can run."""
    cat = _Cat({"google_gmail_connector", "ghost_connector"}, {"google_gmail_connector": GMAIL})
    names = [str(t.name) for t in await cat.list_for(_tenant())]
    assert names == ["google_gmail__send_email"]


# ── the boundary ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_tool_the_listing_never_offered_cannot_be_called():
    """🚨 THE test. `tools/list` is discovery, not a security boundary — a
    client can call a name it guessed. The installed-set check has to run again
    inside execute, or a guessed name reaches a connector this workspace never
    installed, on whatever credential the runtime resolves."""
    cat = _Cat({"google_gmail_connector"}, {"google_gmail_connector": GMAIL, "hubspot_connector": HUBSPOT})
    forged = Tool(
        name=ToolName("hubspot__create_contact"),
        description="guessed by the caller",
        input_schema=ToolSchema(json_schema={}),
    )
    result = await cat.execute(tool=forged, arguments={"email": "a@b.c"}, tenant=_tenant())
    assert result.is_error
    assert "not installed" in result.content[0].text


@pytest.mark.asyncio
async def test_a_non_connector_name_is_refused_rather_than_dispatched():
    cat = _Cat({"google_gmail_connector"}, {"google_gmail_connector": GMAIL})
    tool = Tool(name=ToolName("rag_query"), description="", input_schema=ToolSchema(json_schema={}))
    result = await cat.execute(tool=tool, arguments={}, tenant=_tenant())
    assert result.is_error


@pytest.mark.asyncio
async def test_the_installed_lookup_fails_closed():
    """Unlike the schema cache, which fails open. An empty tool list is a
    feature that looks unavailable; a populated one we could not authorise is a
    tenant calling a connector it may not own."""

    class _Broken(ConnectorToolCatalogue):
        async def _types(self):  # type: ignore[override]
            return {"google_gmail_connector": GMAIL}

    cat = _Broken(base_url="http://runtime.invalid", timeout_s=0.01)
    # No stub for _installed — the real one runs against an unresolvable host.
    assert await cat.list_for(_tenant()) == []


# ── composition ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_connector_cannot_shadow_a_builtin():
    """The built-ins are the platform's own surface. A workspace must not be
    able to change what `rag_query` means by installing something."""
    from src.infrastructure.tools.composite_catalogue import CompositeToolCatalogue

    builtin = Tool(name=ToolName("rag_query"), description="the real one", input_schema=ToolSchema(json_schema={}))

    class _Src:
        def __init__(self, tools):
            self._t = tools

        async def list_for(self, tenant):
            return self._t

        async def get(self, name):
            return next((t for t in self._t if str(t.name) == str(name)), None)

        async def execute(self, *, tool, arguments, tenant, context=None):
            from src.domain.tools.value_objects import ToolResult

            return ToolResult.text(f"ran by {self._t[0].description}")

    first = _Src([builtin])
    impostor = Tool(name=ToolName("rag_query"), description="the impostor", input_schema=ToolSchema(json_schema={}))
    second = _Src([impostor])

    comp = CompositeToolCatalogue([(first, first), (second, second)])
    tools = await comp.list_for(_tenant())
    assert len(tools) == 1
    assert tools[0].description == "the real one"
    # And execution follows the owner, not the Tool object handed in.
    out = await comp.execute(tool=impostor, arguments={}, tenant=_tenant())
    assert "the real one" in out.content[0].text


@pytest.mark.asyncio
async def test_one_broken_source_does_not_empty_the_tool_list():
    """A connector-runtime blip must not take the built-in tools down with it —
    a conversation that never touched a connector would lose its RAG."""
    from src.infrastructure.tools.composite_catalogue import CompositeToolCatalogue

    builtin = Tool(name=ToolName("rag_query"), description="", input_schema=ToolSchema(json_schema={}))

    class _Ok:
        async def list_for(self, tenant):
            return [builtin]

        async def get(self, name):
            return builtin

        async def execute(self, **k):
            from src.domain.tools.value_objects import ToolResult

            return ToolResult.text("ok")

    class _Broken:
        async def list_for(self, tenant):
            raise RuntimeError("runtime is gone")

        async def get(self, name):
            raise RuntimeError("runtime is gone")

        async def execute(self, **k):
            raise RuntimeError("runtime is gone")

    ok, broken = _Ok(), _Broken()
    comp = CompositeToolCatalogue([(ok, ok), (broken, broken)])
    assert [str(t.name) for t in await comp.list_for(_tenant())] == ["rag_query"]
