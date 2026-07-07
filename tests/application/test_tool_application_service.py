"""ToolApplicationService — invariant tests with fake adapters.

We construct fake :class:`ToolCatalogue` + :class:`ToolExecutor`
implementations because the real legacy adapter ties to MCP's
in-memory registry singleton (not the right granularity for unit
testing). The fakes record every call so we can assert audit
behaviour without scraping log output.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.application.tools import ToolApplicationService
from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
)
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import ToolName, ToolResult, ToolSchema


class _FakeCatalogue(ToolCatalogue):
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}
        self.list_calls = 0
        self.get_calls: list[str] = []

    async def list_for(self, tenant: TenantContext) -> list[Tool]:
        self.list_calls += 1
        return [t for t in self._tools.values() if t.is_permitted_for(tenant)]

    async def get(self, name: ToolName) -> Tool | None:
        self.get_calls.append(str(name))
        return self._tools.get(name)


class _FakeExecutor(ToolExecutor):
    def __init__(self, behaviour: dict[str, Any] | None = None) -> None:
        """``behaviour`` maps tool_name → callable or exception.
        Callables get ``(tool, arguments, tenant, context)`` and
        return :class:`ToolResult`. Exceptions are raised."""
        self._behaviour = behaviour or {}
        self.executions: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        tenant: TenantContext,
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        self.executions.append((str(tool.name), dict(arguments)))
        recipe = self._behaviour.get(str(tool.name))
        if recipe is None:
            return ToolResult.text(f"ok:{tool.name}")
        if isinstance(recipe, Exception):
            raise recipe
        return recipe(tool, arguments, tenant, context)


def _tool(name: str, *required: str) -> Tool:
    return Tool(
        name=ToolName(name),
        description=name,
        input_schema=ToolSchema(json_schema={"type": "object"}),
        required_permissions=tuple(required),
    )


def _tenant(*perms: str) -> TenantContext:
    return TenantContext(
        tenant_id="t1",
        user_id="u",
        user_email="u@x",
        permissions=tuple(perms),
    )


@pytest.mark.asyncio
async def test_list_tools_filters_by_permission():
    public = _tool("public")
    admin = _tool("admin", "admin")
    cat = _FakeCatalogue([public, admin])
    svc = ToolApplicationService(catalogue=cat, executor=_FakeExecutor())

    visible = await svc.list_tools(tenant=_tenant())  # no perms
    assert [str(t.name) for t in visible] == ["public"]

    visible2 = await svc.list_tools(tenant=_tenant("admin"))
    assert sorted(str(t.name) for t in visible2) == ["admin", "public"]


@pytest.mark.asyncio
async def test_execute_tool_returns_text_on_happy_path():
    cat = _FakeCatalogue([_tool("get_time")])
    svc = ToolApplicationService(catalogue=cat, executor=_FakeExecutor())

    out = await svc.execute_tool(
        tenant=_tenant(),
        name="get_time",
        arguments={},
    )
    assert out.is_error is False
    assert out.content[0].text == "ok:get_time"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_execute_tool_unknown_raises_not_found():
    svc = ToolApplicationService(
        catalogue=_FakeCatalogue([]),
        executor=_FakeExecutor(),
    )
    with pytest.raises(ToolNotFoundError):
        await svc.execute_tool(tenant=_tenant(), name="missing", arguments={})


@pytest.mark.asyncio
async def test_execute_tool_permission_denied_raises():
    cat = _FakeCatalogue([_tool("admin_op", "admin")])
    svc = ToolApplicationService(catalogue=cat, executor=_FakeExecutor())
    with pytest.raises(ToolPermissionDeniedError):
        await svc.execute_tool(
            tenant=_tenant(),
            name="admin_op",
            arguments={},
        )


@pytest.mark.asyncio
async def test_execute_tool_internal_error_becomes_is_error_result():
    """ToolExecutionError must not bubble out — per MCP spec the
    error surfaces inside the result with is_error=True so the LLM
    can see it."""
    cat = _FakeCatalogue([_tool("flaky")])
    ex = _FakeExecutor(behaviour={"flaky": ToolExecutionError("boom")})
    svc = ToolApplicationService(catalogue=cat, executor=ex)

    out = await svc.execute_tool(tenant=_tenant(), name="flaky", arguments={})
    assert out.is_error is True
    assert "boom" in out.content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_execute_tool_unhandled_exception_caught():
    """Belt-and-braces: a buggy adapter that raises an arbitrary
    exception still produces an is_error result instead of crashing
    the request."""
    cat = _FakeCatalogue([_tool("buggy")])
    ex = _FakeExecutor(behaviour={"buggy": RuntimeError("oops")})
    svc = ToolApplicationService(catalogue=cat, executor=ex)

    out = await svc.execute_tool(tenant=_tenant(), name="buggy", arguments={})
    assert out.is_error is True
    assert "unhandled: RuntimeError" in out.content[0].text  # type: ignore[union-attr]
