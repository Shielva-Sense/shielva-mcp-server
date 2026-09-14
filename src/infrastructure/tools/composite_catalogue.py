"""Several tool sources behind the one port the application talks to.

The built-in tools (rag_query and friends) and a tenant's connector tools are
different kinds of thing with different lifetimes — one is registered at
startup and identical for everybody, the other is looked up per request and
differs per workspace. Neither should have to know the other exists, and the
application service should not have to ask two objects a question the port
already expresses.

🚨 Order is precedence, and FIRST WINS. A connector cannot shadow a built-in by
being named after it: the built-ins are the platform's own surface and a
workspace must not be able to change what `rag_query` means by installing
something. Delegation for execution follows whoever owned the name in the
lookup, so a tool is always run by the source that offered it.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import ToolName, ToolResult

logger = structlog.get_logger(__name__)


class CompositeToolCatalogue(ToolCatalogue, ToolExecutor):
    """Merges N (catalogue, executor) pairs into one of each."""

    def __init__(self, sources: list[tuple[ToolCatalogue, ToolExecutor]]) -> None:
        self._sources = sources

    async def list_for(self, tenant: TenantContext) -> list[Tool]:
        seen: set[str] = set()
        out: list[Tool] = []
        for catalogue, _ in self._sources:
            try:
                tools = await catalogue.list_for(tenant)
            except Exception as exc:
                # 🚨 One source failing must not empty the list. A connector
                # runtime blip would otherwise take the built-in tools down
                # with it, so a conversation that never touched a connector
                # loses its RAG mid-answer.
                logger.warning(
                    "mcp.tool_source_failed",
                    source=type(catalogue).__name__,
                    tenant_id=tenant.tenant_id,
                    error=str(exc)[:200],
                )
                continue
            for tool in tools:
                key = str(tool.name)
                if key in seen:
                    logger.info("mcp.tool_name_shadowed", tool=key, source=type(catalogue).__name__)
                    continue
                seen.add(key)
                out.append(tool)
        return out

    async def get(self, name: ToolName) -> Tool | None:
        for catalogue, _ in self._sources:
            try:
                found = await catalogue.get(name)
            except Exception as exc:
                logger.warning(
                    "mcp.tool_lookup_failed",
                    source=type(catalogue).__name__,
                    error=str(exc)[:200],
                )
                continue
            if found is not None:
                return found
        return None

    async def execute(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        tenant: TenantContext,
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Run it with the executor whose catalogue owns the name.

        🚨 Resolved by asking each catalogue again rather than by trusting the
        `Tool` handed in. The caller could have built one — the name is a
        string on the wire — and picking an executor from an unverified object
        would let a caller choose which subsystem runs their arguments.
        """
        for catalogue, executor in self._sources:
            try:
                owned = await catalogue.get(tool.name)
            except Exception:
                continue
            if owned is not None:
                return await executor.execute(
                    tool=owned,
                    arguments=arguments,
                    tenant=tenant,
                    context=context,
                )
        return ToolResult.text(f"No source owns the tool {tool.name}.", is_error=True)
