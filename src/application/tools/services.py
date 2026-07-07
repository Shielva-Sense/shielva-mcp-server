"""Tools application service — list + call use cases."""

from __future__ import annotations

import time
from typing import Any

import structlog

from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
)
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import ToolName, ToolResult

logger = structlog.get_logger(__name__)


class ToolApplicationService:
    """Use cases for the MCP tools surface.

    Wraps the two domain ports and adds:
        * Audit logging on every call (per SOC2 CC7.2).
        * Duration tracking for the structured log entry.
        * Translation of domain exceptions into ``ToolResult.failure``
          for the executor path — the interface layer reads ``is_error``
          to decide whether to surface as JSON-RPC error or as a
          spec-compliant tool-result content block.
    """

    def __init__(
        self,
        *,
        catalogue: ToolCatalogue,
        executor: ToolExecutor,
    ) -> None:
        self._catalogue = catalogue
        self._executor = executor

    # ── list ──────────────────────────────────────────────────────

    async def list_tools(self, *, tenant: TenantContext) -> list[Tool]:
        """Tools visible to this tenant. Filtering by permission is
        the catalogue's responsibility, but we double-check here so
        a buggy catalogue can't widen visibility silently."""
        tools = await self._catalogue.list_for(tenant)
        out: list[Tool] = []
        for t in tools:
            if t.is_permitted_for(tenant):
                out.append(t)
            else:
                # Belt-and-braces: log if a tool leaked through the
                # adapter that shouldn't have. Helps catch bad
                # adapter implementations in CI.
                logger.warning(
                    "mcp.tool_visibility_mismatch",
                    tool=str(t.name),
                    tenant_id=tenant.tenant_id,
                )
        return out

    # ── call ──────────────────────────────────────────────────────

    async def execute_tool(
        self,
        *,
        tenant: TenantContext,
        name: str,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Run a tool. Returns :class:`ToolResult` (never raises for
        tool-internal errors — those are surfaced as
        ``is_error=True`` per MCP spec).

        Raises:
            ToolNotFoundError           — name not in catalogue.
            ToolPermissionDeniedError   — tenant lacks required perms.
        """
        tool_name = ToolName(name)
        tool = await self._catalogue.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(f"Tool not found: {name}")
        if not tool.is_permitted_for(tenant):
            raise ToolPermissionDeniedError(
                f"Tenant {tenant.tenant_id} lacks permission for tool {name}",
            )

        # Audit envelope BEFORE the call so it's emitted even when
        # the executor hangs (SOP retains the start event; a missing
        # finish_event in monitoring is the alert).
        started = time.monotonic()
        logger.info(
            "mcp.tool_call_start",
            tool=name,
            tenant_id=tenant.tenant_id,
            arg_keys=sorted(arguments.keys()),
            # NB: argument *values* are not logged here — they may
            # carry tenant data. Tools that want their args in the
            # audit log emit it from their own handler with the
            # SOC2 redactor applied.
        )

        try:
            result = await self._executor.execute(
                tool=tool,
                arguments=arguments,
                tenant=tenant,
                context=context,
            )
        except (ToolNotFoundError, ToolPermissionDeniedError):
            # Re-raise as policy decisions; do NOT log here as a
            # tool failure (those are different concerns).
            raise
        except ToolExecutionError as e:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "mcp.tool_call_failed",
                tool=name,
                tenant_id=tenant.tenant_id,
                duration_ms=duration_ms,
                error=str(e)[:200],
            )
            return ToolResult.failure(str(e) or "tool failed")
        except Exception as e:
            # Last line of defence — adapters SHOULD catch their own
            # exceptions, but a buggy one shouldn't bring down the
            # whole MCP request.
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.exception(
                "mcp.tool_call_unhandled_exception",
                tool=name,
                tenant_id=tenant.tenant_id,
                duration_ms=duration_ms,
            )
            return ToolResult.failure(f"unhandled: {type(e).__name__}")

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "mcp.tool_call_ok",
            tool=name,
            tenant_id=tenant.tenant_id,
            duration_ms=duration_ms,
            is_error=result.is_error,
            content_blocks=len(result.content),
        )
        return result
