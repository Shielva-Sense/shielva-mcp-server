"""Admin / stats endpoint."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request

from src.application.tools import ToolApplicationService

from ._deps import get_domain_tenant

_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

admin_router = APIRouter(prefix=_PREFIX, tags=["admin"])


@admin_router.get("/admin/stats")
async def get_stats(
    request: Request,
    tenant=Depends(get_domain_tenant),
) -> dict:
    tool_svc: ToolApplicationService | None = getattr(
        request.app.state, "tool_app_service", None,
    )
    tools_available = 0
    if tool_svc is not None:
        tools = await tool_svc.list_tools(tenant=tenant)
        tools_available = len(tools)

    return {
        "tenant_id":          tenant.tenant_id,
        "queries_processed":  0,   # TODO: instrument via sop_sdk counters
        "tools_available":    tools_available,
        "active_sessions":    0,   # TODO: query InMemoryChatSessionRepository._size
    }
