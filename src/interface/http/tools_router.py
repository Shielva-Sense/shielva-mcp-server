"""HTTP routes for direct tool listing + execution.

These supplement the spec-compliant MCP ``tools/list`` + ``tools/call``
JSON-RPC methods. Other internal services hit the REST surface
directly (no JSON-RPC envelope needed). Both routes delegate to
``ToolApplicationService`` — the same use case the JSON-RPC
dispatcher uses, so behaviour is consistent across surfaces.
"""
from __future__ import annotations

import os

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from src.application.tools import ToolApplicationService
from src.domain.tools.errors import (
    ToolExecutionError, ToolNotFoundError, ToolPermissionDeniedError,
)
from src.domain.tools.value_objects import ToolText
from src.protocol.models import ToolExecutionRequest, ToolExecutionResponse

from ._deps import get_domain_tenant, get_tenant_context

logger = structlog.get_logger(__name__)

_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

tools_router = APIRouter(prefix=_PREFIX, tags=["tools"])


def _service(request: Request) -> ToolApplicationService:
    svc = getattr(request.app.state, "tool_app_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="tool_app_service not initialised")
    return svc


@tools_router.get("/tools")
async def list_tools(
    request: Request,
    tenant=Depends(get_domain_tenant),
) -> dict:
    """REST equivalent of MCP ``tools/list``. Returns the same
    tools the JSON-RPC dispatcher would list for this tenant."""
    tools = await _service(request).list_tools(tenant=tenant)
    return {
        "tools": [
            {
                "name":        str(t.name),
                "description": t.description,
                "inputSchema": t.input_schema.json_schema,
            }
            for t in tools
        ],
    }


@tools_router.post("/tools/{tool_name}/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    request:  Request,
    tool_name: str,
    body:     ToolExecutionRequest,
    tenant_legacy=Depends(get_tenant_context),
    tenant=Depends(get_domain_tenant),
) -> ToolExecutionResponse:
    """REST equivalent of MCP ``tools/call``. Returns the legacy
    :class:`ToolExecutionResponse` shape (so existing callers don't
    change) but the call goes through the new use case path."""
    body.tool_name = tool_name
    service = _service(request)
    try:
        result = await service.execute_tool(
            tenant    = tenant,
            name      = tool_name,
            arguments = dict(body.parameters or {}),
            context   = dict(body.context or {}),
        )
    except ToolNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ToolPermissionDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ToolExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # The legacy shape is "result + success + error". Translate
    # from the domain ToolResult, with the text block as the result
    # payload (callers that expected dicts already parsed the text).
    text = "".join(
        b.text for b in result.content if isinstance(b, ToolText)
    )
    return ToolExecutionResponse(
        tool_name = tool_name,
        result    = text if not result.is_error else None,
        success   = not result.is_error,
        error     = text if result.is_error else None,
        duration_ms = 0,
    )
