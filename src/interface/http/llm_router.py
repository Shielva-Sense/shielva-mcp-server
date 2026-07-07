"""Generic single-turn LLM API.

``POST /mcp/v1/llm/complete`` is a stateless one-shot LLM call —
no session, no bot, no KB, no MCP-side tool execution. The caller
supplies its own messages + tools schema; we return either plain
text or raw ``tool_calls``. The caller drives the tool loop.

Consumer model
--------------
This endpoint is for internal services that *own their tool
runtime* (TMS qa_runner, future presence-core inline QA, future
cms-core assistant). They cannot register handlers in MCP's
tool_registry because their tools touch their own databases. So
MCP only does what MCP is uniquely good at: provider-agnostic
LLM routing with fallback.

Where it fits in the endpoint inventory
---------------------------------------
* ``/mcp/v1/codegen/complete``  — integration-builder, text-only.
* ``/mcp/v1/codegen/fix-agent`` — integration-builder, MCP runs
                                  the tool loop with MCP-registered
                                  codegen tools.
* ``/mcp/v1/query``             — bot-shaped, RAG pipeline.
* ``/mcp/v1/llm/complete``      — this. caller-driven tools, no
                                  RAG, no bot, single turn.
* ``POST /mcp``                 — spec-compliant JSON-RPC with
                                  MCP-registered tools.

Each endpoint owns a distinct (consumer × tool-execution-model)
slot. No OCP violation — adding a new slot is extension, not
modification of an existing one.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.application.llm import LLMApplicationService
from src.domain.llm.value_objects import (
    LLMMessage,
    LLMRequest,
    LLMToolCall,
    MessageRole,
    ModelId,
)
from src.infrastructure.entitlements import require_llm_entitlement

from ._deps import get_domain_tenant

logger = structlog.get_logger(__name__)

_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

llm_router = APIRouter(prefix=_PREFIX, tags=["llm"])


# ── Wire shape ────────────────────────────────────────────────────


class _ToolCallFunctionDTO(BaseModel):
    name: str
    arguments: str = "{}"  # OpenAI sends arguments as JSON string


class _ToolCallDTO(BaseModel):
    id: str
    type: str = "function"
    function: _ToolCallFunctionDTO


class LLMCompleteMessage(BaseModel):
    """One message in the conversation. Mirrors the OpenAI chat-
    completions shape; LiteLLM accepts it verbatim across providers."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[_ToolCallDTO] | None = None
    tool_call_id: str | None = None  # required when role=tool
    name: str | None = None  # optional on role=tool


class LLMCompleteRequest(BaseModel):
    """Request body for one LLM completion."""

    messages: list[LLMCompleteMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history. Last turn is what the model responds to.",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI tool-schema array. Pass on every turn during a "
        "tool-calling loop so the model sees the available tools.",
    )
    tool_choice: Any | None = Field(
        default=None,
        description='"auto" | "none" | {"type":"function","function":{"name":"..."}}.'
        ' Defaults to "auto" when tools is non-empty.',
    )
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.1


class LLMCompleteResponse(BaseModel):
    """Response from one LLM completion turn."""

    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    model: str = ""
    tokens_used: int = 0
    finish_reason: str = ""


# ── Service lookup ────────────────────────────────────────────────


def _service(request: Request) -> LLMApplicationService:
    svc = getattr(request.app.state, "llm_app_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="llm_app_service not wired; check composition root",
        )
    return svc


# ── Route ─────────────────────────────────────────────────────────


@llm_router.post(
    "/llm/complete",
    response_model=LLMCompleteResponse,
    dependencies=[Depends(require_llm_entitlement)],
)
async def llm_complete(
    request: Request,
    body: LLMCompleteRequest,
    tenant=Depends(get_domain_tenant),
) -> LLMCompleteResponse:
    """Single-turn LLM call. Caller drives any tool loop."""
    service = _service(request)

    # Translate wire → domain.
    domain_messages = tuple(
        LLMMessage(
            role=MessageRole(m.role),
            content=m.content or "",
            tool_calls=tuple(
                LLMToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
                for tc in (m.tool_calls or [])
            ),
            tool_call_id=m.tool_call_id or "",
            name=m.name or "",
        )
        for m in body.messages
    )

    domain_request = LLMRequest(
        messages=domain_messages,
        model=ModelId(body.model) if body.model else None,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        tools=tuple(body.tools) if body.tools else None,
        tool_choice=body.tool_choice if body.tools else None,
        stream=False,
    )

    try:
        response = await service.complete(domain_request, tenant=tenant)
    except Exception as exc:
        logger.error(
            "llm_complete_failed",
            tenant_id=tenant.tenant_id,
            error=str(exc)[:200],
        )
        raise HTTPException(
            status_code=500,
            detail=f"LLM completion failed: {type(exc).__name__}",
        ) from exc

    # Translate domain → wire. ``tool_calls`` is None when the model
    # returned plain text; the OpenAI-format dict shape matches what
    # the caller would forward into the next turn's ``messages``.
    tool_calls_out: list[dict[str, Any]] | None = None
    if response.tool_calls:
        tool_calls_out = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in response.tool_calls
        ]

    return LLMCompleteResponse(
        text=response.content or "",
        tool_calls=tool_calls_out,
        model=str(response.model),
        tokens_used=response.usage.total_tokens,
        finish_reason=response.finish_reason.value,
    )


@llm_router.post(
    "/llm/complete/stream",
    dependencies=[Depends(require_llm_entitlement)],
)
async def llm_complete_stream(
    request: Request,
    body: LLMCompleteRequest,
    tenant=Depends(get_domain_tenant),
) -> StreamingResponse:
    """Token-by-token streaming variant of ``/llm/complete`` (Server-Sent Events).

    Frames (``text/event-stream``):
      * ``data: {"delta": "...", "model": "..."}``           per token
      * ``data: {"delta": "", "finish_reason": "stop", ...}`` final token
      * ``data: [DONE]``                                       terminal sentinel
      * ``event: error\\ndata: {"error": "...", ...}``         on failure

    Tools are NOT supported on the streaming path (text-only). Use the
    non-streaming ``/llm/complete`` for tool-calling loops.
    """
    service = _service(request)

    # Same wire→domain translation as /llm/complete (kept inline so the
    # non-streaming path is untouched).
    domain_messages = tuple(
        LLMMessage(
            role=MessageRole(m.role),
            content=m.content or "",
            tool_calls=tuple(
                LLMToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
                for tc in (m.tool_calls or [])
            ),
            tool_call_id=m.tool_call_id or "",
            name=m.name or "",
        )
        for m in body.messages
    )

    domain_request = LLMRequest(
        messages=domain_messages,
        model=ModelId(body.model) if body.model else None,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        tools=None,
        tool_choice=None,
        stream=True,
    )

    async def _event_stream():
        try:
            async for chunk in service.stream(domain_request, tenant=tenant):
                payload: dict[str, Any] = {
                    "delta": chunk.delta,
                    "model": str(chunk.model),
                }
                if chunk.finish_reason is not None:
                    payload["finish_reason"] = chunk.finish_reason.value
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error(
                "llm_complete_stream_failed",
                tenant_id=tenant.tenant_id,
                error=str(exc)[:200],
            )
            err = json.dumps({"error": type(exc).__name__, "detail": str(exc)[:200]})
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush
        },
    )
