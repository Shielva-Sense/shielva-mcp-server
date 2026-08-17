"""HTTP routes for ``POST /mcp/v1/query`` and ``/mcp/v1/query/stream``.

These migrated out of ``main.py`` during slice 4b. The handler logic
moved into :class:`application.chat.HandleQueryUseCase` so the route
itself is now ~30 lines of HTTP framing + application call.

Streaming note
--------------
The legacy ``/query/stream`` endpoint streams the final answer as a
single SSE event after the underlying call completes (no token-by-
token streaming). That behaviour is preserved here — true streaming
arrives when the LLM provider port grows a streaming variant.
"""

from __future__ import annotations

import json
import os

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.application.chat import HandleQueryInput, HandleQueryUseCase
from src.infrastructure.entitlements import require_llm_entitlement
from src.protocol.models import MCPQueryRequest, MCPQueryResponse, Source, ToolCall

from ._deps import get_domain_tenant

logger = structlog.get_logger(__name__)

_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

query_router = APIRouter(prefix=_PREFIX, tags=["query"])


def _use_case(request: Request) -> HandleQueryUseCase:
    uc = getattr(request.app.state, "handle_query_use_case", None)
    if uc is None:
        # Should never happen — lifespan wires this. Fail loud rather
        # than silently route around the new architecture.
        raise HTTPException(
            status_code=503,
            detail="HandleQuery use case not initialised",
        )
    return uc


@query_router.post(
    "/query",
    response_model=MCPQueryResponse,
    dependencies=[Depends(require_llm_entitlement)],
)
async def process_query(
    request: Request,
    body: MCPQueryRequest,
    tenant=Depends(get_domain_tenant),
) -> MCPQueryResponse:
    use_case = _use_case(request)
    try:
        out = await use_case.execute(
            input_=HandleQueryInput(
                query=body.query,
                bot_id=body.bot_id,
                session_id=body.session_id,
                stream=body.stream,
                context=body.context,
                tool_options=body.tool_options,
                custom_prompt=body.custom_prompt,
                model=body.model,
            ),
            tenant=tenant,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("query_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Translate plain-dict outputs back to the legacy Pydantic
    # response model. Slice 4c could expose a fully application-typed
    # response so this translation goes away.
    return MCPQueryResponse(
        query_id=out.query_id,
        answer=out.answer,
        sources=[Source(**s) for s in out.sources],
        tool_calls=[ToolCall(**t) for t in out.tool_calls],
        tokens_used=out.tokens_used,
        latency_ms=out.latency_ms,
        model=out.model,
        session_id=out.session_id,
    )


@query_router.post(
    "/query/stream",
    dependencies=[Depends(require_llm_entitlement)],
)
async def process_query_stream(
    request: Request,
    body: MCPQueryRequest,
    tenant=Depends(get_domain_tenant),
):
    """Streaming variant.

    Two behaviours, chosen by MCP_QUERY_TOKEN_STREAM:

      unset (default) — the answer is emitted as ONE event after the call
        completes. Not streaming; the name is historical. Left as the default
        because it is what every consumer runs on today.

      "1" — real token streaming: `meta` once, then `token` per delta, then
        `done`. Measured on a live phone call, the batched path is 78% of a
        2,261ms turn and the caller hears nothing until the whole answer
        exists, so this is worth ~1.4s per turn.

    Flagged rather than switched because two earlier attempts at streaming in
    this service were reverted for silently dropping tool-calling and RAG. The
    failure mode is a FAST UNGROUNDED ANSWER, which is worse for a caller than
    the slow correct one. Turn it on deliberately, compare a knowledge-base
    question against the batched path, then make it the default.
    """
    body.stream = True
    use_case = _use_case(request)
    token_stream = os.getenv("MCP_QUERY_TOKEN_STREAM", "").strip().lower() in {"1", "true", "yes", "on"}

    def _sse(event: str, payload) -> str:
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    async def generate_tokens():
        """meta -> token* -> done, as Server-Sent Events."""
        try:
            async for kind, payload in use_case.execute_stream(
                input_=HandleQueryInput(
                    query=body.query,
                    bot_id=body.bot_id,
                    session_id=body.session_id,
                    stream=True,
                    context=body.context,
                    tool_options=body.tool_options,
                    custom_prompt=body.custom_prompt,
                    model=body.model,
                ),
                tenant=tenant,
            ):
                yield _sse(kind, payload)
        except Exception as e:
            logger.warning("mcp.query_stream_failed", error=str(e)[:200])
            # A consumer mid-stream cannot retry, so end the frame properly
            # rather than truncating: an unterminated SSE stream hangs it.
            yield _sse("done", {"answer": "", "error": str(e)[:200]})

    async def generate_batched():
        try:
            out = await use_case.execute(
                input_=HandleQueryInput(
                    query=body.query,
                    bot_id=body.bot_id,
                    session_id=body.session_id,
                    stream=True,
                    context=body.context,
                    tool_options=body.tool_options,
                    custom_prompt=body.custom_prompt,
                    model=body.model,
                ),
                tenant=tenant,
            )
            yield out.answer
        except Exception as e:
            yield f"Error: {e}"

    gen = generate_tokens if token_stream else generate_batched
    return StreamingResponse(gen(), media_type="text/event-stream")
