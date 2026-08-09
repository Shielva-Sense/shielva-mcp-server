"""HTTP routes for ``POST /mcp/v1/query`` and ``/mcp/v1/query/stream``.

These migrated out of ``main.py`` during slice 4b. The handler logic
moved into :class:`application.chat.HandleQueryUseCase` so the route
itself is now ~30 lines of HTTP framing + application call.

Streaming note
--------------
``/query/stream`` emits real token-by-token SSE (``meta`` -> ``token`` * n ->
``done``) for bots with NO tools enabled. Bots WITH tools keep the batched tool
loop and receive the finished answer as a single token event: the provider's
streaming path is text-only, and streaming through the tool loop previously
returned empty answers. Correctness over latency for those bots.
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
    """Token-by-token streaming variant.

    Emits proper SSE events so a consumer can start speaking before the answer
    is finished: ``meta`` (retrieval confidence, before any token), many
    ``token`` (text deltas), then ``done`` (full text + confidence).

    Previously this awaited the whole answer and emitted it as one event, which
    meant a phone call sat in silence for the full generation time — measured at
    9,956 ms on a live call. Bots WITH tools enabled still take that batched
    path, because the provider's stream is text-only and streaming through the
    tool loop returned empty answers; they get one token event with the complete
    answer, so behaviour is unchanged for them.
    """
    body.stream = True
    use_case = _use_case(request)

    async def generate():
        try:
            async for evt in use_case.execute_stream(
                input_=HandleQueryInput(
                    query=body.query,
                    bot_id=body.bot_id,
                    session_id=body.session_id,
                    stream=True,
                    context=body.context,
                    tool_options=body.tool_options,
                    custom_prompt=body.custom_prompt,
                    model=body.model,
                    text_only=body.text_only,
                ),
                tenant=tenant,
            ):
                yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"
        except Exception as e:
            # Terminate as a well-formed `done` rather than raw text: a consumer
            # mid-utterance needs a terminal event, not a parse error.
            logger.warning("mcp.query_stream_error", error=str(e)[:200])
            yield f"event: done\ndata: {json.dumps({'text': '', 'confidence': 0.0, 'error': str(e)[:200]})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
