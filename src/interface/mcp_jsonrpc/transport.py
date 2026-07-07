"""Streamable HTTP transport for the MCP JSON-RPC façade.

Wire-format compliance: spec 2025-03-26 (Streamable HTTP) layered on
spec 2024-11-05 (message schemas). Includes the P0 spec-violation
fixes from the GA audit:

  * Origin header validation against an allowlist (DNS rebinding
    defence, spec MUST).
  * HTTP 404 on a stale ``Mcp-Session-Id`` (spec MUST — after a
    server terminates a session it MUST respond with 404 to
    requests carrying that id).
  * HTTP 400 when ``Mcp-Session-Id`` is missing on a non-initialize
    request (spec SHOULD).
  * ``Accept`` header validation on POST (spec requires the client
    to send both ``application/json`` and ``text/event-stream``).

The transport is built via :func:`build_router` so the composition
root can inject the :class:`MCPDispatcher` (which itself owns the
chat application service). No module-level singletons.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from src.domain.chat.errors import SessionNotFoundError, SessionStateError
from src.domain.shared.tenant import TenantContext

from .dispatcher import MCPDispatcher
from .errors import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    PARSE_ERROR,
    JsonRpcException,
)
from .types import JsonRpcError, JsonRpcRequest, JsonRpcResponse

logger = structlog.get_logger(__name__)


# ── Origin allowlist ─────────────────────────────────────────────────


def _allowed_origins() -> set[str]:
    """Comma-separated env. ``*`` disables the check (only valid for
    closed dev networks)."""
    raw = (os.getenv("MCP_ALLOWED_ORIGINS") or "").strip()
    if not raw:
        # Sane default: same-origin only (no Origin header, or
        # Origin matching the configured CORS list). We treat missing
        # Origin as same-origin (curl, server-to-server, mcp-inspector
        # CLI).
        return set()
    if raw == "*":
        return {"*"}
    return {o.strip() for o in raw.split(",") if o.strip()}


def _origin_ok(origin: str | None, allowed: set[str]) -> bool:
    if "*" in allowed:
        return True
    if origin is None or origin == "" or origin.lower() == "null":
        # Same-origin / server-to-server / cli — no Origin header is
        # sent. Allowed by default; the auth header still gates access.
        return True
    return origin in allowed


# ── Tenant extraction ────────────────────────────────────────────────


def _tenant_from_request(request: Request) -> TenantContext:
    """Auth is X-Tenant-ID for now. OAuth 2.1 bearer is on the
    GA-readiness backlog."""
    tenant_id = (request.headers.get("X-Tenant-ID") or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID header")
    return TenantContext(
        tenant_id=tenant_id,
        user_id=request.headers.get("X-User-ID") or "mcp-client",
        user_email=request.headers.get("X-User-Email") or "client@shielva.ai",
        role=request.headers.get("X-User-Role") or "Customer_Basic",
        permissions=(),
    )


# ── one request → one response ──────────────────────────────────────


async def _run_one(
    payload: dict[str, Any],
    *,
    tenant: TenantContext,
    session_id_in: str | None,
    app_state: Any,
    dispatcher: MCPDispatcher,
) -> tuple[dict[str, Any] | None, str | None]:
    """Dispatch one envelope. Returns
    ``(response_dict_or_None, session_id_for_response)``.
    None on the dict means it was a notification → caller drops it."""
    try:
        req = JsonRpcRequest.model_validate(payload)
    except Exception as e:
        return (
            JsonRpcResponse(
                id=payload.get("id") if isinstance(payload, dict) else None,
                error=JsonRpcError(code=INVALID_REQUEST, message=f"Invalid Request: {e}"),
            ).model_dump(exclude_none=True),
            session_id_in,
        )

    is_notification = req.id is None

    if is_notification:
        try:
            await dispatcher.dispatch_notification(
                method=req.method,
                params=req.params,
                tenant=tenant,
                session_id=session_id_in,
                app_state=app_state,
            )
        except Exception:
            # Notifications never return responses, even on error.
            logger.warning("mcp.notification_error", method=req.method)
        return (None, session_id_in)

    try:
        result, session_id_out = await dispatcher.dispatch_request(
            method=req.method,
            params=req.params,
            tenant=tenant,
            session_id=session_id_in,
            app_state=app_state,
        )
    except SessionNotFoundError as e:
        # Spec MUST: HTTP 404. We surface a sentinel that the outer
        # caller translates to HTTP 404; the JSON-RPC envelope still
        # carries an error so well-behaved clients can read it.
        return (
            {
                "__http_status__": 404,
                **JsonRpcResponse(
                    id=req.id,
                    error=JsonRpcError(code=INVALID_REQUEST, message=str(e)),
                ).model_dump(exclude_none=True),
            },
            None,  # drop the session id from the response header
        )
    except SessionStateError as e:
        return (
            JsonRpcResponse(
                id=req.id,
                error=JsonRpcError(code=INVALID_REQUEST, message=str(e)),
            ).model_dump(exclude_none=True),
            session_id_in,
        )
    except JsonRpcException as je:
        return (
            JsonRpcResponse(
                id=req.id,
                error=JsonRpcError(code=je.code, message=je.message, data=je.data),
            ).model_dump(exclude_none=True),
            session_id_in,
        )
    except Exception as exc:
        logger.exception("mcp.dispatch_internal_error", method=req.method)
        return (
            JsonRpcResponse(
                id=req.id,
                error=JsonRpcError(
                    code=INTERNAL_ERROR,
                    message=f"Internal error: {type(exc).__name__}",
                ),
            ).model_dump(exclude_none=True),
            session_id_in,
        )

    return (
        JsonRpcResponse(id=req.id, result=result).model_dump(exclude_none=True),
        session_id_out,
    )


# ── Router factory ──────────────────────────────────────────────────


def build_router(dispatcher: MCPDispatcher) -> APIRouter:
    """Build the FastAPI router that owns ``/mcp``.

    The composition root constructs the dispatcher (with its wired
    application service + repository) and hands it in. The router
    itself is stateless; per-request state is on the request object.
    """
    router = APIRouter(tags=["mcp-protocol"])
    allowed = _allowed_origins()

    def _check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not _origin_ok(origin, allowed):
            raise HTTPException(
                status_code=403,
                detail=f"Origin {origin!r} not in MCP_ALLOWED_ORIGINS",
            )

    # ── POST /mcp ─────────────────────────────────────────────────

    @router.post("/mcp")
    async def mcp_post(request: Request) -> Response:
        _check_origin(request)
        tenant = _tenant_from_request(request)

        # Spec MUST: client Accept header lists both content types.
        # We're lenient — if Accept is missing we assume the spec
        # default (any). If it's present and doesn't include
        # application/json we reject — the response would be
        # rejected by the client anyway.
        accept = (request.headers.get("accept") or "").lower()
        if accept and accept != "*/*":
            wants_json = "application/json" in accept
            wants_sse = "text/event-stream" in accept
            if not (wants_json or wants_sse):
                raise HTTPException(
                    status_code=406,
                    detail="Accept must include application/json or text/event-stream",
                )

        raw = await request.body()
        if not raw:
            return JSONResponse(
                status_code=400,
                content=JsonRpcResponse(
                    error=JsonRpcError(code=PARSE_ERROR, message="Empty body"),
                ).model_dump(exclude_none=True),
            )
        try:
            body = json.loads(raw)
        except Exception:
            return JSONResponse(
                status_code=400,
                content=JsonRpcResponse(
                    error=JsonRpcError(code=PARSE_ERROR, message="Malformed JSON"),
                ).model_dump(exclude_none=True),
            )

        session_id_in = request.headers.get("Mcp-Session-Id") or None

        # ── batch ────────────────────────────────────────────
        if isinstance(body, list):
            if not body:
                return JSONResponse(
                    status_code=400,
                    content=JsonRpcResponse(
                        error=JsonRpcError(code=INVALID_REQUEST, message="Empty batch"),
                    ).model_dump(exclude_none=True),
                )
            responses: list[dict[str, Any]] = []
            http_status = 200
            session_out = session_id_in
            for item in body:
                if not isinstance(item, dict):
                    responses.append(
                        JsonRpcResponse(
                            error=JsonRpcError(code=INVALID_REQUEST, message="Batch item not an object"),
                        ).model_dump(exclude_none=True)
                    )
                    continue
                resp, session_out = await _run_one(
                    item,
                    tenant=tenant,
                    session_id_in=session_out,
                    app_state=request.app.state,
                    dispatcher=dispatcher,
                )
                if resp is None:
                    continue
                # Stale session in a batch upgrades the whole batch
                # to 404 — there's no way to mix 200 + 404 in one
                # HTTP response. Strip the marker before sending.
                if resp.get("__http_status__") == 404:
                    http_status = 404
                    resp.pop("__http_status__", None)
                responses.append(resp)
            if not responses:
                return Response(status_code=202)
            headers: dict[str, str] = {}
            if session_out and http_status == 200:
                headers["Mcp-Session-Id"] = session_out
            return JSONResponse(content=responses, headers=headers, status_code=http_status)

        # ── single ───────────────────────────────────────────
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content=JsonRpcResponse(
                    error=JsonRpcError(code=INVALID_REQUEST, message="Body must be object or array"),
                ).model_dump(exclude_none=True),
            )

        # Spec SHOULD: 400 when Mcp-Session-Id is missing on a non-
        # initialize, non-ping request. Detect early — saves a
        # round-trip through the dispatcher.
        method = body.get("method") or ""
        if not session_id_in and method not in (
            "initialize",
            "ping",
            "notifications/initialized",
        ):
            return JSONResponse(
                status_code=400,
                content=JsonRpcResponse(
                    id=body.get("id"),
                    error=JsonRpcError(
                        code=INVALID_REQUEST,
                        message="Mcp-Session-Id header required",
                    ),
                ).model_dump(exclude_none=True),
            )

        resp, session_out = await _run_one(
            body,
            tenant=tenant,
            session_id_in=session_id_in,
            app_state=request.app.state,
            dispatcher=dispatcher,
        )
        if resp is None:
            # Notification — spec says 202.
            return Response(status_code=202)

        # Stale session → HTTP 404 per spec.
        if resp.get("__http_status__") == 404:
            resp.pop("__http_status__", None)
            return JSONResponse(content=resp, status_code=404)

        headers: dict[str, str] = {}
        if session_out:
            headers["Mcp-Session-Id"] = session_out
        return JSONResponse(content=resp, headers=headers)

    # ── DELETE /mcp ───────────────────────────────────────────────

    @router.delete("/mcp")
    async def mcp_delete(request: Request) -> Response:
        _check_origin(request)
        tenant = _tenant_from_request(request)
        session_id = request.headers.get("Mcp-Session-Id") or ""
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="Mcp-Session-Id header required on DELETE /mcp",
            )
        # Idempotent close — no error if already gone.
        from src.domain.chat.value_objects import SessionId

        chat = dispatcher._chat
        await chat.close(session_id=SessionId(session_id), tenant=tenant)
        return Response(status_code=204)

    # ── GET /mcp ──────────────────────────────────────────────────
    #
    # Spec 2025-03-26: server MAY support server-initiated SSE for
    # notifications/requests. Slice 1 returns 405 because the
    # notifications/progress emitter isn't wired yet. When slice 2
    # adds progress notifications during tool calls, this endpoint
    # will stream them.

    @router.get("/mcp")
    async def mcp_get(request: Request) -> Response:
        _check_origin(request)
        _ = _tenant_from_request(request)
        return Response(
            status_code=405,
            content=b"GET /mcp not yet implemented; use POST",
            media_type="text/plain",
        )

    return router
