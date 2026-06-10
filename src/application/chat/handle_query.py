"""HandleQuery — the RAG-pipeline use case for ``/mcp/v1/query``.

What this use case is responsible for
-------------------------------------
* Translating the inbound :class:`MCPQueryRequest` into a domain
  operation.
* Emitting structured audit at the use-case boundary (so a single
  log query can recover *all* query traffic regardless of which
  transport hit it — REST today, future MCP JSON-RPC method
  tomorrow).
* Owning the policy → retrieval → LLM-with-tools → response shape
  contract.

Where the heavy lifting still lives (deliberately, slice 4b)
------------------------------------------------------------
``protocol.message_handler.MessageHandler.handle_query`` still
orchestrates context_assembler + tool prep + LLM loop. We wrap it
here so the application boundary is clean from the caller's POV
(REST endpoint → application service → infrastructure orchestration);
a future slice will decompose the legacy handler into:

    * ``application/chat.assemble_context``  — pure context building
    * ``application/tools.execute_tool``     — already exists
    * ``application/llm.complete``           — already exists

…and HandleQuery will stitch them together explicitly. Until that
slice lands, this use case is a thin shim with audit + observable
boundaries. The shim shape is intentional: it gives every caller
a stable contract while the internals migrate behind the port.

Why a shim is honest (not technical debt to hide)
-------------------------------------------------
The point of hexagonal architecture isn't "every layer fully
decomposed on day one" — it's "each layer's contract is stable, and
internals can refactor without breaking callers". HandleQuery's
contract is the application-level RAG query operation; whether it
delegates to the legacy MessageHandler or its own decomposed
sub-services is an implementation detail invisible to interface/.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


# DTOs for the use case — same shape as the spec's MCPQueryRequest/
# Response but expressed in application-layer terms. The interface
# adapter translates wire types to/from these.

@dataclass(frozen=True, slots=True)
class HandleQueryInput:
    query:         str
    bot_id:        str
    session_id:    Optional[str]            = None
    stream:        bool                     = False
    context:       Dict[str, Any]           = None  # type: ignore[assignment]
    tool_options:  Dict[str, bool]          = None  # type: ignore[assignment]
    custom_prompt: Optional[str]            = None
    model:         Optional[str]            = None  # per-bot LLM model override


@dataclass(frozen=True, slots=True)
class HandleQueryOutput:
    answer:       str
    sources:      List[Dict[str, Any]]
    tool_calls:   List[Dict[str, Any]]
    tokens_used:  int
    latency_ms:   int
    model:        str
    session_id:   str
    query_id:     str


class HandleQueryUseCase:
    """Slice 4b shim — wraps the legacy MessageHandler.

    The constructor takes the legacy handler by reference (not
    construction). The composition root supplies it because the
    handler holds composed-over-startup wiring (context_assembler,
    tool_registry, …) that we don't re-wire in the application
    layer yet.
    """

    def __init__(self, *, legacy_message_handler: Any) -> None:
        self._handler = legacy_message_handler

    async def execute(
        self,
        *,
        input_:  HandleQueryInput,
        tenant:  TenantContext,
    ) -> HandleQueryOutput:
        started = time.monotonic()
        logger.info(
            "mcp.handle_query_start",
            tenant_id=tenant.tenant_id,
            bot_id=input_.bot_id,
            query_len=len(input_.query or ""),
            has_custom_prompt=bool(input_.custom_prompt),
        )

        # Translate to the legacy DTOs the handler still consumes.
        # Slice 4c will lift this so we use domain types end-to-end.
        from src.protocol.models import (
            MCPQueryRequest as LegacyRequest,
            TenantContext   as LegacyTenant,
        )
        legacy_req = LegacyRequest(
            query         = input_.query,
            bot_id        = input_.bot_id,
            session_id    = input_.session_id,
            stream        = input_.stream,
            context       = dict(input_.context or {}),
            tool_options  = dict(input_.tool_options or {}),
            custom_prompt = input_.custom_prompt,
            model         = input_.model,
        )
        legacy_tenant = LegacyTenant(
            tenant_id   = tenant.tenant_id,
            user_id     = tenant.user_id,
            user_email  = tenant.user_email,
            role        = tenant.role,
            permissions = list(tenant.permissions),
        )

        try:
            legacy_resp = await self._handler.handle_query(
                request=legacy_req, tenant_context=legacy_tenant,
            )
        except PermissionError:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "mcp.handle_query_denied",
                tenant_id=tenant.tenant_id, bot_id=input_.bot_id,
                duration_ms=duration_ms,
            )
            raise
        except Exception as e:  # noqa: BLE001
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.exception(
                "mcp.handle_query_failed",
                tenant_id=tenant.tenant_id, bot_id=input_.bot_id,
                duration_ms=duration_ms, error=str(e)[:200],
            )
            raise

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "mcp.handle_query_ok",
            tenant_id=tenant.tenant_id, bot_id=input_.bot_id,
            duration_ms=duration_ms,
            model=legacy_resp.model,
            tokens_used=legacy_resp.tokens_used,
            source_count=len(legacy_resp.sources),
            tool_call_count=len(legacy_resp.tool_calls),
        )

        # Translate back. We surface the legacy Sources / ToolCalls
        # as plain dicts (their model_dump shape) so the interface
        # adapter doesn't depend on protocol/models.
        return HandleQueryOutput(
            answer       = legacy_resp.answer,
            sources      = [s.model_dump() if hasattr(s, "model_dump") else dict(s)
                            for s in (legacy_resp.sources or [])],
            tool_calls   = [t.model_dump() if hasattr(t, "model_dump") else dict(t)
                            for t in (legacy_resp.tool_calls or [])],
            tokens_used  = int(legacy_resp.tokens_used or 0),
            latency_ms   = int(legacy_resp.latency_ms or 0),
            model        = str(legacy_resp.model or ""),
            session_id   = str(legacy_resp.session_id or ""),
            query_id     = str(legacy_resp.query_id or ""),
        )
