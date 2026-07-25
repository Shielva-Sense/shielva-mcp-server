"""HandleQuery — the RAG-pipeline use case for ``/mcp/v1/query``.

What this use case is responsible for
-------------------------------------
* Owning the ``policy -> assemble context -> tools -> LLM-with-tools ->
  response`` orchestration for a bot query.
* Emitting structured audit at the use-case boundary (so a single log
  query recovers *all* query traffic regardless of which transport hit
  it — REST today, an MCP JSON-RPC method tomorrow).

Where the heavy lifting lives
-----------------------------
This use case orchestrates three infrastructure components injected by
the composition root — the context assembler (bot config + system
prompt + RAG retrieval + prompt-injection fencing), the tool registry
(per-bot tool set), and the LLM router (provider call + tool loop, with
per-tenant BYOK resolution inside). Those components are still the
established singletons; the orchestration that used to live in
``protocol.MessageHandler.handle_query`` now lives *here*, in the
application layer, so the DDD boundary is the real request path rather
than a shim delegating to the legacy handler.

The components consume the protocol-layer ``TenantContext`` /
``SessionContext`` types; we translate the domain tenant to the legacy
shape once, at this boundary. Lifting the components themselves to
domain types (so no translation is needed) is a later slice and does
not change this use case's contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


# DTOs for the use case — same shape as the spec's MCPQueryRequest/
# Response but expressed in application-layer terms. The interface
# adapter translates wire types to/from these.


@dataclass(frozen=True, slots=True)
class HandleQueryInput:
    query: str
    bot_id: str
    session_id: str | None = None
    stream: bool = False
    context: dict[str, Any] = None  # type: ignore[assignment]
    tool_options: dict[str, bool] = None  # type: ignore[assignment]
    custom_prompt: str | None = None
    model: str | None = None  # per-bot LLM model override


@dataclass(frozen=True, slots=True)
class HandleQueryOutput:
    answer: str
    sources: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tokens_used: int
    latency_ms: int
    model: str
    session_id: str
    query_id: str


class HandleQueryUseCase:
    """Orchestrates the bot-query pipeline over the injected components.

    ``context_assembler`` — assembles bot config + system prompt + RAG
    chunks (with the prompt-injection fencing) into an LLM message list.
    ``tool_registry`` — resolves the per-bot enabled tool set.
    ``llm_router`` — runs the provider call + tool loop, resolving the
    tenant's BYOK provider/model/key internally.

    The composition root supplies the three components (constructed
    during the lifespan). This use case owns *ordering + audit*; the
    components own the work.
    """

    def __init__(self, *, context_assembler: Any, tool_registry: Any, llm_router: Any) -> None:
        self._assembler = context_assembler
        self._tools = tool_registry
        self._llm = llm_router

    async def execute(
        self,
        *,
        input_: HandleQueryInput,
        tenant: TenantContext,
    ) -> HandleQueryOutput:
        started = time.monotonic()
        logger.info(
            "mcp.handle_query_start",
            tenant_id=tenant.tenant_id,
            bot_id=input_.bot_id,
            query_len=len(input_.query or ""),
            has_custom_prompt=bool(input_.custom_prompt),
        )

        # The components still consume the protocol-layer tenant/session
        # types — translate the domain tenant once, here at the boundary.
        from src.protocol.models import (
            SessionContext as LegacySession,
        )
        from src.protocol.models import (
            TenantContext as LegacyTenant,
        )

        legacy_tenant = LegacyTenant(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            user_email=tenant.user_email,
            role=tenant.role,
            permissions=list(tenant.permissions),
        )

        try:
            # 1. Policy — trust the gateway-verified principal. RBAC lives
            #    in the IdP/gateway; MCP requires an authenticated,
            #    tenant-scoped caller and scopes all data by tenant_id.
            if not tenant.tenant_id:
                raise PermissionError("Query denied: no authenticated principal")

            # 2. Session — ephemeral per request (not persisted; carries
            #    conversation shape into the assembler).
            session = LegacySession(tenant_context=legacy_tenant, bot_id=input_.bot_id)

            # 3. Assemble context: bot config + system prompt + RAG
            #    retrieval + prompt-injection fencing -> LLM messages.
            context = await self._assembler.assemble(
                query=input_.query,
                session=session,
                tenant_context=legacy_tenant,
                bot_id=input_.bot_id,
                custom_prompt=input_.custom_prompt,
            )

            # 4. Per-bot enabled tool set.
            tools = await self._tools.get_tools_for_bot(
                bot_id=input_.bot_id,
                tenant_context=legacy_tenant,
                enabled_tools=dict(input_.tool_options or {}),
            )

            # 5. LLM + tool loop. Per-tenant BYOK provider/model/key is
            #    resolved inside the router; ``model`` is a per-bot override.
            result = await self._llm.execute(
                messages=context.messages,
                tools=tools,
                tenant_context=legacy_tenant,
                stream=input_.stream,
                model=input_.model,
            )
        except PermissionError:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "mcp.handle_query_denied",
                tenant_id=tenant.tenant_id,
                bot_id=input_.bot_id,
                duration_ms=duration_ms,
            )
            raise
        except Exception as e:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.exception(
                "mcp.handle_query_failed",
                tenant_id=tenant.tenant_id,
                bot_id=input_.bot_id,
                duration_ms=duration_ms,
                error=str(e)[:200],
            )
            raise

        # 6. Map retrieved chunks -> source dicts (trimmed for wire size;
        #    source mapping must never break the query response).
        sources: list[dict[str, Any]] = []
        for chunk in context.retrieved_chunks:
            try:
                meta = getattr(chunk, "metadata", {}) or {}
                sources.append(
                    {
                        "kb_id": getattr(chunk, "kb_id", ""),
                        "kb_name": getattr(chunk, "kb_name", meta.get("kb_name", "")),
                        "document_id": getattr(chunk, "document_id", meta.get("document_id", "")),
                        "document_title": getattr(chunk, "document_title", meta.get("title", "")),
                        "chunk_id": getattr(chunk, "chunk_id", ""),
                        "content": (getattr(chunk, "content", "") or "")[:200],
                        "score": round(getattr(chunk, "score", 0.0), 4),
                        "metadata": {},
                    }
                )
            except Exception:
                pass  # never let source mapping break the query response

        tool_calls = [
            t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in (result.tool_calls or [])
        ]
        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "mcp.handle_query_ok",
            tenant_id=tenant.tenant_id,
            bot_id=input_.bot_id,
            duration_ms=duration_ms,
            model=result.model,
            tokens_used=result.tokens_used,
            source_count=len(sources),
            tool_call_count=len(tool_calls),
        )

        return HandleQueryOutput(
            answer=result.answer or "",
            sources=sources,
            tool_calls=tool_calls,
            tokens_used=int(result.tokens_used or 0),
            latency_ms=duration_ms,
            model=str(result.model or ""),
            session_id=str(session.session_id or ""),
            query_id="",
        )
