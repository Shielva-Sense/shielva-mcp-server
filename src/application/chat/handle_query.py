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
from collections.abc import AsyncIterator
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


_MEMORY_ROLES = {"user", "assistant", "system"}


def _conversation_messages(context: dict[str, Any] | None) -> list[Any]:
    """Build prior-turn messages from the caller-supplied conversation memory.

    Wire shape (from core-api, the owner of the transcript)::

        context = {"messages": [{"role": "user"|"assistant", "content": str}, ...],
                   "summary": "rolling summary of older turns"}

    A ``summary`` is prepended as a single system message so long conversations
    stay coherent without replaying every turn — the caller has already trimmed
    ``messages`` to the bot's configured depth.

    Defensive by contract, like ``_chunks_to_sources``: a malformed entry must
    never break a query, so each is mapped in isolation and bad ones are skipped.
    ``tool`` roles are dropped — tool plumbing is per-turn noise and replaying it
    without the matching call ids produces orphaned tool messages at the provider.
    """
    if not context:
        return []
    from src.protocol.models import MCPMessage, MessageRole

    out: list[Any] = []
    summary = str(context.get("summary") or "").strip()
    if summary:
        out.append(
            MCPMessage(
                role=MessageRole.SYSTEM,
                content=f"Summary of the earlier conversation: {summary}",
            )
        )
    for raw in context.get("messages") or []:
        try:
            role = str(raw.get("role") or "").lower()
            content = str(raw.get("content") or "").strip()
            if role not in _MEMORY_ROLES or not content:
                continue
            out.append(MCPMessage(role=MessageRole(role), content=content))
        except Exception as exc:
            logger.warning("conversation_message_skipped", error=str(exc))
    return out


def _chunks_to_sources(chunks: Any) -> list[dict[str, Any]]:
    """Map retrieval chunks to trimmed source dicts for the wire response.

    Defensive by contract: a malformed chunk must never break the query
    response, so each chunk is mapped in isolation and any failure is logged
    and skipped rather than propagated. Kept as a module function (not inlined
    in ``execute``) so the orchestration stays within the cognitive-complexity
    budget and the mapping is unit-testable on its own.
    """
    sources: list[dict[str, Any]] = []
    for chunk in chunks or []:
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
        except Exception as exc:  # never let source mapping break the query response
            logger.warning("mcp.source_mapping_skipped", error=str(exc)[:200])
    return sources


def _legacy_tenant(tenant: TenantContext) -> Any:
    """Protocol-layer tenant shape the router/assembler still consume."""
    from src.protocol.models import TenantContext as LegacyTenant

    return LegacyTenant(
        tenant_id=tenant.tenant_id,
        user_id=tenant.user_id,
        user_email=tenant.user_email,
        role=tenant.role,
        permissions=list(tenant.permissions),
    )


def _confidence_of(context: Any) -> float:
    """Retrieval confidence for the stream's meta event.

    Derived from the top retrieved chunk's score so the consumer can gate
    auto-respond vs approval BEFORE any token arrives. Defaults mid-scale when
    the assembler exposes no score rather than claiming certainty either way.
    """
    try:
        chunks = getattr(context, "retrieved_chunks", None) or []
        scores = [float(getattr(c, "score", 0.0) or 0.0) for c in chunks]
        return round(max(scores), 4) if scores else 0.6
    except Exception:
        return 0.6


def _stream_request(messages: Any, model: str | None) -> Any:
    """Build the domain LLMRequest for a text-only streamed completion.

    No ``tools``: the provider's stream does not surface them, and the caller
    only reaches this path when the bot has none enabled.
    """
    from src.domain.llm.value_objects import LLMMessage, LLMRequest, MessageRole, ModelId

    out: list[Any] = []
    for m in messages or []:
        role = getattr(m, "role", None)
        role_value = getattr(role, "value", role)
        try:
            mapped = MessageRole(str(role_value).lower())
        except ValueError:
            continue
        out.append(LLMMessage(role=mapped, content=str(getattr(m, "content", "") or "")))
    return LLMRequest(
        messages=tuple(out),
        model=ModelId(model) if model else None,
    )


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

    def __init__(
        self,
        *,
        context_assembler: Any,
        tool_registry: Any,
        llm_router: Any,
        llm_provider: Any = None,
    ) -> None:
        self._assembler = context_assembler
        self._tools = tool_registry
        self._llm = llm_router
        # Token-streaming provider. Optional so existing construction keeps
        # working; when absent, execute_stream degrades to the batched path.
        self._provider = llm_provider

    async def _prepare(
        self,
        *,
        input_: HandleQueryInput,
        tenant: TenantContext,
    ) -> tuple[Any, Any, Any]:
        """Policy, session, context assembly and per-bot tools.

        Extracted so the batched and streamed paths cannot drift: a bot's
        persona, RAG grounding and injection fencing must be identical whichever
        way its answer is delivered. Returns ``(context, tools, session)``.
        """
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

        # 1. Policy — trust the gateway-verified principal. RBAC lives in the
        #    IdP/gateway; MCP requires an authenticated, tenant-scoped caller
        #    and scopes all data by tenant_id.
        if not tenant.tenant_id:
            raise PermissionError("Query denied: no authenticated principal")

        # 2. Session — still ephemeral per request (MCP persists nothing), but
        #    HYDRATED from the caller's conversation memory. core-api owns and
        #    stores the transcript; MCP stays stateless and just receives it.
        session = LegacySession(
            tenant_context=legacy_tenant,
            bot_id=input_.bot_id,
            messages=_conversation_messages(input_.context),
            **({"session_id": input_.session_id} if input_.session_id else {}),
        )

        # 3. Bot config + system prompt + RAG retrieval + injection fencing.
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
        return context, tools, session

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
        legacy_tenant = _legacy_tenant(tenant)

        try:
            # 1-4: policy, session hydration, context assembly, per-bot tools.
            #      Shared verbatim with execute_stream, so a streamed answer is
            #      grounded and personalised identically to a batched one.
            context, tools, session = await self._prepare(input_=input_, tenant=tenant)

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

        # 6. Map retrieved chunks -> source dicts (trimmed for wire size).
        sources = _chunks_to_sources(context.retrieved_chunks)

        tool_calls = [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in (result.tool_calls or [])]
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

    async def execute_stream(
        self,
        *,
        input_: HandleQueryInput,
        tenant: TenantContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """Same pipeline as :meth:`execute`, emitting tokens as they arrive.

        Yields ``{"event": "meta"|"token"|"done", "data": {...}}``.

        WHY THIS EXISTS. A live phone call measured 9,956 ms inside the LLM
        against 217 ms of STT: the caller heard ~15 s of silence because the
        whole answer was assembled before a single byte of audio could be
        synthesized. Emitting tokens lets the caller hear the first sentence
        while the rest is still being generated.

        WHAT IT DELIBERATELY DOES NOT DO. Steps 1-4 are the SAME calls as
        `execute` — policy, context assembly (bot persona + RAG + injection
        fencing) and per-bot tool resolution. Streaming must not become a
        second, thinner pipeline where a bot quietly loses its grounding or its
        identity.

        TOOL-ENABLED BOTS DO NOT STREAM. The provider's `stream` is text-only
        ("tools are not surfaced"), and running the router in stream mode
        previously dropped the tool-calling loop and returned an empty answer.
        So when a bot has tools we run the ordinary batched path and emit the
        finished answer as one token event: the caller still gets a correct
        answer, just without the latency win. Correctness over speed — a fast
        empty answer is worse than a slow right one.
        """
        # Steps 1-4, shared verbatim with execute().
        context, tools, _session = await self._prepare(input_=input_, tenant=tenant)
        legacy_tenant = _legacy_tenant(tenant)

        yield {"event": "meta", "data": {"confidence": _confidence_of(context)}}

        can_stream = self._provider is not None and not tools
        if not can_stream:
            result = await self._llm.execute(
                messages=context.messages,
                tools=tools,
                tenant_context=legacy_tenant,
                stream=False,
                model=input_.model,
            )
            answer = result.answer or ""
            logger.info(
                "mcp.query_stream_batched",
                bot_id=input_.bot_id,
                reason="tools_enabled" if tools else "no_streaming_provider",
                tool_count=len(tools or []),
            )
            yield {"event": "token", "data": {"text": answer}}
            yield {"event": "done", "data": {"text": answer, "confidence": _confidence_of(context)}}
            return

        parts: list[str] = []
        try:
            async for chunk in self._provider.stream(
                _stream_request(context.messages, input_.model),
                tenant=tenant,
            ):
                delta = getattr(chunk, "delta", "") or ""
                if delta:
                    parts.append(delta)
                    yield {"event": "token", "data": {"text": delta}}
        except Exception as e:
            # Mid-stream failure. Tokens already sent cannot be recalled, so
            # close the stream cleanly with what was said rather than raising
            # into the SSE body — the consumer is a live phone call.
            logger.warning("mcp.query_stream_failed", bot_id=input_.bot_id, error=str(e)[:200])

        full = "".join(parts).strip()
        if not full:
            # An empty stream would be silence on the call. Fall back once.
            logger.warning("mcp.query_stream_empty_fallback", bot_id=input_.bot_id)
            result = await self._llm.execute(
                messages=context.messages,
                tools=tools,
                tenant_context=legacy_tenant,
                stream=False,
                model=input_.model,
            )
            full = result.answer or ""
            if full:
                yield {"event": "token", "data": {"text": full}}
        yield {"event": "done", "data": {"text": full, "confidence": _confidence_of(context)}}
