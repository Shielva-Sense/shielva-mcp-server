"""LiteLLM-backed LLMProvider adapter.

Wraps the existing :class:`src.routing.llm_router.LLMRouter` so the
new use-case path can call the domain port without duplicating the
provider key + fallback + tool-loop logic. The router is shared
with the legacy codegen/fix-agent paths during the slice 3-4
transition; once those callers migrate, this adapter becomes the
sole owner of LiteLLM interaction and the legacy router can be
deleted.

Translation rules
-----------------
``LLMRequest.messages`` → list of OpenAI-format dicts. LiteLLM
accepts that shape verbatim across every provider it abstracts
(Gemini, Anthropic, OpenAI, Bedrock, …).

``LLMRequest.tools`` → passed through. Each tool dict is already in
OpenAI ``{"type":"function","function":{...}}`` shape — the same
shape ``LLMRouter._prepare_tools`` would have produced from ToolSpecs,
but pre-built. We bypass ``LLMRouter.execute`` because that method
runs a *local tool execution loop* via ``tool_registry`` — we want
raw ``tool_calls`` surfaced to the caller so MCP's tool-call loop
can run in the application layer (which talks to the domain
``ToolExecutor`` port, not the legacy registry).

Concretely: we call ``litellm.acompletion`` directly (same as
:class:`LLMRouter._execute_sync`'s inner provider call) and skip the
router's loop.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from src.domain.llm.repositories import LLMProvider
from src.domain.llm.value_objects import (
    FinishReason, LLMMessage, LLMRequest, LLMResponse, LLMStreamChunk, LLMToolCall,
    LLMUsage, MessageRole, ModelId,
)
from src.domain.shared.tenant import TenantContext
from src.routing.tenant_llm_resolver import get_tenant_llm_resolver

logger = structlog.get_logger(__name__)


class LiteLLMProviderAdapter(LLMProvider):
    """Single-turn LiteLLM caller with provider fallback.

    The legacy router is injected via composition so we share
    process-level config (default model, fallback chain, api keys).
    No second LiteLLM client — same import, same global config.
    """

    def __init__(self, legacy_router: Any) -> None:
        self._router = legacy_router

    async def complete(
        self,
        request: LLMRequest,
        *,
        tenant: TenantContext,
    ) -> LLMResponse:
        # Lazy litellm import — adapter is loaded at composition
        # time but acompletion is only resolved on first call.
        from litellm import acompletion

        messages = _to_litellm_messages(request.messages)

        kwargs: Dict[str, Any] = {
            "messages":    messages,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            kwargs["tools"]       = list(request.tools)
            kwargs["tool_choice"] = request.tool_choice or "auto"

        # (model, api_key, api_base) attempts: the per-tenant target first (when
        # the caller didn't pin a model), then the platform fallback chain.
        attempts = await _build_attempts(request, tenant, self._router)

        last_exc: Optional[Exception] = None
        for candidate, ak, base in attempts:
            _apply_routing(kwargs, candidate, ak, base)
            try:
                provider_resp = await acompletion(**kwargs)
                return _from_litellm_response(provider_resp, model_used=candidate)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "mcp.llm_provider_attempt_failed",
                    model=candidate,
                    tenant_id=tenant.tenant_id,
                    error=str(exc)[:200],
                )

        # All providers exhausted.
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        request: LLMRequest,
        *,
        tenant: TenantContext,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Token-by-token streaming via ``litellm.acompletion(stream=True)``.

        Provider fallback covers only the INITIAL connection: once the stream
        begins emitting tokens we never switch providers (that would duplicate
        or garble already-sent output). A mid-stream error propagates. Streaming
        is text-only here — tools are not surfaced (the streaming consumers are
        plain-text explanations / analyses).
        """
        from litellm import acompletion

        messages = _to_litellm_messages(request.messages)
        kwargs: Dict[str, Any] = {
            "messages":    messages,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
            "stream":      True,
        }

        # (model, api_key, api_base) attempts: per-tenant target first, then the
        # platform fallback chain. Connect — fallback applies to connect-time only.
        attempts = await _build_attempts(request, tenant, self._router)
        resp_stream: Any = None
        used_model = attempts[0][0]
        last_exc: Optional[Exception] = None
        for candidate, ak, base in attempts:
            _apply_routing(kwargs, candidate, ak, base)
            try:
                resp_stream = await acompletion(**kwargs)
                used_model = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "mcp.llm_stream_connect_failed",
                    model=candidate, tenant_id=tenant.tenant_id, error=str(exc)[:200],
                )

        if resp_stream is None:
            assert last_exc is not None
            raise last_exc

        async for part in resp_stream:
            choices = getattr(part, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta_obj = getattr(choice, "delta", None)
            text = (getattr(delta_obj, "content", None) or "") if delta_obj else ""
            finish_raw = getattr(choice, "finish_reason", None)
            finish: Optional[FinishReason] = None
            if finish_raw:
                try:
                    finish = FinishReason(str(finish_raw).lower())
                except ValueError:
                    finish = FinishReason.OTHER
            if text or finish is not None:
                yield LLMStreamChunk(
                    delta=text, finish_reason=finish, model=ModelId(used_model),
                )


# ── helpers ───────────────────────────────────────────────────────

async def _build_attempts(
    request: LLMRequest,
    tenant: TenantContext,
    router: Any,
) -> List[tuple]:
    """Ordered (model, api_key, api_base) attempts.

    When the caller pinned an explicit ``request.model`` it is honoured as-is.
    Otherwise we ask shielva-platform for the tenant's active provider/model +
    BYOK key (per-tenant routing) and put that first; a miss falls through to the
    platform default. The platform fallback chain (with platform keys) always
    follows, so a tenant key failure still degrades gracefully.
    """
    primary_model: Optional[str] = str(request.model) if request.model else None
    primary_key: Optional[str] = None
    primary_base: Optional[str] = None

    if request.model is None:
        resolved = await get_tenant_llm_resolver().resolve(
            getattr(tenant, "tenant_id", None)
        )
        if resolved is not None:
            primary_model = resolved.model
            primary_key = resolved.api_key
            primary_base = resolved.api_base

    if primary_model is None:
        primary_model = router.default_model
    if primary_key is None:
        primary_key = router._get_api_key(primary_model)  # noqa: SLF001

    attempts: List[tuple] = [(primary_model, primary_key, primary_base)]
    for fb in router.fallback_models:
        if fb != primary_model:
            attempts.append((fb, router._get_api_key(fb), None))  # noqa: SLF001
    return attempts


def _apply_routing(
    kwargs: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    api_base: Optional[str],
) -> None:
    """Set model/api_key/api_base on the litellm kwargs, clearing stale values."""
    kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs.pop("api_key", None)
    if api_base:
        kwargs["api_base"] = api_base
    else:
        kwargs.pop("api_base", None)


def _to_litellm_messages(messages) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        d: Dict[str, Any] = {"role": m.role.value, "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                }
                for tc in m.tool_calls
            ]
        if m.role == MessageRole.TOOL:
            d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
        out.append(d)
    return out


def _from_litellm_response(provider_resp: Any, *, model_used: str) -> LLMResponse:
    """Provider responses follow the OpenAI shape via LiteLLM.

    ``choices[0].message`` carries ``content`` and optionally
    ``tool_calls``. ``choices[0].finish_reason`` is one of
    ``stop / length / tool_calls / content_filter``."""
    choice = provider_resp.choices[0]
    msg = choice.message
    finish_raw = (getattr(choice, "finish_reason", None) or "other").lower()
    try:
        finish = FinishReason(finish_raw)
    except ValueError:
        finish = FinishReason.OTHER

    raw_tool_calls = getattr(msg, "tool_calls", None) or ()
    tool_calls = tuple(
        LLMToolCall(
            id        = getattr(tc, "id", "") or "",
            name      = (getattr(tc, "function", None).name
                         if getattr(tc, "function", None) else "") or "",
            arguments = (getattr(tc, "function", None).arguments
                         if getattr(tc, "function", None) else "{}") or "{}",
        )
        for tc in raw_tool_calls
    )

    usage_obj = getattr(provider_resp, "usage", None)
    usage = LLMUsage(
        prompt_tokens     = int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0),
        total_tokens      = int(getattr(usage_obj, "total_tokens", 0) or 0),
    )

    return LLMResponse(
        content       = getattr(msg, "content", None) or "",
        tool_calls    = tool_calls,
        model         = ModelId(model_used),
        finish_reason = finish,
        usage         = usage,
    )
