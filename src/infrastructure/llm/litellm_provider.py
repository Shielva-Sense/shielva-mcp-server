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

from collections.abc import AsyncIterator
from typing import Any

import structlog

from src.domain.llm.repositories import LLMProvider
from src.domain.llm.value_objects import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    LLMToolCall,
    LLMUsage,
    MessageRole,
    ModelId,
)
from src.domain.shared.tenant import TenantContext
from src.infrastructure.metering.usage_reporter import report_llm_usage
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
        # 🚨 The allowance gate belongs HERE, on the path that actually runs.
        #
        # It lived only in llm_router, and this adapter calls `acompletion`
        # directly — it uses the router for model/key resolution, not to make
        # the call. Metering was moved here when LLM usage stopped being
        # recorded; the CEILING was left behind, so a workspace could spend
        # past both the included tokens and the output cap and only ever be
        # billed for it after the fact.
        #
        # Fails OPEN, exactly as the router's copy does: a billing lookup that
        # is slow or unconfigured must never be the reason a phone line stops
        # answering. `is_exhausted` refuses only when it positively knows.
        from src.infrastructure.metering.allowance import is_exhausted
        from src.routing.llm_router import AllowanceExhausted

        if await is_exhausted(getattr(tenant, "tenant_id", None)):
            raise AllowanceExhausted(
                "This workspace has used the model tokens included in its plan. Top up in Billing to keep going."
            )

        # Lazy litellm import — adapter is loaded at composition
        # time but acompletion is only resolved on first call.
        from litellm import acompletion

        messages = _to_litellm_messages(request.messages)

        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            kwargs["tools"] = list(request.tools)
            kwargs["tool_choice"] = request.tool_choice or "auto"

        # (model, api_key, api_base) attempts: the per-tenant target first (when
        # the caller didn't pin a model), then the platform fallback chain.
        attempts = await _build_attempts(request, tenant, self._router)

        last_exc: Exception | None = None
        for candidate, ak, base in attempts:
            _apply_routing(kwargs, candidate, ak, base)
            try:
                provider_resp = await acompletion(**kwargs)
                response = _from_litellm_response(provider_resp, model_used=candidate)
                # Metering is scheduled, never awaited — the customer's answer
                # does not wait on a billing write, and a failure here cannot
                # turn a good completion into an error.
                report_llm_usage(
                    tenant_id=getattr(tenant, "tenant_id", None),
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    # 🚨 Cache hits are several times cheaper where a provider
                    # offers them; a ledger that cannot see them overstates
                    # what the traffic cost.
                    cached_tokens=_cached_tokens(getattr(provider_resp, "usage", None)),
                    model_ref=candidate,
                    # Without this the provider column — the reason cost can be
                    # attributed at all — was NULL on every LLM row.
                    provider=candidate.split("/")[0] if "/" in candidate else None,
                    request_id=_request_ref(tenant),
                )
                return response
            except Exception as exc:
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
        # 🚨 The allowance gate belongs HERE, on the path that actually runs.
        #
        # It lived only in llm_router, and this adapter calls `acompletion`
        # directly — it uses the router for model/key resolution, not to make
        # the call. Metering was moved here when LLM usage stopped being
        # recorded; the CEILING was left behind, so a workspace could spend
        # past both the included tokens and the output cap and only ever be
        # billed for it after the fact.
        #
        # Fails OPEN, exactly as the router's copy does: a billing lookup that
        # is slow or unconfigured must never be the reason a phone line stops
        # answering. `is_exhausted` refuses only when it positively knows.
        from src.infrastructure.metering.allowance import is_exhausted
        from src.routing.llm_router import AllowanceExhausted

        if await is_exhausted(getattr(tenant, "tenant_id", None)):
            raise AllowanceExhausted(
                "This workspace has used the model tokens included in its plan. Top up in Billing to keep going."
            )

        from litellm import acompletion

        messages = _to_litellm_messages(request.messages)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            # 🚨 Ask the provider for usage on the final chunk. Without it a
            # streamed answer carries no token counts at all, and streaming was
            # therefore metered as zero — every streamed phone answer was free.
            "stream_options": {"include_usage": True},
        }

        # (model, api_key, api_base) attempts: per-tenant target first, then the
        # platform fallback chain. Connect — fallback applies to connect-time only.
        attempts = await _build_attempts(request, tenant, self._router)
        resp_stream: Any = None
        used_model = attempts[0][0]
        last_exc: Exception | None = None
        for candidate, ak, base in attempts:
            _apply_routing(kwargs, candidate, ak, base)
            try:
                resp_stream = await acompletion(**kwargs)
                used_model = candidate
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "mcp.llm_stream_connect_failed",
                    model=candidate,
                    tenant_id=tenant.tenant_id,
                    error=str(exc)[:200],
                )

        if resp_stream is None:
            assert last_exc is not None
            raise last_exc

        # Usage arrives on the final chunk when the provider supports it, and
        # is accumulated rather than read once: a provider that reports per
        # chunk would otherwise have only its last chunk billed.
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        async for part in resp_stream:
            usage = getattr(part, "usage", None)
            if usage is not None:
                prompt_tokens = max(prompt_tokens, int(getattr(usage, "prompt_tokens", 0) or 0))
                completion_tokens = max(completion_tokens, int(getattr(usage, "completion_tokens", 0) or 0))
                cached_tokens = max(cached_tokens, _cached_tokens(usage))

            choices = getattr(part, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta_obj = getattr(choice, "delta", None)
            text = (getattr(delta_obj, "content", None) or "") if delta_obj else ""
            finish_raw = getattr(choice, "finish_reason", None)
            finish: FinishReason | None = None
            if finish_raw:
                try:
                    finish = FinishReason(str(finish_raw).lower())
                except ValueError:
                    finish = FinishReason.OTHER
            if text or finish is not None:
                yield LLMStreamChunk(
                    delta=text,
                    finish_reason=finish,
                    model=ModelId(used_model),
                )

        # Reported once the stream is exhausted, for the same reason the
        # non-streaming path reports after the completion: the customer's
        # answer never waits on a billing write.
        if prompt_tokens or completion_tokens:
            report_llm_usage(
                tenant_id=getattr(tenant, "tenant_id", None),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                model_ref=used_model,
                provider=used_model.split("/")[0] if "/" in used_model else None,
                request_id=_request_ref(tenant),
            )


# ── helpers ───────────────────────────────────────────────────────


def _request_ref(tenant: TenantContext) -> str:
    """A stable id for one inbound request, so a redelivered report dedupes.

    🚨 This read `tenant.request_id`, an attribute TenantContext does not have
    — it is a frozen slots dataclass of tenant_id, user_id, user_email, role
    and permissions — so the getattr default always won, every report got a
    fresh UUID, and the partial unique index could never absorb a redelivery.
    """
    import uuid

    for attr in ("request_id", "correlation_id", "trace_id"):
        value = getattr(tenant, attr, None)
        if value:
            return str(value)
    return str(uuid.uuid4())


def _cached_tokens(usage: Any) -> int:
    """Prompt tokens the provider served from cache, when it reports them."""
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    for attr in ("cached_tokens", "cache_read_input_tokens"):
        value = getattr(details, attr, None) if details is not None else None
        if value is None:
            value = getattr(usage, attr, None)
        if value:
            return int(value)
    return 0


async def _build_attempts(
    request: LLMRequest,
    tenant: TenantContext,
    router: Any,
) -> list[tuple]:
    """Ordered (model, api_key, api_base) attempts.

    When the caller pinned an explicit ``request.model`` it is honoured as-is.
    Otherwise we ask shielva-platform for the tenant's active provider/model +
    BYOK key (per-tenant routing) and put that first; a miss falls through to the
    platform default. The platform fallback chain (with platform keys) always
    follows, so a tenant key failure still degrades gracefully.
    """
    primary_model: str | None = str(request.model) if request.model else None
    primary_key: str | None = None
    primary_base: str | None = None

    if request.model is None:
        resolved = await get_tenant_llm_resolver().resolve(getattr(tenant, "tenant_id", None))
        if resolved is not None:
            primary_model = resolved.model
            primary_key = resolved.api_key
            primary_base = resolved.api_base

    if primary_model is None:
        primary_model = router.default_model
    if primary_key is None:
        primary_key = router._get_api_key(primary_model)

    attempts: list[tuple] = [(primary_model, primary_key, primary_base)]
    for fb in router.fallback_models:
        if fb != primary_model:
            attempts.append((fb, router._get_api_key(fb), None))
    return attempts


def _apply_routing(
    kwargs: dict[str, Any],
    model: str,
    api_key: str | None,
    api_base: str | None,
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


def _to_litellm_messages(messages) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        # 🚨 Parts WIN when present: a multimodal turn's payload is the list,
        # and LiteLLM passes the OpenAI content-parts shape through to whichever
        # provider is configured. This is what lets the workspace's own
        # provisioned model read a scanned page — no OCR vendor, no second
        # pipeline, and the call is metered exactly like any other.
        d: dict[str, Any] = {
            "role": m.role.value,
            "content": list(m.parts) if m.parts else (m.content or ""),
        }
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
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
            id=getattr(tc, "id", "") or "",
            name=(getattr(tc, "function", None).name if getattr(tc, "function", None) else "") or "",
            arguments=(getattr(tc, "function", None).arguments if getattr(tc, "function", None) else "{}") or "{}",
        )
        for tc in raw_tool_calls
    )

    usage_obj = getattr(provider_resp, "usage", None)
    usage = LLMUsage(
        prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
    )

    return LLMResponse(
        content=getattr(msg, "content", None) or "",
        tool_calls=tool_calls,
        model=ModelId(model_used),
        finish_reason=finish,
        usage=usage,
    )
