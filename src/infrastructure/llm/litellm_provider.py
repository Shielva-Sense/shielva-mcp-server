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

from typing import Any, Dict, List, Optional

import structlog

from src.domain.llm.repositories import LLMProvider
from src.domain.llm.value_objects import (
    FinishReason, LLMMessage, LLMRequest, LLMResponse, LLMToolCall, LLMUsage,
    MessageRole, ModelId,
)
from src.domain.shared.tenant import TenantContext

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

        model = str(request.model) if request.model else self._router.default_model

        messages = _to_litellm_messages(request.messages)

        kwargs: Dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  request.max_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            kwargs["tools"]       = list(request.tools)
            kwargs["tool_choice"] = request.tool_choice or "auto"

        api_key = self._router._get_api_key(model)  # noqa: SLF001 — slice 4 lifts
        if api_key:
            kwargs["api_key"] = api_key

        last_exc: Optional[Exception] = None
        candidates = [model, *(m for m in self._router.fallback_models if m != model)]
        for candidate in candidates:
            kwargs["model"] = candidate
            ak = self._router._get_api_key(candidate)  # noqa: SLF001
            if ak:
                kwargs["api_key"] = ak
            elif "api_key" in kwargs:
                kwargs.pop("api_key", None)
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


# ── helpers ───────────────────────────────────────────────────────

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
