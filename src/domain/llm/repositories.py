"""LLM provider port.

Why a single-method port (rather than separate ``ChatProvider`` /
``CompletionProvider`` / ``StreamingProvider``):
    Every modern LLM (OpenAI, Claude, Gemini, Mistral, Bedrock) is
    really one operation — multi-turn chat with optional tools and
    optional streaming. Splitting the port into roles maps poorly to
    the providers and would force adapters to fake half of each
    interface.

Streaming is folded into ``complete`` via the ``LLMRequest.stream``
flag — the adapter returns the final aggregated response either way.
True token-by-token streaming (yielding partial deltas) would
warrant a separate ``stream(...)`` method; we'll add it the day
``notifications/progress`` over MCP actually emits per-token events.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..shared.tenant import TenantContext
from .value_objects import LLMRequest, LLMResponse


class LLMProvider(ABC):
    """Routes a chat-completion request to the underlying model.

    Adapters MAY:
        * Implement fallback chains (primary → secondary models on
          provider errors).
        * Inject provider-specific API keys.
        * Translate domain types to the provider's wire format and
          back.

    Adapters MUST NOT:
        * Execute tool calls themselves. Tool execution is the
          *caller's* concern — providers surface raw ``tool_calls``
          on the response, the application's tool-call loop
          dispatches them. (This keeps the port general-purpose:
          MCP, fix-agent, and codegen all want raw tool_calls.)
    """

    @abstractmethod
    async def complete(
        self,
        request: LLMRequest,
        *,
        tenant: TenantContext,
    ) -> LLMResponse:
        """Produce one assistant turn. Raise on transport/auth errors;
        return ``LLMResponse`` for both successful completions AND
        provider-side content-filter rejections (the caller can read
        ``finish_reason == CONTENT_FILTER`` and act)."""
