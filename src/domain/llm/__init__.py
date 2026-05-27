"""LLM bounded context.

Owns the abstraction over Large Language Model providers. The
domain is provider-agnostic: an :class:`LLMProvider` port exposes a
``complete(...)`` method that takes a tenant-scoped
:class:`LLMRequest` and returns an :class:`LLMResponse` — with
optional ``tool_calls`` for caller-driven tool loops.

Why this is a separate bounded context (not a value object inside
``chat``):
    * LLM choice is orthogonal to chat session lifecycle. The same
      provider serves codegen, fix-agent, RAG queries, and chat.
    * Provider swapping (LiteLLM → direct Gemini SDK → Bedrock) is
      a real ops concern; isolating it behind a port keeps the
      blast radius bounded.

Adapters under ``infrastructure/llm/`` implement the port. The
slice-3 adapter wraps the existing LiteLLM-based router so nothing
on the wire changes.
"""
from .repositories import LLMProvider
from .value_objects import (
    FinishReason, LLMMessage, LLMRequest, LLMResponse, LLMToolCall,
    LLMUsage, MessageRole, ModelId,
)

__all__ = [
    "LLMProvider",
    "LLMMessage", "LLMRequest", "LLMResponse", "LLMToolCall",
    "LLMUsage", "FinishReason", "MessageRole", "ModelId",
]
