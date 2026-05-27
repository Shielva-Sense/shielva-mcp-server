"""LLM provider adapters — implement ``domain.llm.LLMProvider``.

Slice 3 ships :class:`LiteLLMProviderAdapter` which wraps the
existing ``src.routing.llm_router.LLMRouter``. Once codegen +
fix-agent move onto the new port in slice 4, the wrapper becomes a
proper standalone adapter (and the legacy router can be deleted).
"""
from .litellm_provider import LiteLLMProviderAdapter

__all__ = ["LiteLLMProviderAdapter"]
