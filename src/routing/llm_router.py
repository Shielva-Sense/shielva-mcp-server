"""
MCP LLM Router
Routes LLM requests through LiteLLM with:
- Provider abstraction (OpenAI, Anthropic, Azure)
- Tool/function calling
- Streaming support
- Fallback handling
- Token tracking
"""

import json
from dataclasses import dataclass, field
from typing import Any

import litellm
import structlog
from litellm import acompletion

from config.settings import get_settings
from src.protocol.models import Source, TenantContext, ToolCall, ToolCallStatus
from src.routing.tenant_llm_resolver import (
    format_model_for_provider,
    get_tenant_llm_resolver,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


def _unwrap_secret(value):
    """Return the plaintext of a SecretStr-like setting (or the value itself).

    Sealed-config holds API keys as ``SecretStr``; litellm needs the raw string.
    Returns ``None`` for empty/missing keys so the caller falls through cleanly.
    """
    if value is None:
        return None
    raw = value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)
    return raw or None


# Configure LiteLLM
litellm.set_verbose = settings.debug


@dataclass
class LLMResponse:
    """Response from LLM execution"""

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    tokens_used: int = 0
    model: str = ""
    finish_reason: str = ""


@dataclass
class ToolSpec:
    """Tool specification for LLM"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Any  # Async function to call


class LLMRouter:
    """
    Routes LLM requests through LiteLLM.

    Features:
    - Multi-provider support
    - Tool/function calling
    - Streaming
    - Automatic fallback
    - Token usage tracking
    """

    def __init__(
        self,
        tool_registry=None,
        default_model: str = None,
        fallback_models: list[str] = None,
    ):
        """
        Initialize LLM router.

        Args:
            tool_registry: Registry for tool execution
            default_model: Primary model to use
            fallback_models: Fallback models if primary fails
        """
        self.tool_registry = tool_registry
        self.default_model = default_model or settings.litellm_model
        self.fallback_models = fallback_models or settings.litellm_fallback_models

        logger.info(
            "LLMRouter initialized",
            default_model=self.default_model,
            fallbacks=self.fallback_models,
        )

    async def execute(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] = None,
        tenant_context: TenantContext = None,
        stream: bool = False,
        model: str = None,
    ) -> LLMResponse:
        """
        Execute LLM request with optional tools.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tools for function calling
            tenant_context: Tenant context for tracking
            stream: Whether to stream response
            model: Optional model override (an explicit caller override wins over
                   per-tenant routing)

        Returns:
            LLMResponse with answer and metadata
        """
        # Per-tenant routing: when the caller did NOT pin a model, ask
        # shielva-platform for this tenant's active provider/model + BYOK key.
        # A miss (no config / managed tier / error) returns None and we fall
        # through to the platform default — never a regression.
        # `model` here is an optional PER-BOT override (a bare model id like
        # "gemini-2.5-pro"). The tenant config is ALWAYS resolved for the key +
        # provider + tenant-default model — even when a per-bot model is given —
        # so a BYOK tenant keeps using its own key with the bot's chosen model
        # (previously an explicit model skipped resolution and leaked to the
        # platform key). The bot's bare model is formatted with the tenant's
        # provider; with no per-bot override the tenant default is used.
        tenant_api_key: str | None = None
        tenant_api_base: str | None = None
        if tenant_context is not None:
            resolved = await get_tenant_llm_resolver().resolve(getattr(tenant_context, "tenant_id", None))
            if resolved is not None:
                tenant_api_key = resolved.api_key
                tenant_api_base = resolved.api_base
                model = format_model_for_provider(resolved.provider, model) if model else resolved.model
            elif model:
                # No tenant config (platform default) but a per-bot override is
                # set → format it with the platform's default provider.
                model = format_model_for_provider(settings.default_llm_provider, model)

        model = model or self.default_model

        logger.info(
            "Executing LLM request",
            model=model,
            num_messages=len(messages),
            has_tools=bool(tools),
            stream=stream,
            tenant_routed=bool(tenant_api_key or tenant_api_base),
        )

        try:
            if stream:
                return await self._execute_streaming(
                    messages=messages,
                    tools=tools,
                    model=model,
                    tenant_context=tenant_context,
                    api_key=tenant_api_key,
                    api_base=tenant_api_base,
                )
            return await self._execute_sync(
                messages=messages,
                tools=tools,
                model=model,
                tenant_context=tenant_context,
                api_key=tenant_api_key,
                api_base=tenant_api_base,
            )

        except Exception as e:
            logger.error("LLM execution failed", error=str(e), model=model)

            # Try fallback models
            for fallback in self.fallback_models:
                if fallback != model:
                    logger.info("Trying fallback model", fallback=fallback)
                    try:
                        return await self._execute_sync(
                            messages=messages,
                            tools=tools,
                            model=fallback,
                            tenant_context=tenant_context,
                        )
                    except Exception as fallback_error:
                        logger.error("Fallback failed", model=fallback, error=str(fallback_error))

            raise

    async def _execute_sync(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec],
        model: str,
        tenant_context: TenantContext,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> LLMResponse:
        """
        Execute synchronous (non-streaming) LLM request.

        Handles tool calls in a loop until completion.

        api_key/api_base, when provided, are the tenant's BYOK credentials
        (per-tenant routing); otherwise the platform key for the model is used.
        """
        all_tool_calls = []
        current_messages = messages.copy()
        extra = {"api_base": api_base} if api_base else {}

        # Tool calling loop
        max_iterations = 5
        for iteration in range(max_iterations):
            # Prepare tools for LiteLLM
            litellm_tools = self._prepare_tools(tools) if tools else None

            # Call LLM
            response = await acompletion(
                model=model,
                messages=current_messages,
                tools=litellm_tools,
                tool_choice="auto" if litellm_tools else None,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                api_key=api_key or self._get_api_key(model),
                **extra,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Check for tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Execute tool calls
                tool_results = await self._execute_tools(message.tool_calls, tools, tenant_context)

                all_tool_calls.extend(tool_results)

                # Add assistant message with tool calls
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                # Add tool results
                for result in tool_results:
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.id,
                            "content": json.dumps(result.result) if result.result else result.error,
                        }
                    )
            else:
                # No more tool calls, return response
                return LLMResponse(
                    answer=message.content or "",
                    tool_calls=all_tool_calls,
                    sources=[],  # Sources would come from RAG
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    model=model,
                    finish_reason=finish_reason,
                )

        # Max iterations reached
        logger.warning("Max tool iterations reached")
        return LLMResponse(
            answer="I apologize, but I'm having trouble completing this request.",
            tool_calls=all_tool_calls,
            tokens_used=0,
            model=model,
            finish_reason="max_iterations",
        )

    async def _execute_streaming(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec],
        model: str,
        tenant_context: TenantContext,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> LLMResponse:
        """
        Execute a "streaming" LLM request.

        The one consumer — ``/mcp/v1/query/stream`` — aggregates the whole
        answer and emits it as a single SSE event (there is no token-by-token
        streaming at this layer yet; that arrives with the domain LLM provider's
        streaming variant). Running LiteLLM in ``stream=True`` mode here bought
        no token streaming AND silently dropped the tool-calling loop, so a
        tool-enabled bot returned an empty answer. We therefore run the same
        tool-capable sync path; the endpoint aggregates identically. True
        token streaming (with mid-stream tool handling) lives in
        ``infrastructure.llm.LiteLLMProviderAdapter.stream``.

        api_key/api_base, when provided, are the tenant's BYOK credentials.
        """
        return await self._execute_sync(
            messages=messages,
            tools=tools,
            model=model,
            tenant_context=tenant_context,
            api_key=api_key,
            api_base=api_base,
        )

    def _get_api_key(self, model: str) -> str | None:
        """Return the correct API key for a given model string.

        Settings hold these as ``SecretStr`` (sealed-config / envelope). They
        MUST be unwrapped before reaching litellm — passing the SecretStr
        object makes litellm stringify it to ``'**********'``, which the
        providers reject as ``API_KEY_INVALID``.
        """
        if "gemini" in model:
            return _unwrap_secret(settings.gemini_api_key)
        if "claude" in model or "anthropic" in model:
            return _unwrap_secret(settings.anthropic_api_key)
        if "gpt" in model or "openai" in model:
            return _unwrap_secret(settings.openai_api_key)
        if "azure" in model:
            return _unwrap_secret(settings.azure_openai_api_key)
        return None

    def _prepare_tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """
        Convert tool specs to LiteLLM/OpenAI format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of tool definitions in OpenAI format
        """
        if not tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    async def _execute_tools(
        self,
        tool_calls: list[Any],
        tools: list[ToolSpec],
        tenant_context: TenantContext,
    ) -> list[ToolCall]:
        """
        Execute tool calls and return results.

        Args:
            tool_calls: Tool calls from LLM
            tools: Available tool specs
            tenant_context: Tenant context

        Returns:
            List of ToolCall results
        """
        results = []

        # Build tool lookup
        tool_map = {tool.name: tool for tool in tools}

        for tc in tool_calls:
            tool_name = tc.function.name

            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            logger.info("Executing tool", tool_name=tool_name, arguments=arguments)

            tool_call = ToolCall(
                id=tc.id,
                tool_name=tool_name,
                arguments=arguments,
                status=ToolCallStatus.RUNNING,
            )

            try:
                if tool_name in tool_map:
                    tool = tool_map[tool_name]
                    result = await tool.handler(tenant_context=tenant_context, **arguments)
                    tool_call.result = result
                    tool_call.status = ToolCallStatus.COMPLETED
                else:
                    tool_call.error = f"Unknown tool: {tool_name}"
                    tool_call.status = ToolCallStatus.FAILED

            except Exception as e:
                logger.error("Tool execution failed", tool=tool_name, error=str(e))
                tool_call.error = str(e)
                tool_call.status = ToolCallStatus.FAILED

            results.append(tool_call)

        return results
