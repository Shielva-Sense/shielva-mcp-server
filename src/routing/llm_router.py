"""
MCP LLM Router
Routes LLM requests through LiteLLM with:
- Provider abstraction (OpenAI, Anthropic, Azure)
- Tool/function calling
- Streaming support
- Fallback handling
- Token tracking
"""
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
import structlog
import litellm
from litellm import acompletion
import json

from src.protocol.models import TenantContext, ToolCall, ToolCallStatus, Source
from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Configure LiteLLM
litellm.set_verbose = settings.debug


@dataclass
class LLMResponse:
    """Response from LLM execution"""
    answer: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    tokens_used: int = 0
    model: str = ""
    finish_reason: str = ""


@dataclass
class ToolSpec:
    """Tool specification for LLM"""
    name: str
    description: str
    parameters: Dict[str, Any]
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
        fallback_models: List[str] = None
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
            fallbacks=self.fallback_models
        )
    
    async def execute(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolSpec] = None,
        tenant_context: TenantContext = None,
        stream: bool = False,
        model: str = None
    ) -> LLMResponse:
        """
        Execute LLM request with optional tools.
        
        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tools for function calling
            tenant_context: Tenant context for tracking
            stream: Whether to stream response
            model: Optional model override
            
        Returns:
            LLMResponse with answer and metadata
        """
        model = model or self.default_model
        
        logger.info(
            "Executing LLM request",
            model=model,
            num_messages=len(messages),
            has_tools=bool(tools),
            stream=stream
        )
        
        try:
            if stream:
                return await self._execute_streaming(
                    messages=messages,
                    tools=tools,
                    model=model,
                    tenant_context=tenant_context
                )
            else:
                return await self._execute_sync(
                    messages=messages,
                    tools=tools,
                    model=model,
                    tenant_context=tenant_context
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
                            tenant_context=tenant_context
                        )
                    except Exception as fallback_error:
                        logger.error(
                            "Fallback failed",
                            model=fallback,
                            error=str(fallback_error)
                        )
            
            raise
    
    async def _execute_sync(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolSpec],
        model: str,
        tenant_context: TenantContext
    ) -> LLMResponse:
        """
        Execute synchronous (non-streaming) LLM request.
        
        Handles tool calls in a loop until completion.
        """
        all_tool_calls = []
        current_messages = messages.copy()
        
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
                api_key=self._get_api_key(model)
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Execute tool calls
                tool_results = await self._execute_tools(
                    message.tool_calls,
                    tools,
                    tenant_context
                )
                
                all_tool_calls.extend(tool_results)
                
                # Add assistant message with tool calls
                current_messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                # Add tool results
                for result in tool_results:
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": result.id,
                        "content": json.dumps(result.result) if result.result else result.error
                    })
            else:
                # No more tool calls, return response
                return LLMResponse(
                    answer=message.content or "",
                    tool_calls=all_tool_calls,
                    sources=[],  # Sources would come from RAG
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                    model=model,
                    finish_reason=finish_reason
                )
        
        # Max iterations reached
        logger.warning("Max tool iterations reached")
        return LLMResponse(
            answer="I apologize, but I'm having trouble completing this request.",
            tool_calls=all_tool_calls,
            tokens_used=0,
            model=model,
            finish_reason="max_iterations"
        )
    
    async def _execute_streaming(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolSpec],
        model: str,
        tenant_context: TenantContext
    ) -> LLMResponse:
        """
        Execute streaming LLM request.
        
        Yields tokens as they're generated.
        """
        litellm_tools = self._prepare_tools(tools) if tools else None
        
        response = await acompletion(
            model=model,
            messages=messages,
            tools=litellm_tools,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            api_key=self._get_api_key(model),
            stream=True
        )

        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token

        # Return final response
        return LLMResponse(
            answer=full_response,
            tool_calls=[],
            tokens_used=0,  # Not available in streaming
            model=model,
            finish_reason="stop"
        )

    def _get_api_key(self, model: str) -> Optional[str]:
        """Return the correct API key for a given model string."""
        if "gemini" in model:
            return settings.gemini_api_key
        if "claude" in model or "anthropic" in model:
            return settings.anthropic_api_key
        if "gpt" in model or "openai" in model:
            return settings.openai_api_key
        if "azure" in model:
            return settings.azure_openai_api_key
        return None

    def _prepare_tools(self, tools: List[ToolSpec]) -> List[Dict[str, Any]]:
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
                    "parameters": tool.parameters
                }
            }
            for tool in tools
        ]
    
    async def _execute_tools(
        self,
        tool_calls: List[Any],
        tools: List[ToolSpec],
        tenant_context: TenantContext
    ) -> List[ToolCall]:
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
            
            logger.info(
                "Executing tool",
                tool_name=tool_name,
                arguments=arguments
            )
            
            tool_call = ToolCall(
                id=tc.id,
                tool_name=tool_name,
                arguments=arguments,
                status=ToolCallStatus.RUNNING
            )
            
            try:
                if tool_name in tool_map:
                    tool = tool_map[tool_name]
                    result = await tool.handler(
                        tenant_context=tenant_context,
                        **arguments
                    )
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
