"""
MCP Tool Registry
Manages tool registration, permissions, and execution.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

from src.protocol.models import (
    TenantContext,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from src.routing.llm_router import ToolSpec

logger = structlog.get_logger(__name__)


@dataclass
class RegisteredTool:
    """A tool registered in the system"""

    definition: ToolDefinition
    handler: Callable
    enabled: bool = True


class ToolRegistry:
    """
    Registry for MCP tools.

    Manages:
    - Tool registration
    - Permission checking
    - Tool execution dispatch
    - Bot-specific tool configuration
    """

    def __init__(self, bot_registry=None):
        """Initialize tool registry."""
        self._tools: dict[str, RegisteredTool] = {}
        self._bot_tool_configs: dict[str, dict[str, bool]] = {}
        self.bot_registry = bot_registry
        self.rag_client = None

        logger.info("ToolRegistry initialized")

    def set_rag_client(self, rag_client: Any) -> None:
        """Set the RAG client for knowledge tools."""
        self.rag_client = rag_client
        logger.info("RAG client set in ToolRegistry")

    def set_bot_registry(self, bot_registry: Any) -> None:
        """Set the bot registry for context lookup."""
        self.bot_registry = bot_registry
        logger.info("Bot registry set in ToolRegistry")

    def register(self, definition: ToolDefinition, handler: Callable) -> None:
        """
        Register a new tool.

        Args:
            definition: Tool definition
            handler: Async function to execute tool
        """
        logger.info("Registering tool", tool_name=definition.name)

        self._tools[definition.name] = RegisteredTool(
            definition=definition,
            handler=handler,
            enabled=definition.enabled_by_default,
        )

    def get_all_tools(self) -> list[ToolDefinition]:
        """Get all registered tool definitions."""
        return [tool.definition for tool in self._tools.values()]

    async def get_tools_for_bot(
        self,
        bot_id: str,
        tenant_context: TenantContext,
        enabled_tools: dict[str, bool] = None,
    ) -> list[ToolSpec]:
        """
        Get tools available for a specific bot.

        Filters based on:
        - Bot configuration
        - User permissions
        - Request-time overrides

        Args:
            bot_id: Bot identifier
            tenant_context: Tenant context
            enabled_tools: Runtime tool overrides

        Returns:
            List of ToolSpec objects for LLM
        """
        specs = []

        for name, tool in self._tools.items():
            # Check if tool is enabled for this bot
            if not self._is_tool_enabled(name, bot_id, enabled_tools):
                continue

            # Check permissions
            if not self._check_permissions(tool.definition, tenant_context):
                continue

            # Special handling for rag_query to inject bot-specific KBs
            handler = tool.handler
            if name == "rag_query":
                # Create a closure to inject bot-specific defaults
                async def wrapped_rag_handler(tenant_context, query, kb_ids=None, top_k=5):
                    # If LLM didn't provide KB IDs, try to fetch from bot config
                    if not kb_ids and self.bot_registry:
                        bot_config = await self.bot_registry.get_bot(bot_id, tenant_context.tenant_id)
                        if bot_config:
                            kb_ids = bot_config.get("kb_ids", [])

                    # Call the actual rag_query_tool logic (which might be self.rag_query_tool)
                    # We pass 'self' because it might be a method or we can call it directly
                    return await self.rag_query_tool(tenant_context, query, kb_ids=kb_ids, top_k=top_k)

                handler = wrapped_rag_handler

            # Create ToolSpec
            specs.append(
                ToolSpec(
                    name=name,
                    description=tool.definition.description,
                    parameters=self._build_parameters_schema(tool.definition),
                    handler=handler,
                )
            )

        logger.info("Tools prepared for bot", bot_id=bot_id, num_tools=len(specs))

        return specs

    async def execute_tool(self, request: ToolExecutionRequest, tenant_context: TenantContext) -> ToolExecutionResponse:
        """
        Execute a tool directly.

        Args:
            request: Tool execution request
            tenant_context: Tenant context

        Returns:
            ToolExecutionResponse
        """
        import time

        start = time.time()

        tool_name = request.tool_name

        if tool_name not in self._tools:
            return ToolExecutionResponse(
                tool_name=tool_name,
                result=None,
                success=False,
                error=f"Tool not found: {tool_name}",
                duration_ms=0,
            )

        tool = self._tools[tool_name]

        # Check permissions
        if not self._check_permissions(tool.definition, tenant_context):
            return ToolExecutionResponse(
                tool_name=tool_name,
                result=None,
                success=False,
                error="Permission denied",
                duration_ms=0,
            )

        try:
            result = await tool.handler(tenant_context=tenant_context, **request.parameters)

            duration_ms = int((time.time() - start) * 1000)

            return ToolExecutionResponse(
                tool_name=tool_name,
                result=result,
                success=True,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error("Tool execution failed", tool=tool_name, error=str(e))

            return ToolExecutionResponse(
                tool_name=tool_name,
                result=None,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

    def configure_bot_tools(self, bot_id: str, tool_config: dict[str, bool]) -> None:
        """
        Configure which tools are enabled for a bot.

        Args:
            bot_id: Bot identifier
            tool_config: Dict of tool_name -> enabled
        """
        self._bot_tool_configs[bot_id] = tool_config
        logger.info("Bot tools configured", bot_id=bot_id, config=tool_config)

    def _is_tool_enabled(self, tool_name: str, bot_id: str, overrides: dict[str, bool] = None) -> bool:
        """Check if tool is enabled for bot."""
        # Runtime override takes precedence
        if overrides and tool_name in overrides:
            return overrides[tool_name]

        # Check bot-specific config
        if bot_id in self._bot_tool_configs:
            bot_config = self._bot_tool_configs[bot_id]
            if tool_name in bot_config:
                return bot_config[tool_name]

        # Fall back to tool default
        if tool_name in self._tools:
            return self._tools[tool_name].enabled

        return False

    def _check_permissions(self, definition: ToolDefinition, tenant_context: TenantContext) -> bool:
        """Check if user has permission to use tool."""
        # Shielva-internal service callers (identity == internal@shielva.ai)
        # bypass tenant tool-permission gating: internal infra (e.g. the
        # integration builder's codegen-guideline RAG) is not tenant feature
        # use. Identity is gateway-controlled, so a tenant cannot forge it.
        if (getattr(tenant_context, "user_email", "") or "").strip().lower() == "internal@shielva.ai":
            return True

        required_perms = definition.requires_permissions

        if not required_perms:
            return True

        user_perms = set(tenant_context.permissions)

        return all(perm in user_perms for perm in required_perms)

    def _build_parameters_schema(self, definition: ToolDefinition) -> dict[str, Any]:
        """Build JSON schema for tool parameters."""
        properties = {}
        required = []

        for param in definition.parameters:
            param_schema = {"type": param.type, "description": param.description}

            # Add items field for array types (required by Gemini)
            if param.type == "array" and param.items:
                param_schema["items"] = param.items

            # Add default value if present
            if param.default is not None:
                param_schema["default"] = param.default

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {"type": "object", "properties": properties, "required": required}

    # ===== Built-in Tools =====

    async def rag_query_tool(
        self,
        tenant_context: TenantContext,
        query: str,
        kb_ids: list[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Query knowledge bases for relevant information.
        """
        if not self.rag_client:
            logger.warning("RAG client not initialized in ToolRegistry")
            return {"results": [], "error": "RAG engine unavailable"}

        try:
            # Call RAG engine
            results = await self.rag_client.retrieve(
                query=query,
                tenant_id=tenant_context.tenant_id,
                kb_ids=kb_ids or [],
                top_k=top_k,
                rerank=True,
            )

            # Format results for LLM
            formatted_results = []
            for res in results:
                formatted_results.append(
                    {
                        "content": res.content,
                        "metadata": res.metadata,
                        "score": res.score,
                        "source": res.metadata.get("source", "Unknown") if res.metadata else "Unknown",
                        "author": res.metadata.get("author", "Unknown") if res.metadata else "Unknown",
                    }
                )

            return {
                "results": formatted_results,
                "query": query,
                "count": len(formatted_results),
            }
        except Exception as e:
            logger.error("RAG tool execution failed", error=str(e))
            return {"results": [], "error": str(e)}


async def current_time_tool(tenant_context: TenantContext) -> dict[str, Any]:
    """Get current date and time."""
    from datetime import datetime

    now = datetime.utcnow()
    return {"utc_time": now.isoformat(), "timestamp": now.timestamp()}


# Default tool definitions
DEFAULT_TOOLS = [
    (
        ToolDefinition(
            name="rag_query",
            description="Search the knowledge base for relevant information. Use this when you need to find specific information from documents.",
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "description": "The search query",
                    "required": True,
                },
                {
                    "name": "kb_ids",
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional specific KB IDs to search",
                },
                {
                    "name": "top_k",
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
            ],
            requires_permissions=["rag_access"],
            enabled_by_default=True,
        ),
        None,  # Handler will be resolved via ToolRegistry methods
    ),
    (
        ToolDefinition(
            name="get_current_time",
            description="Get the current date and time in UTC",
            parameters=[],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        current_time_tool,
    ),
]


def create_registry_with_defaults() -> ToolRegistry:
    """Create a tool registry with default tools."""
    registry = ToolRegistry()

    for definition, handler in DEFAULT_TOOLS:
        if definition.name == "rag_query":
            # Link to the instance method
            registry.register(definition, registry.rag_query_tool)
        else:
            registry.register(definition, handler)

    return registry
