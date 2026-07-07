"""JSON-RPC method dispatcher.

The dispatcher knows the spec method names and routes each to the
right application service. It is **the only place** that knows the
mapping ``"tools/list" → application.tools.list_tools`` etc.

Why the dispatcher is in the interface layer (not application):
    * The set of spec methods is a property of the *protocol* we
      speak (MCP). If we added a different inbound protocol (e.g.,
      gRPC) it would have its own dispatcher with its own method
      names — but it'd call the same application services.
    * Method names like ``"tools/list"`` are protocol vocabulary,
      not domain vocabulary.

Until later slices land, the tools/resources/prompts handlers still
call ``app_state.tool_registry`` etc. directly. The dispatcher keeps
that coupling localised — the application/chat layer is already
clean, and the rest will follow context-by-context.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.application.chat import ChatApplicationService
from src.application.tools import ToolApplicationService
from src.domain.chat.errors import SessionNotFoundError, SessionStateError
from src.domain.chat.value_objects import ClientInfo, SessionId
from src.domain.shared.tenant import TenantContext
from src.domain.tools.errors import (
    ToolNotFoundError,
    ToolPermissionDeniedError,
)
from src.domain.tools.value_objects import ToolImage as DomainToolImage
from src.domain.tools.value_objects import ToolText as DomainToolText

from .errors import (
    FORBIDDEN,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    RESOURCE_NOT_FOUND,
    TOOL_NOT_FOUND,
    JsonRpcException,
)
from .types import (
    Implementation,
    InitializeParams,
    InitializeResult,
    LoggingSetLevelParams,
    Prompt,
    PromptArgument,
    PromptGetResult,
    PromptListResult,
    PromptMessage,
    Resource,
    ResourceContents,
    ResourceListResult,
    ResourceReadResult,
    ServerCapabilities,
    ToolCallParams,
    ToolCallResult,
    ToolContentText,
    ToolListItem,
    ToolListResult,
)

logger = structlog.get_logger(__name__)


# Spec version we implement (2024-11-05 stable; Streamable HTTP from
# 2025-03-26 is wire-compatible).
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "shielva-mcp"


class MCPDispatcher:
    """Routes one JSON-RPC method call to the right handler."""

    def __init__(
        self,
        *,
        chat_service: ChatApplicationService,
        server_version: str,
    ) -> None:
        self._chat = chat_service
        self._server_version = server_version

        self._requests = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "logging/setLevel": self._handle_logging_set_level,
        }
        self._notifications = {
            "notifications/initialized": self._handle_initialized,
            "notifications/cancelled": self._handle_cancelled,
        }

    # ── entry point ────────────────────────────────────────────────

    async def dispatch_request(
        self,
        *,
        method: str,
        params: dict[str, Any] | None,
        tenant: TenantContext,
        session_id: str | None,
        app_state: Any,
    ) -> tuple[dict[str, Any], str | None]:
        """Handle a JSON-RPC request that expects a response.

        Returns ``(result, session_id_for_response_header)``. Raises
        :class:`JsonRpcException` on any application error so the
        transport can shape the JSON-RPC error envelope.
        """
        handler = self._requests.get(method)
        if handler is None:
            raise JsonRpcException(METHOD_NOT_FOUND, f"Unknown method: {method}")

        # Spec: every non-initialize, non-ping request requires a
        # session id (transport already enforced HTTP-level missing-
        # header → 400). State-machine check happens per-method.
        sid: SessionId | None = SessionId(session_id) if session_id else None

        if method == "initialize":
            result, new_sid = await handler(tenant=tenant, params=params or {}, app_state=app_state)
            return result, str(new_sid) if new_sid else None

        if method == "ping":
            result = await handler(tenant=tenant, session_id=sid, params=params or {}, app_state=app_state)
            return result, session_id

        if sid is None:
            raise JsonRpcException(
                INVALID_REQUEST,
                "Mcp-Session-Id header required (other than initialize)",
            )
        # Enforce state machine for non-initialize methods.
        await self._chat.assert_method_allowed(
            session_id=sid,
            tenant=tenant,
            method=method,
        )
        result = await handler(tenant=tenant, session_id=sid, params=params or {}, app_state=app_state)
        return result, session_id

    async def dispatch_notification(
        self,
        *,
        method: str,
        params: dict[str, Any] | None,
        tenant: TenantContext,
        session_id: str | None,
        app_state: Any,
    ) -> None:
        """Handle a notification (no id, no response).

        Per spec: unknown notifications are silently ignored.
        """
        handler = self._notifications.get(method)
        if handler is None:
            logger.debug("mcp.unknown_notification", method=method)
            return
        sid = SessionId(session_id) if session_id else None
        await handler(tenant=tenant, session_id=sid, params=params or {}, app_state=app_state)

    # ── request handlers ───────────────────────────────────────────

    async def _handle_initialize(
        self, *, tenant: TenantContext, params: dict[str, Any], app_state: Any
    ) -> tuple[dict[str, Any], SessionId]:
        try:
            ip = InitializeParams.model_validate(params)
        except Exception as e:
            raise JsonRpcException(INVALID_PARAMS, f"Invalid initialize: {e}")

        client_info = ClientInfo(
            name=ip.clientInfo.name,
            version=ip.clientInfo.version,
        )
        session = await self._chat.initialize(
            tenant=tenant,
            client_info=client_info,
            client_protocol_version=ip.protocolVersion,
        )

        caps = ServerCapabilities(
            tools={"listChanged": False},
            resources={"subscribe": False, "listChanged": False},
            prompts={"listChanged": False},
            logging={},
        )
        result = InitializeResult(
            protocolVersion=str(session.protocol_version),
            capabilities=caps,
            serverInfo=Implementation(name=SERVER_NAME, version=self._server_version),
            instructions=(
                "shielva-mcp tenant-scoped RAG/LLM server. Tools, "
                "resources (knowledge bases), and prompts (bot "
                "templates) are scoped to the X-Tenant-ID supplied "
                "with each request."
            ),
        ).model_dump(exclude_none=True)
        return result, session.id

    async def _handle_ping(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId | None,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        await self._chat.ping(session_id=session_id, tenant=tenant)
        return {}

    async def _handle_logging_set_level(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        try:
            p = LoggingSetLevelParams.model_validate(params)
        except Exception as e:
            raise JsonRpcException(INVALID_PARAMS, f"Invalid logging/setLevel: {e}")
        await self._chat.set_log_level(
            session_id=session_id,
            tenant=tenant,
            level=p.level,
        )
        return {}

    # ── tools/* — goes through the domain port (slice 2) ──────────

    def _resolve_tool_service(self, app_state: Any) -> ToolApplicationService:
        """The tool application service is constructed in the
        FastAPI lifespan (after the legacy registry is populated by
        registration hooks) and stashed on ``app.state``. The
        dispatcher itself stays stateless — it pulls the wired
        service per request so test code can swap it without
        rebuilding the dispatcher singleton."""
        svc = getattr(app_state, "tool_app_service", None)
        if svc is None:
            raise JsonRpcException(
                INTERNAL_ERROR,
                "tool_app_service not wired — check lifespan composition",
            )
        return svc

    async def _handle_tools_list(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        service = self._resolve_tool_service(app_state)
        tools = await service.list_tools(tenant=tenant)
        items = [
            ToolListItem(
                name=str(t.name),
                description=t.description,
                inputSchema=t.input_schema.json_schema,
            )
            for t in tools
        ]
        return ToolListResult(tools=items).model_dump(exclude_none=True)

    async def _handle_tools_call(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        try:
            tcp = ToolCallParams.model_validate(params)
        except Exception as e:
            raise JsonRpcException(INVALID_PARAMS, f"Invalid tools/call: {e}")

        service = self._resolve_tool_service(app_state)
        try:
            result = await service.execute_tool(
                tenant=tenant,
                name=tcp.name,
                arguments=dict(tcp.arguments or {}),
                context={"mcp_session_id": str(session_id)},
            )
        except ToolNotFoundError:
            raise JsonRpcException(TOOL_NOT_FOUND, f"Tool not found: {tcp.name}")
        except ToolPermissionDeniedError as e:
            raise JsonRpcException(FORBIDDEN, str(e))

        # Translate the domain ToolResult (text|image content blocks
        # + is_error flag) into the spec's JSON envelope.
        content_dto = []
        for block in result.content:
            if isinstance(block, DomainToolText):
                content_dto.append(ToolContentText(text=block.text))
            elif isinstance(block, DomainToolImage):
                from .types import ToolContentImage

                content_dto.append(
                    ToolContentImage(
                        data=block.data,
                        mimeType=block.mimeType,
                    )
                )
            else:  # pragma: no cover — extra block types arrive in spec 2025-03-26
                content_dto.append(ToolContentText(text=str(block)))

        return ToolCallResult(
            content=content_dto,
            isError=result.is_error,
        ).model_dump(exclude_none=True)

    # ── resources/* — goes through KnowledgeApplicationService ────

    def _resolve_knowledge_service(self, app_state: Any):
        svc = getattr(app_state, "knowledge_app_service", None)
        if svc is None:
            raise JsonRpcException(
                INTERNAL_ERROR,
                "knowledge_app_service not wired — check lifespan composition",
            )
        return svc

    async def _handle_resources_list(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        service = self._resolve_knowledge_service(app_state)
        kbs = await service.list_knowledge_bases(tenant=tenant)
        resources = [
            Resource(
                uri=f"kb://{tenant.tenant_id}/{kb.id}",
                name=str(kb.name) or str(kb.id),
                description=None,
                mimeType="application/vnd.shielva.kb+json",
            )
            for kb in kbs
        ]
        return ResourceListResult(resources=resources).model_dump(exclude_none=True)

    async def _handle_resources_read(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        uri = (params or {}).get("uri") or ""
        if not isinstance(uri, str) or not uri.startswith("kb://"):
            raise JsonRpcException(INVALID_PARAMS, "uri must be a kb:// URI")
        _, _, rest = uri.partition("kb://")
        tenant_id, _, kb_id = rest.partition("/")
        if tenant_id != tenant.tenant_id:
            raise JsonRpcException(RESOURCE_NOT_FOUND, "Resource not in this tenant scope")

        service = self._resolve_knowledge_service(app_state)
        from src.domain.knowledge.errors import KnowledgeBaseNotFoundError

        try:
            kb = await service.read_knowledge_base(kb_id=kb_id, tenant=tenant)
        except KnowledgeBaseNotFoundError:
            raise JsonRpcException(RESOURCE_NOT_FOUND, f"KB not found: {kb_id}")

        envelope = {
            "kb_id": str(kb.id),
            "name": str(kb.name),
            "status": kb.status.value,
            "documents": kb.doc_count,
            "hint": (
                "Retrieve KB content via the 'rag_query' tool. The full "
                "chunk set is not exposed as a single resource because "
                "typical KBs contain tens of thousands of chunks."
            ),
        }
        return ResourceReadResult(
            contents=[
                ResourceContents(
                    uri=uri,
                    mimeType="application/vnd.shielva.kb+json",
                    text=json.dumps(envelope, ensure_ascii=False),
                )
            ],
        ).model_dump(exclude_none=True)

    # ── prompts/* — goes through BotApplicationService ────────────

    def _resolve_bot_service(self, app_state: Any):
        svc = getattr(app_state, "bot_app_service", None)
        if svc is None:
            raise JsonRpcException(
                INTERNAL_ERROR,
                "bot_app_service not wired — check lifespan composition",
            )
        return svc

    async def _handle_prompts_list(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        # Bot listing is best-effort — the legacy registry has no
        # tenant-wide enumeration (it's bot-by-id), so the typical
        # response is an empty array. That's spec-conformant; clients
        # discover prompts by other means (config, docs).
        service = self._resolve_bot_service(app_state)
        bots = await service.list_bots(tenant=tenant)
        prompts = [
            Prompt(
                name=f"bot/{bot.id}",
                description=bot.description or f"Bot prompt template for {bot.name}",
                arguments=[PromptArgument(name=v.name, required=v.required) for v in bot.prompt_template.variables],
            )
            for bot in bots
        ]
        return PromptListResult(prompts=prompts).model_dump(exclude_none=True)

    async def _handle_prompts_get(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId,
        params: dict[str, Any],
        app_state: Any,
    ) -> dict[str, Any]:
        name = (params or {}).get("name") or ""
        if not name.startswith("bot/"):
            raise JsonRpcException(INVALID_PARAMS, "Prompt name must be 'bot/<bot_id>'")
        bot_id = name[len("bot/") :]
        service = self._resolve_bot_service(app_state)
        from src.domain.bots.errors import BotNotFoundError

        try:
            rendered = await service.render_prompt(
                bot_id=bot_id,
                tenant=tenant,
                arguments=(params or {}).get("arguments") or {},
            )
            bot = await service.get_bot(bot_id=bot_id, tenant=tenant)
        except BotNotFoundError:
            raise JsonRpcException(RESOURCE_NOT_FOUND, f"Bot not found: {bot_id}")
        return PromptGetResult(
            description=bot.description or None,
            messages=[
                PromptMessage(role="user", content=ToolContentText(text=rendered)),
            ],
        ).model_dump(exclude_none=True)

    # ── notification handlers ──────────────────────────────────────

    async def _handle_initialized(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId | None,
        params: dict[str, Any],
        app_state: Any,
    ) -> None:
        if session_id is None:
            return  # malformed; spec says drop silently
        try:
            await self._chat.mark_initialized(session_id=session_id, tenant=tenant)
        except SessionNotFoundError:
            logger.warning("mcp.initialized_unknown_session", session_id=str(session_id))
        except SessionStateError as e:
            logger.warning("mcp.initialized_bad_state", error=str(e))

    async def _handle_cancelled(
        self,
        *,
        tenant: TenantContext,
        session_id: SessionId | None,
        params: dict[str, Any],
        app_state: Any,
    ) -> None:
        # v1: log only. v2 will propagate cancellation into in-
        # flight tool calls via an asyncio.Event keyed on requestId.
        logger.info(
            "mcp.cancelled_notification",
            session_id=str(session_id) if session_id else None,
            request_id=(params or {}).get("requestId"),
            reason=(params or {}).get("reason"),
        )


# All bot/kb/legacy bridge helpers removed — resources/* + prompts/*
# now go through the application layer (slices 3 + 4a).
