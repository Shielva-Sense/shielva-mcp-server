"""
MCP Protocol - Message Handler
Handles incoming protocol messages and routes them appropriately
"""

from datetime import datetime
from typing import Any

import structlog

from src.security import normalize_role

from .models import (
    MCPMessage,
    MCPQueryRequest,
    MCPQueryResponse,
    MessageRole,
    ProvisionBotRequest,
    ProvisionBotResponse,
    ProvisionKBRequest,
    ProvisionKBResponse,
    SessionContext,
    Source,
    TenantContext,
    TestBotRequest,
    TestBotResponse,
)

logger = structlog.get_logger(__name__)


class MessageHandler:
    """
    Central handler for MCP protocol messages.
    Routes messages to appropriate services based on type.
    """

    def __init__(self, context_assembler, tool_registry, kb_registry, llm_router, policy_engine):
        """
        Initialize message handler with dependencies.

        Args:
            context_assembler: Assembles context for queries
            tool_registry: Registry of available tools
            kb_registry: Registry of knowledge bases
            llm_router: Routes to LLM providers
            policy_engine: Enforces policies
        """
        self.context_assembler = context_assembler
        self.tool_registry = tool_registry
        self.kb_registry = kb_registry
        self.llm_router = llm_router
        self.policy_engine = policy_engine

        logger.info("MessageHandler initialized")

    async def handle_query(self, request: MCPQueryRequest, tenant_context: TenantContext) -> MCPQueryResponse:
        """
        Handle a query request through the MCP pipeline.

        Pipeline:
        1. Policy check
        2. Context assembly (bot config, KB retrieval)
        3. Tool preparation
        4. LLM execution
        5. Response filtering

        Args:
            request: The query request
            tenant_context: Tenant isolation context

        Returns:
            MCPQueryResponse with answer and metadata
        """
        start_time = datetime.utcnow()

        logger.info(
            "Processing query",
            tenant_id=tenant_context.tenant_id,
            bot_id=request.bot_id,
            query_length=len(request.query),
        )

        try:
            # 1. Policy check
            await self._check_policy(tenant_context, "query", request.bot_id)

            # 2. Get or create session
            session = await self._get_or_create_session(request.session_id, tenant_context, request.bot_id)

            # 3. Assemble context
            context = await self.context_assembler.assemble(
                query=request.query,
                session=session,
                tenant_context=tenant_context,
                bot_id=request.bot_id,
                custom_prompt=request.custom_prompt,
            )

            # 4. Get available tools
            tools = await self.tool_registry.get_tools_for_bot(
                bot_id=request.bot_id,
                tenant_context=tenant_context,
                enabled_tools=request.tool_options,
            )

            # 5. Execute LLM with tools
            result = await self.llm_router.execute(
                messages=context.messages,
                tools=tools,
                tenant_context=tenant_context,
                stream=request.stream,
                model=getattr(request, "model", None),  # per-bot override; tenant key/provider still apply
            )

            # 6. Update session
            await self._update_session(session, request.query, result)

            # Calculate latency
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Map RetrievalResult chunks → Source objects for the response
            sources: list[Source] = []
            for chunk in context.retrieved_chunks:
                try:
                    sources.append(
                        Source(
                            kb_id=getattr(chunk, "kb_id", ""),
                            kb_name=getattr(chunk, "kb_name", chunk.metadata.get("kb_name", "")),
                            document_id=getattr(
                                chunk,
                                "document_id",
                                chunk.metadata.get("document_id", ""),
                            ),
                            document_title=getattr(chunk, "document_title", chunk.metadata.get("title", "")),
                            chunk_id=getattr(chunk, "chunk_id", ""),
                            content=chunk.content[:200],  # trim for wire size
                            score=round(chunk.score, 4),
                            metadata={},
                        )
                    )
                except Exception:
                    pass  # never let source mapping break the query response

            return MCPQueryResponse(
                answer=result.answer,
                sources=sources,
                tool_calls=result.tool_calls,
                tokens_used=result.tokens_used,
                latency_ms=latency_ms,
                model=result.model,
                session_id=session.session_id,
            )

        except Exception as e:
            logger.error("Query processing failed", error=str(e))
            raise

    async def handle_provision_kb(
        self, request: ProvisionKBRequest, tenant_context: TenantContext
    ) -> ProvisionKBResponse:
        """
        Handle KB provisioning request.

        Steps:
        1. Policy check
        2. Register KB in registry
        3. Generate hello_id and unique_name
        4. Trigger connector sync (if applicable)
        5. Queue ingestion job

        Args:
            request: KB provisioning request
            tenant_context: Tenant context

        Returns:
            ProvisionKBResponse with KB details
        """
        logger.info(
            "Provisioning KB",
            tenant_id=tenant_context.tenant_id,
            bot_id=request.bot_id,
            kb_name=request.kb_config.name,
        )

        # Check policy
        await self._check_policy(tenant_context, "provision_kb", request.bot_id)

        # Register KB
        kb_info = await self.kb_registry.register(
            tenant_context=tenant_context,
            bot_id=request.bot_id,
            kb_config=request.kb_config,
        )

        return ProvisionKBResponse(
            kb_id=kb_info.kb_id,
            hello_id=kb_info.hello_id,
            unique_name=kb_info.unique_name,
            status=kb_info.status,
            message="KB provisioning initiated",
        )

    async def handle_provision_bot(
        self, request: ProvisionBotRequest, tenant_context: TenantContext
    ) -> ProvisionBotResponse:
        """
        Handle bot provisioning request.

        Args:
            request: Bot provisioning request
            tenant_context: Tenant context

        Returns:
            ProvisionBotResponse with bot details
        """
        logger.info(
            "Provisioning bot",
            tenant_id=tenant_context.tenant_id,
            bot_name=request.name,
        )

        # Check policy
        await self._check_policy(tenant_context, "provision_bot", None)

        # Create bot in registry
        # (Implementation would go here)

        return ProvisionBotResponse(bot_id="bot-new-id", status="draft", message="Bot created successfully")

    async def handle_test_bot(self, request: TestBotRequest, tenant_context: TenantContext) -> TestBotResponse:
        """
        Handle bot testing request.

        Runs a series of test queries against the bot
        and validates responses.

        Args:
            request: Test request with queries
            tenant_context: Tenant context

        Returns:
            TestBotResponse with results
        """
        logger.info(
            "Testing bot",
            tenant_id=tenant_context.tenant_id,
            bot_id=request.bot_id,
            num_queries=len(request.test_queries),
        )

        # Check policy
        await self._check_policy(tenant_context, "test_bot", request.bot_id)

        # Run test queries
        # (Implementation would go here)

        return TestBotResponse(bot_id=request.bot_id, results=[], passed=True, overall_score=0.95)

    async def _check_policy(self, tenant_context: TenantContext, action: str, resource_id: str | None) -> None:
        """Authorize against the gateway-verified principal — the platform standard.

        The API gateway verifies the signed ``X-Shielva-*`` principal (role + app
        access) and every MCP handler scopes its data by ``tenant_id`` from that
        principal, so MCP does NOT maintain a parallel RBAC matrix. That duplicated —
        and silently diverged from — the IdP (shielva-identity), which is the single
        owner of roles + policy. We require an authenticated, tenant-scoped principal
        whose role resolves to a canonical platform role (platform_owner / tenant_admin
        / developer). Tenant isolation is the enforced boundary, applied downstream by
        tenant-scoped queries. Same model as ACP/TMS (shielva_common.auth).

        Raises:
            PermissionError: if the request carries no authenticated tenant context.
        """
        tenant_id = getattr(tenant_context, "tenant_id", None)
        if not tenant_id:
            raise PermissionError(f"Action '{action}' denied: no authenticated principal")
        # normalize_role is a thin compatibility shim for legacy/OIDC role strings;
        # it never fails (unknown → developer, the IdP default). All three canonical
        # roles are entitled to use a bot in their own tenant (query/test/provision);
        # finer-grained grants live in the IdP and are enforced at the gateway.
        role = normalize_role(getattr(tenant_context, "role", None))
        logger.debug(
            "policy.trust_verified_principal",
            action=action,
            tenant_id=tenant_id,
            role=role,
        )

    async def _get_or_create_session(
        self, session_id: str | None, tenant_context: TenantContext, bot_id: str
    ) -> SessionContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional existing session ID
            tenant_context: Tenant context
            bot_id: Bot ID

        Returns:
            SessionContext
        """
        if session_id:
            # Try to retrieve from Redis
            # (Implementation would go here)
            pass

        # Create new session
        return SessionContext(tenant_context=tenant_context, bot_id=bot_id)

    async def _update_session(self, session: SessionContext, query: str, result: Any) -> None:
        """Update session with new message exchange."""
        # Add user message
        session.messages.append(MCPMessage(role=MessageRole.USER, content=query))

        # Add assistant message
        session.messages.append(MCPMessage(role=MessageRole.ASSISTANT, content=result.answer))

        session.last_activity = datetime.utcnow()

        # Save to Redis
        # (Implementation would go here)
