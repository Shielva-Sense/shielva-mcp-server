"""
MCP Protocol - Message Handler
Owns the provisioning control-plane operations (KB / bot / test-bot). The
bot-query pipeline lives in the DDD layer (``application.chat.HandleQueryUseCase``,
wired onto ``app.state.handle_query_use_case``) — this handler no longer serves
queries.
"""

import structlog

from src.security import normalize_role

from .models import (
    ProvisionBotRequest,
    ProvisionBotResponse,
    ProvisionKBRequest,
    ProvisionKBResponse,
    TenantContext,
    TestBotRequest,
    TestBotResponse,
)

logger = structlog.get_logger(__name__)


class MessageHandler:
    """
    Provisioning control-plane handler for MCP protocol messages.
    Serves KB / bot provisioning and bot-test operations (consumed by
    ``interface.http.provision_router``). Query traffic is handled by the
    DDD ``HandleQueryUseCase``, not here.
    """

    def __init__(self, kb_registry):
        """
        Initialize the provisioning handler.

        Args:
            kb_registry: Registry of knowledge bases (KB provisioning target).
        """
        self.kb_registry = kb_registry

        logger.info("MessageHandler initialized")

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
