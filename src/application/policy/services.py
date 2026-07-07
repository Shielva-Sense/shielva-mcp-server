"""Policy application service."""

from __future__ import annotations

from typing import Any

import structlog

from src.domain.policy.errors import PolicyDeniedError
from src.domain.policy.repositories import PolicyEngine
from src.domain.policy.value_objects import (
    FeatureDecision,
    PolicyDecision,
    PolicyRequest,
    QuotaDecision,
)
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


class PolicyApplicationService:
    def __init__(self, *, engine: PolicyEngine) -> None:
        self._engine = engine

    async def require_permission(
        self,
        *,
        tenant: TenantContext,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Raise :class:`PolicyDeniedError` on deny."""
        request = PolicyRequest(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            user_role=tenant.role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata or {},
        )
        decision = await self._engine.evaluate(request)
        if not decision.allowed:
            logger.warning(
                "mcp.policy_denied",
                tenant_id=tenant.tenant_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                reason=decision.reason,
            )
            raise PolicyDeniedError(f"{action} on {resource_type} denied: {decision.reason or 'no reason given'}")
        return decision

    async def check_quota(self, *, tenant: TenantContext, resource_type: str, period: str = "monthly") -> QuotaDecision:
        return await self._engine.check_quota(
            tenant_id=tenant.tenant_id,
            resource_type=resource_type,
            period=period,
        )

    async def check_feature(self, *, tenant: TenantContext, feature: str) -> FeatureDecision:
        return await self._engine.check_feature(
            tenant_id=tenant.tenant_id,
            feature=feature,
        )
